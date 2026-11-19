"""Penjelasan grafik dan hasil prediksi memakai Groq API (dipanggil dari views)."""
import os
import re
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# .env berada di folder utama proyek (sejajar dengan manage.py)
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

ML_DIR = Path(__file__).resolve().parent / "ml"
TIMEOUT = 25  # detik

# Model utama & daftar cadangan bisa diatur lewat .env tanpa mengubah kode
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_FALLBACK_MODELS = [
    m.strip()
    for m in os.getenv(
        "GROQ_FALLBACK_MODELS",
        "openai/gpt-oss-20b,qwen/qwen3-32b,llama-3.1-8b-instant",
    ).split(",")
    if m.strip()
]

# Cache jawaban sukses agar hemat kuota API
_cache = {}
_cache_prediksi = {}

# Pelacakan status per model agar bisa otomatis berpindah saat satu model limit/404
_limit_per_model = {}        # model_id -> timestamp reset
_model_tidak_tersedia = set()  # model_id yang mengembalikan 404
_daftar_model_cache = []     # hasil GET /openai/v1/models


def _muat_dataset_unik():
    return pd.read_csv(ML_DIR / "bestsellers.csv").drop_duplicates(subset="Name")


def ringkasan_data():
    """Statistik harga per genre, dihitung dari 351 buku unik (sama dengan data latih)."""
    df = _muat_dataset_unik()
    stat = df.groupby("Genre")["Price"].agg(["count", "mean", "median", "std", "min", "max"])
    return stat.round(2).to_dict("index")


def ringkasan_fitur_genre():
    """Rata-rata dan median tiap fitur per genre sebagai konteks penjelasan prediksi."""
    df = _muat_dataset_unik()
    fitur = ["User Rating", "Reviews", "Price", "Year"]
    baris = []
    for genre, grup in df.groupby("Genre"):
        info = [f"{f}: rata-rata {grup[f].mean():.1f} (median {grup[f].median():.1f})" for f in fitur]
        baris.append(f"- {genre}: " + ", ".join(info))
    return "\n".join(baris)


def buat_prompt(ringkasan):
    baris = "\n".join(
        f"- {genre}: {int(d['count'])} buku, rata-rata ${d['mean']}, median ${d['median']}, "
        f"simpangan baku ${d['std']}, harga ${d['min']} sampai ${d['max']}"
        for genre, d in ringkasan.items()
    )
    return (
        "Kamu asisten analisis data. Berikut statistik harga buku bestseller Amazon "
        "(buku unik, 2009-2019) per genre:\n"
        f"{baris}\n\n"
        "Tulis penjelasan singkat dalam bahasa Indonesia, 3 sampai 4 kalimat, tanpa markdown "
        "dan tanpa daftar, tentang grafik batang rata-rata harga per genre. "
        "Sebutkan genre mana yang rata-rata harganya lebih tinggi beserta selisihnya, "
        "bandingkan dengan median, dan beri satu peringatan bahwa selisih rata-rata belum tentu "
        "bermakna karena sebaran harga lebar. Hanya gunakan angka di atas dan jangan mengarang data."
    )


def buat_prompt_prediksi(data_buku, hasil, konteks_fitur):
    return (
        "Kamu asisten toko buku yang menjelaskan hasil prediksi machine learning kepada "
        "pengguna awam dalam bahasa Indonesia yang sederhana, hangat, dan mudah dipahami.\n\n"
        f"Data buku yang dimasukkan pengguna:\n"
        f"- User Rating: {data_buku['user_rating']} dari 5.0\n"
        f"- Jumlah Reviews: {data_buku['reviews']}\n"
        f"- Harga: ${data_buku['price']}\n"
        f"- Tahun masuk bestseller: {data_buku['year']}\n\n"
        f"Hasil Prediksi Model: {hasil['label']} (tingkat keyakinan {hasil['keyakinan']}%)\n"
        "Faktor paling berpengaruh di model: Jumlah Reviews (~40%), User Rating (~27%), "
        "Tahun (~18%), Harga (~15%).\n"
        f"Referensi rata-rata dataset per genre:\n{konteks_fitur}\n\n"
        "Tulis penjelasan singkat 3 sampai 4 kalimat tanpa markdown dan tanpa daftar:\n"
        "1. Jelaskan arti hasil prediksi dan persentase keyakinan tersebut secara sederhana.\n"
        "2. Jelaskan mengapa angka buku yang dimasukkan cenderung mengarah ke kategori itu dibanding rata-rata dataset.\n"
        "3. Beri satu pengingat singkat bahwa akurasi model sekitar 66% sehingga hasil ini perkiraan, bukan kepastian."
    )


def _parse_durasi_detik(nilai, default=60.0):
    """Ubah header 'retry-after' atau durasi Groq (mis. '1m30s', '45s', '500ms') ke detik."""
    if not nilai:
        return default
    teks = str(nilai).strip().lower()
    try:
        return max(1.0, float(teks))
    except ValueError:
        pass
    total = 0.0
    for angka, satuan in re.findall(r"(\d+(?:\.\d+)?)(ms|h|m|s)", teks):
        val = float(angka)
        if satuan == "h":
            total += val * 3600
        elif satuan == "m":
            total += val * 60
        elif satuan == "s":
            total += val
        elif satuan == "ms":
            total += val / 1000.0
    return max(1.0, total) if total > 0 else default


def _tandai_limit_model(model, durasi_detik):
    _limit_per_model[model] = time.time() + max(1.0, durasi_detik)


def _sisa_limit_model(model):
    reset_pada = _limit_per_model.get(model)
    if not reset_pada:
        return 0
    sisa = reset_pada - time.time()
    if sisa <= 0:
        _limit_per_model.pop(model, None)
        return 0
    return max(1, int(sisa))


def _perbarui_limit_dari_header(model, headers):
    """Pantau sisa kuota per model dari header respons Groq."""
    if not headers:
        _limit_per_model.pop(model, None)
        return
    rem_req = headers.get("x-ratelimit-remaining-requests")
    rem_tok = headers.get("x-ratelimit-remaining-tokens")
    if rem_req is not None and str(rem_req).isdigit() and int(rem_req) <= 0:
        detik = _parse_durasi_detik(headers.get("x-ratelimit-reset-requests"), default=60.0)
        _tandai_limit_model(model, detik)
        return
    if rem_tok is not None and str(rem_tok).isdigit() and int(rem_tok) < 150:
        detik = _parse_durasi_detik(headers.get("x-ratelimit-reset-tokens"), default=60.0)
        _tandai_limit_model(model, detik)
        return
    _limit_per_model.pop(model, None)


def _semua_kandidat_model():
    """Daftar unik model berurutan: utama -> cadangan -> hasil penemuan API."""
    urutan = [GROQ_MODEL] + GROQ_FALLBACK_MODELS + _daftar_model_cache
    unik = []
    for m in urutan:
        if m and m not in unik and m not in _model_tidak_tersedia:
            unik.append(m)
    return unik


def status_groq():
    """Kembalikan status ketersediaan Groq saat ini (limit hanya jika semua model penuh)."""
    if not os.getenv("GROQ_API_KEY"):
        return {
            "limit": True,
            "alasan": "key_kosong",
            "sisa_detik": 0,
            "pesan": (
                "GROQ_API_KEY belum diisi di file .env. Penjelasan AI tidak aktif, "
                "namun Anda tetap bisa melakukan prediksi kategori seperti biasa."
            ),
        }

    kandidat = _semua_kandidat_model()
    if not kandidat:
        return {"limit": False, "alasan": "", "sisa_detik": 0, "pesan": ""}

    sisa_list = [_sisa_limit_model(m) for m in kandidat]
    if all(s > 0 for s in sisa_list):
        detik = min(sisa_list)
        return {
            "limit": True,
            "alasan": "rate_limit",
            "sisa_detik": detik,
            "pesan": (
                f"Semua model Groq sedang mencapai batas limit (coba lagi dalam ~{detik} detik). "
                "Penjelasan AI sementara tidak tersedia, tetapi hasil prediksi kategori tetap berjalan normal."
            ),
        }

    return {"limit": False, "alasan": "", "sisa_detik": 0, "pesan": ""}


def _muat_daftar_model_groq(kunci):
    """Ambil daftar model chat aktif di akun Groq dan simpan ke cache."""
    global _daftar_model_cache
    r = requests.get(
        "https://api.groq.com/openai/v1/models",
        headers={"Authorization": f"Bearer {kunci}"},
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    kandidat = []
    for item in r.json().get("data", []):
        mid = item.get("id", "")
        low = mid.lower()
        if not mid or any(x in low for x in ("whisper", "tts", "guard", "embed", "moderation")):
            continue
        kandidat.append(mid)

    prioritas = ("openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen", "llama")
    terurut = []
    for pref in prioritas:
        for mid in kandidat:
            if pref in mid.lower() and mid not in terurut:
                terurut.append(mid)
    for mid in kandidat:
        if mid not in terurut:
            terurut.append(mid)
    _daftar_model_cache = terurut
    return terurut


def _kirim_chat_groq(kunci, model, prompt):
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {kunci}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 500,
        },
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    _perbarui_limit_dari_header(model, r.headers)
    return r.json()["choices"][0]["message"]["content"].strip(), model


def tanya_groq(prompt):
    kunci = os.getenv("GROQ_API_KEY")
    if not kunci:
        raise RuntimeError("GROQ_API_KEY belum diisi di file .env")

    sudah_dicoba = set()
    sudah_cari_model = False
    error_terakhir = None

    while True:
        kandidat = [
            m for m in _semua_kandidat_model()
            if m not in sudah_dicoba and _sisa_limit_model(m) == 0
        ]
        if not kandidat:
            if not sudah_cari_model:
                sudah_cari_model = True
                try:
                    _muat_daftar_model_groq(kunci)
                    continue
                except Exception:
                    pass
            break

        model = kandidat[0]
        sudah_dicoba.add(model)
        try:
            return _kirim_chat_groq(kunci, model, prompt)
        except requests.HTTPError as e:
            error_terakhir = e
            resp = e.response
            kode = resp.status_code if resp is not None else None
            if kode == 404:
                _model_tidak_tersedia.add(model)
                if not sudah_cari_model:
                    sudah_cari_model = True
                    try:
                        _muat_daftar_model_groq(kunci)
                    except Exception:
                        pass
                continue
            if kode == 429:
                detik = _parse_durasi_detik(
                    resp.headers.get("retry-after")
                    or resp.headers.get("x-ratelimit-reset-requests")
                    or resp.headers.get("x-ratelimit-reset-tokens"),
                    default=60.0,
                )
                _tandai_limit_model(model, detik)
                continue
            raise

    if error_terakhir is not None:
        raise error_terakhir
    raise RuntimeError(status_groq()["pesan"] or "Tidak ada model Groq yang tersedia saat ini.")


def _detail_http_error(resp):
    """Ambil pesan error singkat dari respons API tanpa membocorkan kredensial."""
    if resp is None:
        return ""
    try:
        err = resp.json().get("error", {})
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])[:180]
    except Exception:
        pass
    return ""


def _jalankan(fungsi, model, prompt):
    """Jalankan satu pemanggilan API; kegagalan dikembalikan sebagai pesan, bukan error."""
    try:
        hasil = fungsi(prompt)
        if isinstance(hasil, tuple):
            teks, model_dipakai = hasil
        else:
            teks, model_dipakai = hasil, model
        return {"ok": True, "limit": False, "teks": teks, "model": model_dipakai}
    except RuntimeError as e:
        st = status_groq()
        return {"ok": False, "limit": st["limit"], "teks": str(e), "model": model}
    except requests.Timeout:
        return {
            "ok": False,
            "limit": False,
            "teks": "Server Groq terlalu lama merespons. Hasil prediksi tetap valid; coba muat ulang penjelasan.",
            "model": model,
        }
    except requests.HTTPError as e:
        resp = e.response
        kode = resp.status_code if resp is not None else "?"
        if kode == 429:
            st = status_groq()
            sisa = st.get("sisa_detik", 60)
            return {
                "ok": False,
                "limit": True,
                "model": model,
                "teks": (
                    f"Semua model Groq sedang mencapai batas limit (HTTP 429, coba lagi dalam ~{sisa} detik). "
                    "Penjelasan sementara tidak tersedia, namun hasil prediksi tidak terpengaruh."
                ),
            }
        detail = _detail_http_error(resp)
        pesan = f"Permintaan ditolak (HTTP {kode}). Periksa API key, nama model, atau batas kuota."
        if detail:
            pesan += f" Detail: {detail}"
        return {"ok": False, "limit": False, "model": model, "teks": pesan}
    except Exception:
        return {"ok": False, "limit": False, "teks": "Penjelasan belum tersedia saat ini.", "model": model}


def penjelasan_grafik(paksa=False):
    """Kembalikan {'groq': {...}, 'status': {...}}."""
    if not paksa and "groq" in _cache:
        return {"groq": _cache["groq"], "status": status_groq()}

    st = status_groq()
    if st["limit"]:
        return {
            "groq": {"ok": False, "limit": True, "teks": st["pesan"], "model": GROQ_MODEL},
            "status": st,
        }

    prompt = buat_prompt(ringkasan_data())
    res = _jalankan(tanya_groq, GROQ_MODEL, prompt)
    if res["ok"]:
        _cache["groq"] = res
    return {"groq": res, "status": status_groq()}


def penjelasan_prediksi(data_buku, hasil):
    """Jelaskan satu hasil prediksi memakai Groq API (terpisah dari hasil prediksi ML)."""
    kunci_cache = (
        data_buku["user_rating"],
        data_buku["reviews"],
        data_buku["price"],
        data_buku["year"],
        hasil["label"],
        hasil["keyakinan"],
    )
    if kunci_cache in _cache_prediksi:
        res = dict(_cache_prediksi[kunci_cache])
        res["status"] = status_groq()
        return res

    st = status_groq()
    if st["limit"]:
        return {
            "ok": False,
            "limit": True,
            "teks": st["pesan"],
            "model": GROQ_MODEL,
            "status": st,
        }

    prompt = buat_prompt_prediksi(data_buku, hasil, ringkasan_fitur_genre())
    res = _jalankan(tanya_groq, GROQ_MODEL, prompt)
    if res["ok"]:
        _cache_prediksi[kunci_cache] = res
    out = dict(res)
    out["status"] = status_groq()
    return out
