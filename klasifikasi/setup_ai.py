"""Pemasang otomatis fitur "Penjelasan Grafik AI" (Groq + Gemini) untuk BookClass.

Cara pakai (dari folder yang berisi manage.py, dengan (venv) aktif):
    python setup_ai.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()

FILES = {}

FILES['klasifikasi/views.py'] = r'''import base64
import io
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # wajib di server, tanpa jendela GUI
import matplotlib.pyplot as plt
import pandas as pd
from django.http import JsonResponse
from django.shortcuts import render

from .ai_explain import penjelasan_grafik
from .forms import BukuForm

ML_DIR = Path(__file__).resolve().parent / "ml"

# Model dimuat sekali saat server menyala
bundle = joblib.load(ML_DIR / "model.pkl")
model = bundle["model"]
FEATURES = bundle["features"]   # urutan fitur sama dengan saat pelatihan
LABELS = bundle["labels"]

# nama kolom dataset -> nama field di form
FIELD_MAP = {
    "User Rating": "user_rating",
    "Reviews": "reviews",
    "Price": "price",
    "Year": "year",
}


def buat_grafik():
    """Bar chart rata-rata harga Fiction vs Non Fiction (base64)."""
    df = pd.read_csv(ML_DIR / "bestsellers.csv").drop_duplicates(subset="Name")
    rata = df.groupby("Genre")["Price"].mean().reindex(["Fiction", "Non Fiction"])

    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#fffdf7")
    ax.set_facecolor("#fffdf7")
    batang = ax.bar(rata.index, rata.values, color=["#7a9a5a", "#c9a66b"], width=0.5)
    ax.set_ylabel("Rata-rata harga (USD)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for b in batang:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                f"${b.get_height():.2f}", ha="center", va="bottom", fontweight="bold")

    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def index(request):
    hasil = None

    if request.method == "POST":
        form = BukuForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            # Susun input sesuai fitur & urutan model
            X = pd.DataFrame([{f: data[FIELD_MAP[f]] for f in FEATURES}])
            pred = int(model.predict(X)[0])
            proba = model.predict_proba(X)[0]
            kelas = list(model.classes_)
            hasil = {
                "label": LABELS[pred],
                "keyakinan": round(float(proba[kelas.index(pred)]) * 100, 1),
            }
    else:
        form = BukuForm()

    return render(request, "klasifikasi/index.html", {
        "form": form,
        "hasil": hasil,
        "grafik": buat_grafik(),
    })


def penjelasan(request):
    """Endpoint JSON: penjelasan grafik dari Groq dan Gemini."""
    paksa = request.GET.get("refresh") == "1"
    return JsonResponse(penjelasan_grafik(paksa))
'''

FILES['klasifikasi/urls.py'] = r'''from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("penjelasan/", views.penjelasan, name="penjelasan"),
]
'''

FILES['klasifikasi/ai_explain.py'] = r'''"""Penjelasan grafik memakai Groq dan Gemini (dipanggil dari views.penjelasan)."""
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# .env berada di folder utama proyek (sejajar dengan manage.py)
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

ML_DIR = Path(__file__).resolve().parent / "ml"
TIMEOUT = 25  # detik

# Nama model bisa diganti lewat .env tanpa mengubah kode
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Data grafik tidak berubah, jadi jawaban yang sukses disimpan agar kuota hemat
_cache = {}


def ringkasan_data():
    """Statistik harga per genre, dihitung dari 351 buku unik (sama dengan data latih)."""
    df = pd.read_csv(ML_DIR / "bestsellers.csv").drop_duplicates(subset="Name")
    stat = df.groupby("Genre")["Price"].agg(["count", "mean", "median", "std", "min", "max"])
    return stat.round(2).to_dict("index")


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


def tanya_groq(prompt):
    kunci = os.getenv("GROQ_API_KEY")
    if not kunci:
        raise RuntimeError("GROQ_API_KEY belum diisi di file .env")
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {kunci}"},
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 500,
        },
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def tanya_gemini(prompt):
    kunci = os.getenv("GEMINI_API_KEY")
    if not kunci:
        raise RuntimeError("GEMINI_API_KEY belum diisi di file .env")
    r = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
        headers={"x-goog-api-key": kunci, "Content-Type": "application/json"},
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1500},
        },
        timeout=TIMEOUT,
    )
    r.raise_for_status()
    bagian = r.json()["candidates"][0]["content"]["parts"]
    return "".join(b.get("text", "") for b in bagian).strip()


def _jalankan(fungsi, model, prompt):
    """Jalankan satu pemanggilan API; kegagalan dikembalikan sebagai pesan, bukan error."""
    try:
        return {"ok": True, "teks": fungsi(prompt), "model": model}
    except RuntimeError as e:                      # kunci belum diisi
        return {"ok": False, "teks": str(e), "model": model}
    except requests.Timeout:
        return {"ok": False, "teks": "Server AI terlalu lama merespons. Coba lagi.", "model": model}
    except requests.HTTPError as e:
        kode = e.response.status_code if e.response is not None else "?"
        return {"ok": False, "model": model,
                "teks": f"Permintaan ditolak (HTTP {kode}). Periksa API key, nama model, atau batas kuota."}
    except Exception:
        return {"ok": False, "teks": "Penjelasan belum tersedia saat ini.", "model": model}


def penjelasan_grafik(paksa=False):
    """Kembalikan {'groq': {...}, 'gemini': {...}}; keduanya dipanggil bersamaan."""
    prompt = buat_prompt(ringkasan_data())
    penyedia = {
        "groq": (tanya_groq, GROQ_MODEL),
        "gemini": (tanya_gemini, GEMINI_MODEL),
    }

    hasil = {}
    antre = {}
    for nama, (fungsi, model) in penyedia.items():
        if not paksa and nama in _cache:
            hasil[nama] = _cache[nama]
        else:
            antre[nama] = (fungsi, model)

    with ThreadPoolExecutor(max_workers=2) as pool:
        tugas = {n: pool.submit(_jalankan, f, m, prompt) for n, (f, m) in antre.items()}
        for nama, t in tugas.items():
            res = t.result()
            hasil[nama] = res
            if res["ok"]:
                _cache[nama] = res
    return hasil
'''

FILES['klasifikasi/templates/klasifikasi/index.html'] = r'''<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BookClass | Toko Buku Prediksi</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Lora:wght@500;600;700&display=swap" rel="stylesheet">
  <style>
    /* ===== PALET MATCHA (ubah di sini saja) ===== */
    :root {
      --matcha: #8fae6b;
      --matcha-deep: #3f5a2a;
      --matcha-soft: #e4ecd3;
      --paper: #f5f0e1;
      --card: #fffdf7;
      --kraft: #c9a66b;
      --wood: #8a6a49;
      --border: #ddd5bd;
      --text: #2f3a26;
      --muted: #75795f;
      --error: #b4443a;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: 'Inter', 'Segoe UI', Arial, sans-serif;
      font-size: 15px;
      line-height: 1.55;
      color: var(--text);
      background: var(--paper);
    }

    /* ===== Kanopi toko ===== */
    .awning {
      height: 54px;
      background: repeating-linear-gradient(90deg, var(--matcha) 0 40px, #fffaf0 40px 80px);
      -webkit-mask: conic-gradient(from -45deg at bottom, #0000, #000 1deg 89deg, #0000 90deg) 50% / 40px 100%;
              mask: conic-gradient(from -45deg at bottom, #0000, #000 1deg 89deg, #0000 90deg) 50% / 40px 100%;
    }

    /* ===== Papan nama ===== */
    .sign { text-align: center; padding: 22px 20px 6px; }
    .sign .plate {
      display: inline-block;
      padding: 10px 34px 12px;
      background: var(--matcha-deep);
      border: 3px double var(--kraft);
      border-radius: 6px;
      color: #fffaf0;
      box-shadow: 0 4px 0 rgba(63, 90, 42, 0.25);
    }
    .sign h1 { margin: 0; font-family: 'Lora', Georgia, serif; font-size: 1.9rem; letter-spacing: 0.02em; }
    .sign .est { font-size: 0.72rem; letter-spacing: 0.25em; text-transform: uppercase; color: var(--matcha-soft); }
    .sign p { max-width: 560px; margin: 14px auto 0; color: var(--muted); }

    /* ===== Rak buku ===== */
    .shelf { max-width: 760px; margin: 18px auto 0; padding: 0 20px; }
    .books { display: flex; align-items: flex-end; justify-content: center; gap: 3px; height: 92px; }
    .books i {
      display: block; width: 22px; border-radius: 3px 3px 0 0;
      border-top: 4px solid rgba(255, 255, 255, 0.35);
    }
    .books i.lean { transform: rotate(8deg); transform-origin: bottom left; margin-left: 6px; margin-right: 10px; }
    .plank { height: 14px; background: var(--wood); border-radius: 3px; box-shadow: 0 5px 0 rgba(138, 106, 73, 0.3); }

    /* ===== Konten ===== */
    .container { max-width: 960px; margin: 0 auto; padding: 34px 20px 40px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; align-items: start; }
    @media (max-width: 760px) { .grid { grid-template-columns: 1fr; } }

    .card {
      position: relative;
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 26px 24px 24px;
      box-shadow: 0 2px 0 var(--border), 0 10px 22px rgba(63, 90, 42, 0.08);
    }
    .tab {
      position: absolute; top: -13px; left: 20px;
      padding: 3px 14px;
      background: var(--kraft); color: #fffaf0;
      font-size: 0.68rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase;
      border-radius: 4px 4px 0 0;
    }
    .card h2 { margin: 0 0 4px; font-family: 'Lora', Georgia, serif; font-size: 1.2rem; color: var(--matcha-deep); }
    .card .desc { margin: 0 0 18px; color: var(--muted); font-size: 0.86rem; }

    /* ===== Form ===== */
    .field { margin: 0 0 14px; }
    label { display: block; margin-bottom: 5px; font-size: 0.84rem; font-weight: 500; }
    input {
      width: 100%; padding: 10px 12px;
      font-family: inherit; font-size: 0.95rem; color: var(--text);
      background: #fff; border: 1px solid #cfc7ac; border-radius: 6px; outline: none;
      transition: border-color .15s, box-shadow .15s;
    }
    input:focus { border-color: var(--matcha); box-shadow: 0 0 0 3px rgba(143, 174, 107, 0.3); }
    input[aria-invalid="true"] { border-color: var(--error); }
    input[aria-invalid="true"]:focus { box-shadow: 0 0 0 3px rgba(180, 68, 58, 0.2); }
    .helptext { display: none; }

    /* Pesan error: tampil tepat di bawah kolom yang bermasalah */
    .err { display: block; margin-top: 4px; color: var(--error); font-size: 0.8rem; }
    .form-error {
      margin: 0 0 14px; padding: 10px 12px;
      color: var(--error); font-size: 0.85rem;
      background: #fbeeec; border-left: 3px solid var(--error); border-radius: 4px;
    }

    button {
      width: 100%; padding: 12px 16px; margin-top: 4px;
      font-family: inherit; font-size: 0.95rem; font-weight: 600; color: #fff;
      background: var(--matcha-deep); border: 0; border-radius: 6px; cursor: pointer;
      box-shadow: 0 3px 0 #2c4020;
      transition: transform .1s, box-shadow .1s, background .15s;
    }
    button:hover { background: #4b6a33; }
    button:active { transform: translateY(2px); box-shadow: 0 1px 0 #2c4020; }
    button:focus-visible { outline: 3px solid var(--kraft); outline-offset: 2px; }

    /* ===== Hasil + pita pembatas buku ===== */
    .ribbon {
      position: absolute; top: -2px; right: 22px; width: 30px; height: 46px;
      background: var(--matcha);
      clip-path: polygon(0 0, 100% 0, 100% 100%, 50% 78%, 0 100%);
    }
    .result-label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); }
    .result-value { margin: 2px 0 18px; font-family: 'Lora', Georgia, serif; font-size: 2.1rem; font-weight: 700; color: var(--matcha-deep); }
    .meter-head { display: flex; justify-content: space-between; margin-bottom: 6px; font-size: 0.85rem; }
    .meter-head strong { color: var(--matcha-deep); }
    .meter { height: 10px; background: var(--matcha-soft); border: 1px solid #cdd9b5; border-radius: 999px; overflow: hidden; }
    .meter span { display: block; height: 100%; background: var(--matcha); }
    .note { margin: 16px 0 0; padding: 10px 12px; font-size: 0.8rem; color: var(--muted);
            background: var(--matcha-soft); border-left: 3px solid var(--matcha); border-radius: 4px; }

    .placeholder { text-align: center; padding: 30px 10px; color: var(--muted); }
    .placeholder .icon { font-size: 1.8rem; margin-bottom: 6px; }

    .chart-card { margin-top: 30px; }
    .chart-card img { display: block; max-width: 100%; height: auto; margin: 0 auto; }

    footer { text-align: center; padding: 0 20px 28px; color: var(--muted); font-size: 0.8rem; }
    footer .rule { width: 60px; height: 3px; margin: 0 auto 12px; background: var(--kraft); border-radius: 2px; }

    /* ===== Penjelasan AI ===== */
    .ai-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    @media (max-width: 760px) { .ai-grid { grid-template-columns: 1fr; } }
    .ai-box { padding: 16px 18px; background: var(--matcha-soft); border: 1px solid #cdd9b5; border-radius: 6px; }
    .ai-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 8px; }
    .ai-name { font-family: 'Lora', Georgia, serif; font-weight: 600; color: var(--matcha-deep); }
    .ai-model { font-size: 0.7rem; color: var(--muted); background: #fff; border: 1px solid #cdd9b5;
                padding: 2px 8px; border-radius: 999px; }
    .ai-body { margin: 0; font-size: 0.9rem; white-space: pre-line; }
    .ai-box.is-loading .ai-body { color: var(--muted); font-style: italic; }
    .ai-box.is-error { background: #fbeeec; border-color: #e8c4bf; }
    .ai-box.is-error .ai-body { color: var(--error); }
    .ai-foot { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-top: 14px; }
    .ai-foot small { color: var(--muted); font-size: 0.78rem; }
    .btn-small { width: auto; margin: 0; padding: 7px 14px; font-size: 0.8rem; box-shadow: 0 2px 0 #2c4020; }
    .btn-small:disabled { opacity: .6; cursor: wait; }
  </style>
</head>
<body>

  <div class="awning"></div>

  <header class="sign">
    <div class="plate">
      <h1>BookClass</h1>
      <div class="est">Toko Buku Prediksi &middot; Est. 2026</div>
    </div>
    <p>Tebak apakah sebuah buku bestseller Amazon termasuk Fiction atau Non Fiction
       dari rating, jumlah ulasan, harga, dan tahun terbitnya.</p>
  </header>

  <div class="shelf">
    <div class="books" aria-hidden="true">
      <i style="height:70px;background:#7a9a5a"></i>
      <i style="height:86px;background:#3f5a2a"></i>
      <i style="height:62px;background:#c9a66b"></i>
      <i style="height:78px;background:#a9c08a"></i>
      <i style="height:90px;background:#8a6a49"></i>
      <i style="height:66px;background:#e4d6a8"></i>
      <i style="height:82px;background:#5f7f3f"></i>
      <i class="lean" style="height:74px;background:#c9a66b"></i>
      <i style="height:88px;background:#7a9a5a"></i>
      <i style="height:64px;background:#3f5a2a"></i>
      <i style="height:80px;background:#d8c48c"></i>
      <i style="height:72px;background:#a9c08a"></i>
      <i style="height:90px;background:#6b8a4b"></i>
      <i style="height:68px;background:#8a6a49"></i>
    </div>
    <div class="plank"></div>
  </div>

  <main class="container">
    <div class="grid">

      <section class="card" aria-labelledby="catalog-heading">
        <span class="tab">Kartu Katalog</span>
        <h2 id="catalog-heading">Data Buku</h2>
        <p class="desc">Isi karakteristik buku yang ingin diprediksi.</p>

        <form method="post" action="{% url 'index' %}" novalidate>
          {% csrf_token %}

          {% if form.non_field_errors %}
            <div class="form-error" role="alert">
              {% for e in form.non_field_errors %}{{ e }}{% endfor %}
            </div>
          {% endif %}

          {% for field in form %}
            <div class="field">
              {{ field.label_tag }}
              {{ field }}
              {% for e in field.errors %}
                <span class="err" role="alert">{{ e }}</span>
              {% endfor %}
            </div>
          {% endfor %}

          <button type="submit">Prediksi Kategori</button>
        </form>
      </section>

      <section class="card" aria-labelledby="result-heading">
        <span class="tab">Rekomendasi Rak</span>
        {% if hasil %}<span class="ribbon"></span>{% endif %}
        <h2 id="result-heading">Hasil Prediksi</h2>
        <p class="desc">Kategori dan tingkat keyakinan model.</p>

        {% if hasil %}
          <div role="status" aria-live="polite">
            <div class="result-label">Letakkan di rak</div>
            <div class="result-value">{{ hasil.label }}</div>
          </div>

          <div class="meter-head">
            <span>Tingkat keyakinan</span>
            <strong>{{ hasil.keyakinan }}%</strong>
          </div>
          <div class="meter" role="progressbar" aria-label="Tingkat keyakinan" aria-valuemin="0" aria-valuemax="100" aria-valuenow="{{ hasil.keyakinan|floatformat:0 }}">
            <span style="width: {{ hasil.keyakinan|stringformat:'.1f' }}%"></span>
          </div>

          <p class="note">Akurasi model pada data uji sekitar 66%. Hasil ini perkiraan, bukan kepastian.</p>
        {% else %}
          <div class="placeholder">
            <div class="icon">&#128214;</div>
            Isi kartu katalog lalu klik <strong>Prediksi Kategori</strong><br>untuk melihat rak yang cocok.
          </div>
        {% endif %}
      </section>

    </div>

    <section class="card chart-card" aria-labelledby="chart-heading">
      <span class="tab">Papan Harga</span>
      <h2 id="chart-heading">Rata-rata Harga per Genre</h2>
      <p class="desc">Perbandingan rata-rata harga buku Fiction dan Non Fiction pada dataset.</p>
      {% if grafik %}
        <img src="data:image/png;base64,{{ grafik }}" alt="Grafik rata-rata harga buku Fiction dan Non Fiction">
      {% else %}
        <p class="placeholder">Grafik tidak tersedia.</p>
      {% endif %}
    </section>

    <section class="card chart-card" aria-labelledby="ai-heading">
      <span class="tab">Ulasan Pustakawan AI</span>
      <h2 id="ai-heading">Penjelasan Grafik</h2>
      <p class="desc">Dua model AI menjelaskan grafik di atas berdasarkan statistik dataset.</p>

      <div class="ai-grid" aria-live="polite">
        <div class="ai-box is-loading" id="ai-groq">
          <div class="ai-head"><span class="ai-name">Groq</span><span class="ai-model"></span></div>
          <p class="ai-body">Menyusun penjelasan...</p>
        </div>
        <div class="ai-box is-loading" id="ai-gemini">
          <div class="ai-head"><span class="ai-name">Gemini</span><span class="ai-model"></span></div>
          <p class="ai-body">Menyusun penjelasan...</p>
        </div>
      </div>

      <div class="ai-foot">
        <small>Dibuat otomatis oleh AI dari statistik dataset. Periksa kembali sebelum dikutip.</small>
        <button type="button" id="ai-refresh" class="btn-small">Buat ulang</button>
      </div>
    </section>
  </main>

  <footer>
    <div class="rule"></div>
    BookClass &copy; 2026 &middot; Dibuat dengan Django dan scikit-learn
  </footer>


  <script>
  (function () {
    var url = "{% url 'penjelasan' %}";
    var kotak = { groq: document.getElementById("ai-groq"), gemini: document.getElementById("ai-gemini") };
    var tombol = document.getElementById("ai-refresh");

    function tampil(nama, data) {
      var el = kotak[nama];
      el.classList.remove("is-loading");
      el.classList.toggle("is-error", !data.ok);
      el.querySelector(".ai-body").textContent = data.teks;   // textContent: aman dari XSS
      el.querySelector(".ai-model").textContent = data.model || "";
    }

    function mulaiMuat() {
      Object.keys(kotak).forEach(function (n) {
        kotak[n].classList.add("is-loading");
        kotak[n].classList.remove("is-error");
        kotak[n].querySelector(".ai-body").textContent = "Menyusun penjelasan...";
      });
    }

    function muat(refresh) {
      mulaiMuat();
      tombol.disabled = true;
      fetch(url + (refresh ? "?refresh=1" : ""))
        .then(function (res) { return res.json(); })
        .then(function (data) { tampil("groq", data.groq); tampil("gemini", data.gemini); })
        .catch(function () {
          var gagal = { ok: false, teks: "Tidak dapat memuat penjelasan. Coba lagi.", model: "" };
          tampil("groq", gagal); tampil("gemini", gagal);
        })
        .finally(function () { tombol.disabled = false; });
    }

    tombol.addEventListener("click", function () { muat(true); });
    muat(false);
  })();
  </script>

</body>
</html>
'''

FILES['bookclass/urls.py'] = r'''from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("klasifikasi.urls")),
]
'''


def info(teks):
    print(teks)


def baca_env(path):
    data = {}
    if path.exists():
        for baris in path.read_text(encoding="utf-8").splitlines():
            if "=" in baris and not baris.strip().startswith("#"):
                k, v = baris.split("=", 1)
                data[k.strip()] = v.strip()
    return data


def main():
    if not (ROOT / "manage.py").is_file():
        info("GAGAL: file manage.py tidak ditemukan di folder ini.")
        info("Jalankan perintah ini dari folder yang berisi manage.py.")
        info("Folder sekarang: " + str(ROOT))
        return
    if not (ROOT / "klasifikasi").is_dir() or not (ROOT / "bookclass").is_dir():
        info("GAGAL: folder 'klasifikasi' atau 'bookclass' tidak ditemukan.")
        return

    # 1. Tulis semua file (file lama dicadangkan sebagai .bak)
    info("\n[1/4] Menulis file...")
    for rel, isi in FILES.items():
        target = ROOT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_text(encoding="utf-8", errors="ignore") != isi:
            target.with_name(target.name + ".bak").write_text(
                target.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
        target.write_text(isi, encoding="utf-8")
        info("   OK  " + rel)

    # 2. Pasang library
    info("\n[2/4] Memasang library requests dan python-dotenv...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "python-dotenv"])
    except Exception:
        info("   Gagal otomatis. Jalankan manual: python -m pip install requests python-dotenv")

    # 3. API key
    info("\n[3/4] API key (tekan Enter untuk melewati; key tidak dikirim ke mana pun).")
    env_path = ROOT / ".env"
    env = baca_env(env_path)
    for nama, label in (("GROQ_API_KEY", "Groq  "), ("GEMINI_API_KEY", "Gemini")):
        nilai = input("   Tempel key " + label + ": ").strip().strip('"').strip("'")
        if nilai:
            env[nama] = nilai
    if env:
        env_path.write_text("\n".join(k + "=" + v for k, v in env.items()) + "\n", encoding="utf-8")
        info("   .env disimpan.")
    else:
        info("   Tidak ada key diisi. Kotak AI akan menampilkan 'belum diisi' sampai .env dibuat.")

    gi = ROOT / ".gitignore"
    isi_gi = gi.read_text(encoding="utf-8") if gi.exists() else ""
    if ".env" not in isi_gi.splitlines():
        gi.write_text(isi_gi + ("\n" if isi_gi and not isi_gi.endswith("\n") else "") + ".env\n", encoding="utf-8")

    # 4. Pemeriksaan
    info("\n[4/4] Pemeriksaan...")
    settings = (ROOT / "bookclass" / "settings.py").read_text(encoding="utf-8", errors="ignore")
    if "'klasifikasi'" not in settings and '"klasifikasi"' not in settings:
        info("   PERHATIAN: tambahkan 'klasifikasi', ke INSTALLED_APPS di bookclass/settings.py")
    if not (ROOT / "klasifikasi" / "ml" / "model.pkl").is_file():
        info("   PERHATIAN: model.pkl belum ada. Jalankan: python klasifikasi\\ml\\train_model.py")
    if not (ROOT / "klasifikasi" / "ml" / "bestsellers.csv").is_file():
        info("   PERHATIAN: bestsellers.csv belum ada di klasifikasi\\ml")

    info("\nSELESAI. Sekarang jalankan:")
    info("   python manage.py runserver")
    info("lalu buka http://127.0.0.1:8000/ dan tekan Ctrl + F5.")


if __name__ == "__main__":
    main()