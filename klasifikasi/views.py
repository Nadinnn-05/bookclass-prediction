import base64
import io
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # wajib di server, tanpa jendela GUI
import matplotlib.pyplot as plt
import pandas as pd
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST

from .ai_explain import penjelasan_grafik, penjelasan_prediksi, status_groq
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
    payload_prediksi = None

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
            payload_prediksi = {
                "user_rating": data["user_rating"],
                "reviews": data["reviews"],
                "price": data["price"],
                "year": data["year"],
                "label": hasil["label"],
                "keyakinan": hasil["keyakinan"],
            }
    else:
        form = BukuForm()

    return render(request, "klasifikasi/index.html", {
        "form": form,
        "hasil": hasil,
        "payload_prediksi": payload_prediksi,
        "grafik": buat_grafik(),
        "status_ai": status_groq(),
    })


def status_ai(request):
    """Endpoint JSON: status ketersediaan/limit Groq sebelum user memprediksi."""
    return JsonResponse(status_groq())


def penjelasan(request):
    """Endpoint JSON: penjelasan grafik dari Groq."""
    paksa = request.GET.get("refresh") == "1"
    return JsonResponse(penjelasan_grafik(paksa))


@require_POST
def penjelasan_prediksi_view(request):
    """Endpoint JSON: penjelasan hasil prediksi memakai Groq API."""
    try:
        body = json.loads(request.body.decode("utf-8"))
        data_buku = {
            "user_rating": float(body["user_rating"]),
            "reviews": int(body["reviews"]),
            "price": float(body["price"]),
            "year": int(body["year"]),
        }
        hasil = {
            "label": str(body["label"]),
            "keyakinan": float(body["keyakinan"]),
        }
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return JsonResponse(
            {"ok": False, "teks": "Data prediksi tidak valid.", "model": ""},
            status=400,
        )

    if hasil["label"] not in LABELS.values():
        return JsonResponse(
            {"ok": False, "teks": "Label prediksi tidak dikenali.", "model": ""},
            status=400,
        )

    return JsonResponse(penjelasan_prediksi(data_buku, hasil))