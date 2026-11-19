from pathlib import Path
from sklearn.base import clone
import joblib
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "bestsellers.csv"
MODEL_PATH = BASE_DIR / "model.pkl"

# 1. Baca data
df = pd.read_csv(CSV_PATH)
print("Jumlah baris awal:", len(df))
print(df["Genre"].value_counts(), "\n")

# 2. Hapus duplikat judul (satu buku bisa muncul di banyak tahun -> data leakage)
df = df.drop_duplicates(subset="Name").reset_index(drop=True)
print("Jumlah baris setelah hapus duplikat:", len(df), "\n")

# 3. Ubah target jadi angka: 0 = Fiction, 1 = Non Fiction
df["Target"] = df["Genre"].map({"Fiction": 0, "Non Fiction": 1})
LABELS = {0: "Fiction", 1: "Non Fiction"}

# 4. Bandingkan model, dengan dan tanpa fitur Year
fitur_dengan_year = ["User Rating", "Reviews", "Price", "Year"]
fitur_tanpa_year = ["User Rating", "Reviews", "Price"]

model_kandidat = {
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Random Forest": RandomForestClassifier(
    n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42
),
}

hasil = []
for nama_set, fitur in [("dengan Year", fitur_dengan_year), ("tanpa Year", fitur_tanpa_year)]:
    X = df[fitur]
    y = df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    for nama_model, model_asli in model_kandidat.items():
        model = clone(model_asli)   # salinan baru, tidak saling menimpa
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        hasil.append((acc, nama_model, nama_set, fitur, model, X_test, y_test))
        print(f"{nama_model:15s} {nama_set:12s} akurasi: {acc:.3f}")

# 5. Pilih yang akurasinya tertinggi
acc, nama_model, nama_set, fitur, model, X_test, y_test = max(
    hasil, key=lambda h: (h[0], h[1] == "Random Forest")  # kalau seri, pilih Random Forest
)
print(f"\nModel terbaik: {nama_model} ({nama_set}), akurasi {acc:.3f}\n")
print(classification_report(y_test, model.predict(X_test), target_names=["Fiction", "Non Fiction"]))

# 6. Simpan model + info pendukung
joblib.dump(
    {
        "model": model,
        "features": fitur,          # urutan fitur harus sama saat prediksi
        "labels": LABELS,
        "sklearn_version": sklearn.__version__,
    },
    MODEL_PATH,
)
print("Model disimpan ke:", MODEL_PATH)
