# BookClass — Aplikasi Prediksi Genre Buku Bestseller (Django + Machine Learning)

BookClass adalah aplikasi web berbasis Django yang memprediksi kategori buku bestseller Amazon (**Fiction** atau **Non Fiction**) berdasarkan rating pembaca, jumlah ulasan, harga, dan tahun masuk daftar bestseller. Selain menampilkan hasil prediksi dan grafik perbandingan harga, aplikasi ini juga menyediakan penjelasan otomatis berbahasa awam menggunakan Groq API agar pengguna yang tidak memahami istilah machine learning tetap bisa mengerti arti output yang ditampilkan.

Proyek ini disusun untuk memenuhi tugas **Mini Proyek A (Aplikasi Prediksi ML)** pada mata kuliah Sistem Cerdas.

---

## Fitur Utama

1. **Form Input dengan Validasi Batas**
   Pengguna memasukkan 4 karakteristik buku melalui form:
   - **User Rating**: nilai `1.0` sampai `5.0`
   - **Jumlah Reviews**: nilai `0` sampai `100.000`
   - **Harga (USD)**: nilai `0` sampai `150`
   - **Tahun masuk bestseller**: nilai `2009` sampai `2019`
   Input di luar rentang akan ditolak otomatis oleh validasi form Django sebelum diproses model.

2. **Prediksi Kategori ML (`RandomForestClassifier`)**
   Model membaca input pengguna, menyusun urutan fitur sesuai saat pelatihan (`User Rating`, `Reviews`, `Price`, `Year`), lalu menampilkan kategori prediksi (**Fiction** atau **Non Fiction**) beserta persentase tingkat keyakinan model.

3. **Grafik Data Perbandingan Harga**
   Aplikasi menampilkan grafik batang (*bar chart*) rata-rata harga buku **Fiction** dibandingkan **Non Fiction** yang dihitung dari 351 judul buku unik pada dataset.

4. **Penjelasan Output Menggunakan Groq API (Opsional / Terpisah dari Prediksi)**
   - **Penjelasan Prediksi**: Menerjemahkan hasil prediksi dan persentase keyakinan ke bahasa Indonesia yang sederhana dengan membandingkan input pengguna terhadap rata-rata dataset.
   - **Penjelasan Grafik**: Menjelaskan makna grafik rata-rata harga per genre beserta catatan sebaran data.
   - **Fallback Model & Peringatan Limit**: Jika model utama Groq mencapai batas kuota (*rate limit* / HTTP 429) atau tidak tersedia (HTTP 404), aplikasi otomatis mencoba model cadangan. Apabila seluruh model sedang limit, banner pemberitahuan muncul di atas tombol **Prediksi Kategori**.
   - **Independensi Prediksi**: Proses prediksi ML berjalan sepenuhnya secara lokal menggunakan `model.pkl`. Gangguan atau limit pada Groq API tidak akan mengubah ataupun menghambat hasil prediksi.

---

## Sumber & Lisensi Dataset

- **File Dataset**: `klasifikasi/ml/bestsellers.csv`
- **Nama Dataset**: Amazon Top 50 Bestselling Books 2009–2019
- **Sumber**: Dataset terbuka dari Kaggle (*Amazon Top 50 Bestselling Books 2009 - 2019*)
- **Lisensi**: CC0: Public Domain (bebas digunakan untuk keperluan edukasi dan riset)
- **Catatan Preprocessing**: Dataset awal berisi 550 baris karena satu judul buku dapat muncul di beberapa tahun. Pada tahap pelatihan (`klasifikasi/ml/train_model.py`), duplikat judul dihapus menggunakan `drop_duplicates(subset="Name")` sehingga tersisa 351 buku unik guna mencegah kebocoran data (*data leakage*).

---

## Arsitektur Singkat & Alur Kerja

1. **Pelatihan Model (`klasifikasi/ml/train_model.py`)**:
   - Membandingkan tiga algoritma: Decision Tree, KNN (dengan `StandardScaler`), dan Random Forest, masing-masing dengan dan tanpa fitur `Year`.
   - Model dengan akurasi terbaik dipilih dan disimpan ke `klasifikasi/ml/model.pkl` menggunakan `joblib.dump`.
2. **Aplikasi Web (`klasifikasi/views.py`)**:
   - Model dimuat sekali saat server Django menyala menggunakan `joblib.load`.
   - Halaman utama (`/`) memproses form POST, menjalankan `model.predict()` dan `model.predict_proba()`, serta membuat grafik menggunakan `matplotlib`.
   - JavaScript di `klasifikasi/templates/klasifikasi/index.html` memanggil endpoint JSON `/penjelasan/` dan `/penjelasan-prediksi/` secara asinkron (*fetch*) agar halaman utama tidak lambat menunggu respons AI.

---

## Struktur Direktori Proyek

```text
bookclass-project/
├── bookclass/                  # Konfigurasi utama proyek Django
│   ├── settings.py             # Pengaturan aplikasi, database, dan bahasa
│   └── urls.py                 # Routing utama yang mengarah ke app klasifikasi
├── klasifikasi/                # Aplikasi utama prediksi & penjelasan AI
│   ├── ml/
│   │   ├── bestsellers.csv     # Dataset buku bestseller Amazon (2009-2019)
│   │   ├── model.pkl           # Model ML terlatih yang disimpan dengan joblib
│   │   └── train_model.py      # Skrip pelatihan dan pemilihan model terbaik
│   ├── templates/klasifikasi/
│   │   └── index.html          # Antarmuka web (form, hasil, grafik, dan AI)
│   ├── ai_explain.py           # Integrasi Groq API, prompt, cache, dan fallback limit
│   ├── forms.py                # Definisi dan validasi form input buku
│   ├── tests.py                # Unit test untuk view, endpoint AI, dan fallback limit
│   ├── urls.py                 # Routing URL aplikasi klasifikasi
│   └── views.py                # Logika prediksi ML, pembuatan grafik, dan endpoint JSON
├── .env                        # Penyimpanan API key (tidak di-commit ke Git)
├── manage.py                   # Utilitas CLI Django
├── requirements.txt            # Daftar library Python yang dibutuhkan
└── README.md                   # Dokumentasi proyek
```

---

## Cara Menjalankan Proyek

### 1. Prasyarat
- Python 3.10 atau lebih baru
- Koneksi internet (hanya dibutuhkan untuk fitur penjelasan AI Groq; fitur prediksi ML dan grafik tetap berjalan offline)

### 2. Aktifkan Virtual Environment & Install Dependensi
Buka terminal di folder utama proyek (folder yang berisi `manage.py`), lalu jalankan perintah berikut.

Pada Windows (PowerShell / Command Prompt):
```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```
*(Jika folder `venv` sudah tersedia, cukup jalankan `.\venv\Scripts\activate`.)*

### 3. Konfigurasi File `.env`
Buat file bernama `.env` di folder utama proyek (sejajar dengan `manage.py`), lalu isi dengan API key Groq Anda:

```env
GROQ_API_KEY=isi_api_key_groq_anda_di_sini
GROQ_MODEL=openai/gpt-oss-120b
GROQ_FALLBACK_MODELS=openai/gpt-oss-20b,qwen/qwen3-32b,llama-3.1-8b-instant
```

Catatan:
- `GROQ_MODEL` dan `GROQ_FALLBACK_MODELS` bersifat opsional; jika tidak ditulis, aplikasi otomatis memakai nilai default di atas.
- Jangan membagikan atau mengunggah file `.env` ke repositori publik.

### 4. (Opsional) Melatih Ulang Model ML
File `klasifikasi/ml/model.pkl` sudah tersedia dan siap pakai. Namun, jika Anda ingin melatih ulang model dari `bestsellers.csv`, jalankan:

```powershell
python klasifikasi/ml/train_model.py
```

### 5. Menjalankan Server Django
Jalankan perintah berikut dari folder utama proyek:

```powershell
python manage.py runserver
```

Buka browser dan akses alamat:
`http://127.0.0.1:8000/`

---

## Pengujian (Unit Test)

Proyek ini dilengkapi unit test untuk memastikan halaman utama, prediksi ML, validasi input, endpoint penjelasan AI, dan mekanisme fallback saat limit berjalan dengan benar:

```powershell
python manage.py check
python manage.py test klasifikasi -v 2
```
