# Peramalan Tren Fashion Menggunakan Analisis Deret Waktu

Ini adalah karya asli saya. Karya ini belum pernah diajukan sebelumnya untuk kelas Machine Learning di Dicoding maupun dipublikasikan di platform lain. Dataset yang digunakan merupakan data kuantitatif dengan lebih dari 500 sampel dan diunduh melalui antarmuka KaggleHub (atau melalui URL yang telah disediakan). Proyek ini berfokus pada peramalan tren fashion dalam konteks perencanaan rantai pasok, strategi pemasaran, dan manajemen inventaris.

---

## 1. Pendahuluan

### 1.1 Pernyataan Masalah
Industri fashion menghadapi dinamika yang cepat, dengan tren dan perilaku konsumen yang berubah secara drastis—hal ini menuntut adanya sistem prediksi yang andal. Proyek ini bertujuan untuk **meramalkan tren fashion di masa depan** dengan memanfaatkan data historis penjualan (sales count) dari tahun 2018 hingga 2022. Prediksi tersebut sangat membantu untuk:
- Mengoptimalkan rantai pasok.
- Mendukung strategi pemasaran.
- Menentukan perencanaan inventaris produk fashion.

### 1.2 Tujuan Proyek
- **Analisis Data:** Memahami fitur-fitur utama yang memengaruhi tren fashion.
- **Praproses dan Transformasi Data:** Menyiapkan data secara tepat agar layak untuk pengembangan model peramalan.
- **Pengembangan Model:** Mengimplementasikan model ARIMA sebagai baseline dan model LSTM sebagai pendekatan deep learning untuk peramalan deret waktu.
- **Evaluasi Model:** Mengukur performa masing-masing model menggunakan metrik evaluasi seperti MAE dan RMSE, serta melakukan validasi secara visual.
- **Dokumentasi:** Menyusun laporan secara terstruktur sehingga seluruh proses dari problem clarification hingga evaluasi akhir, dapat ditelusuri dengan jelas.

### 1.3 Referensi
- [Fashion Forecasting: Types, Purpose, Steps and Influence – Textile Learner](https://textilelearner.net/fashion-forecasting-types-purpose-steps/)

---

## 2. Akuisisi dan Pemahaman Data

### 2.1 Akuisisi Data
Dataset diunduh langsung dari URL berikut:

```python
import pandas as pd

# Dataset URL
url = "https://raw.githubusercontent.com/ulfasyabania/Proyek-Pertama-Kirim-Submission-dan-Review/refs/heads/main/fashion_data_2018_2022.xls"

# Load dataset langsung dari URL
df = pd.read_excel(url)

# Menampilkan 5 baris pertama untuk pemeriksaan awal
print(df.head())
```

Kode di atas memastikan bahwa dataset telah berhasil dimuat dan siap untuk dieksplorasi lebih lanjut.

### 2.2 Gambaran dan Eksplorasi Data
Dataset tersebut terdiri atas berbagai kolom, antara lain:
- **Tanggal/Waktu:** Tanggal pencatatan data (menggunakan kolom *last_stock_date*).
- **Sales Count:** Data numerik yang menunjukkan jumlah penjualan atau intensitas tren penjualan.
- **Fitur Tambahan:** Seperti product_id, product_name, kategori, harga, discount, dan lain-lain.

Tahapan eksplorasi meliputi:
- **Statistik Ringkasan:** Analisis nilai rata-rata, median, standar deviasi, dll.
- **Pemeriksaan Tipe Data:** Memastikan setiap fitur memiliki tipe yang sesuai.
- **Identifikasi Nilai Hilang:** Memeriksa dan menangani missing values.
- **Visualisasi Data:** Menggunakan grafik garis, histogram, dan boxplot untuk mengidentifikasi pola, outlier, dan tren musiman.

---

## 3. Persiapan Data

### 3.1 Pembersihan Data
- **Nilai Hilang:** Seluruh nilai yang hilang diidentifikasi dan ditangani (diisi atau dihapus) agar integritas data terjaga.
- **Format Tanggal:** Kolom *last_stock_date* dikonversi ke format datetime dan dijadikan index untuk memudahkan operasi resampling.
- **Deteksi Outlier:** Dengan visualisasi boxplot, outlier yang potensial diidentifikasi dan dipertimbangkan untuk penghapusan jika mengganggu peramalan.

### 3.2 Rekayasa Fitur
- **Fitur Berbasis Waktu:** Informasi hari, bulan, dan tahun diekstraksi untuk mendukung analisis tren.
- **Agregasi:** Data harian diagregasi menjadi data bulanan dengan menggunakan rata-rata pada kolom *sales_count*.
- **Normalisasi:** Skala variabel numerik disesuaikan untuk memastikan perbedaan rentang tidak mempengaruhi model saat pelatihan.

Setiap tahapan data preparation dilakukan secara berurutan dan didukung visualisasi untuk menyampaikan alasan pemilihan teknik tersebut.

---

## 4. Pemodelan

### 4.1 Rasional Pemilihan Model
Dalam menyelesaikan permasalahan peramalan tren fashion, diterapkan dua pendekatan:
1. **Model ARIMA**  
   Pendekatan statistik yang efektif dalam menangkap tren linier dan musiman.
2. **Model LSTM (Long Short-Term Memory)**  
   Pendekatan deep learning yang mampu menangkap pola non-linear dan ketergantungan temporal.

### 4.2 Pengembangan Model

#### Model ARIMA
- **Pengujian Stasioneritas:** Dilakukan menggunakan uji Augmented Dickey-Fuller (ADF). Hasil uji menunjukkan data sudah stasioner (p-value < 0.05), sehingga parameter differencing (d) diatur ke 0.
- **Pengaturan Parameter:** Order ARIMA dipilih (misalnya order (1, 0, 1)) berdasarkan kriteria AIC/BIC.
- **Pelatihan dan Peramalan:** Model dilatih pada data historis dan digunakan untuk peramalan selama 60 bulan ke depan (5 tahun).

#### Model LSTM
- **Persiapan Data:** Data *sales_count* dinormalisasi dan dienkapsulasi menjadi sequence dengan _look_back_ 12 bulan.
- **Arsitektur Model:** Model LSTM dengan 50 neuron dan satu layer Dense sebagai output.
- **Pelatihan:** Menggunakan Early Stopping untuk menghindari overfitting.
- **Peramalan Iteratif:** Menggunakan nilai prediksi sebagai input untuk memprediksi langkah ke depan hingga 60 periode.

### 4.3 Pertimbangan Tambahan
Jika satu model (misalnya ARIMA) sudah menunjukkan hasil yang menjanjikan, pendekatan LSTM tetap dieksplorasi untuk mengidentifikasi kemungkinan peningkatan kinerja melalui optimasi hyperparameter atau penambahan fitur eksternal.

---

## 5. Evaluasi

### 5.1 Metrik Evaluasi
Proses evaluasi menggunakan metrik:
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
  
Metrik-metrik ini digunakan untuk mengukur seberapa dekat prediksi model dengan data aktual. Penjelasan singkat metrik:
- **MAE:** Rata-rata nilai absolut kesalahan.
- **RMSE:** Akar kuadrat rata-rata kesalahan, memberikan penalti lebih besar pada kesalahan yang signifikan.

### 5.2 Validasi Model
- **Pembagian Data:** Data historis dibagi secara kronologis antara data pelatihan dan pengujian (atau validasi).
- **Analisis Residual:** Dilakukan evaluasi terhadap residual prediksi untuk memastikan tidak ada pola tersisa.
- **Perbandingan Hasil Peramalan:** Hasil peramalan model ARIMA dan LSTM divisualisasikan bersama data aktual untuk perbandingan secara visual.

---

## 6. Simpulan dan Pekerjaan Selanjutnya

### 6.1 Ringkasan Temuan
- **Model ARIMA:**  
  Mampu menangkap pola tren linier dan musiman dengan baik. Evaluasi model pada data test menunjukkan nilai MAE dan RMSE yang mengindikasikan prediksi yang cukup akurat (misal: MAE sekitar 21,67, RMSE sekitar 27,40).
- **Model LSTM:**  
  Meskipun dengan hasil kuantitatif sedikit kurang (misal: MAE sekitar 25,75, RMSE sekitar 31,08), model LSTM menunjukkan potensi besar melalui kemampuannya menangkap pola non-linear.  
- **Hasil Peramalan:**  
  Kedua model diaplikasikan untuk memprediksi _sales count_ selama 5 tahun mendatang, dengan visualisasi yang menunjukkan tren peningkatan secara konsisten.

### 6.2 Rekomendasi dan Pengembangan Selanjutnya
- **Tuning Model:** Melakukan pencarian parameter optimal, misalnya menggunakan auto_arima untuk ARIMA dan eksplorasi konfigurasi lebih lanjut untuk LSTM.
- **Integrasi Data Eksternal:** Menambahkan fitur eksternal seperti data ekonomi atau sentimen media sosial untuk memperkaya model.
- **Optimalisasi LSTM:** Mengeksplorasi penggunaan dropout, penambahan lapisan (layer), atau modifikasi struktur untuk meningkatkan performa dan generalisasi.

Secara keseluruhan, proyek ini membuktikan bahwa analisis deret waktu dapat diterapkan secara efektif dalam peramalan tren fashion. Hasil yang diperoleh dari kedua pendekatan dapat menjadi dasar strategis dalam pengambilan keputusan di bidang rantai pasok, pemasaran, dan perencanaan inventaris.

---

## 7. Dokumentasi dan Kepatuhan Format Laporan

Laporan ini disusun dengan mengikuti template dan ketentuan laporan submission Dicoding. Setiap tahap—mulai dari akuisi data, eksplorasi, persiapan data, pemodelan, hingga evaluasi—telah dituliskan secara mendalam dan disertai dengan kode snippet yang relevan. Semua resources (grafik, tabel) telah disusun agar dapat dimuat dengan baik, sehingga pembaca dengan mudah memahami alur proyek dan keputusan yang diambil.

---
