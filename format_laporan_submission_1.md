# Peramalan Tren Fashion Menggunakan Analisis Deret Waktu

*Ini adalah karya asli saya. Karya ini belum pernah diajukan sebelumnya untuk kelas Machine Learning di Dicoding maupun dipublikasikan di platform lain. Dataset yang digunakan adalah data kuantitatif dengan lebih dari 500 sampel dan diunduh menggunakan antarmuka KaggleHub.*

---

## 1. Pendahuluan

### 1.1 Pernyataan Masalah
Industri fashion mengalami perubahan tren yang cepat dan perilaku konsumen yang dinamis. Pada proyek ini, tujuan utamanya adalah **meramalkan tren fashion di masa depan** dengan menggunakan data historis (dari tahun 2018 hingga 2022). Kemampuan untuk memprediksi tren di masa depan sangat bermanfaat untuk optimasi rantai pasok, strategi pemasaran, dan manajemen inventaris. Dalam studi ini, akan dikembangkan dan dievaluasi model peramalan berbasis deret waktu.

### 1.2 Tujuan
- **Menganalisis dataset** untuk memahami fitur-fitur utama yang mendorong tren dalam dunia fashion.
- **Melakukan praproses dan transformasi data** agar data tersebut layak digunakan untuk model peramalan deret waktu.
- **Mengembangkan model peramalan** (dengan menggunakan metode seperti ARIMA atau pendekatan deep learning) untuk memprediksi tren yang akan datang.
- **Mengevaluasi kinerja model** menggunakan metrik yang relevan seperti Mean Absolute Error (MAE) dan Root Mean Squared Error (RMSE).
- **Mendokumentasikan seluruh proses** dengan penjelasan mendalam di dalam sel teks pada notebook.

### 1.3 Referensi
https://textilelearner.net/fashion-forecasting-types-purpose-steps/

---

## 2. Akuisisi dan Pemahaman Data

### 2.1 Akuisisi Data
Dataset diunduh menggunakan snippet berikut:

```python
import pandas as pd

# Dataset URL
url = "https://raw.githubusercontent.com/ulfasyabania/Proyek-Pertama-Kirim-Submission-dan-Review/refs/heads/main/fashion_data_2018_2022.xls"

# Load dataset langsung dari URL
df = pd.read_excel(url)

# Menampilkan beberapa baris pertama untuk pemeriksaan awal
print(df.head())
```

Snippet di atas menegaskan bahwa dataset telah siap untuk dieksplorasi lebih lanjut.

### 2.2 Gambaran dan Eksplorasi Data
Dataset ini berisi catatan kuantitatif yang menggambarkan tren fashion selama beberapa tahun. Kolom-kolom utama yang terdapat di dataset mungkin meliputi:
- **Tanggal/Waktu:** Tanggal pencatatan data tren.
- **Penjualan atau Indeks Tren:** Data numerik yang menunjukkan tingkat atau intensitas tren/penjualan.
- **Fitur Tambahan:** Fitur numerik lainnya seperti indikator lokasi, kategori produk, dan lain-lain.

Selama analisis eksplorasi data (EDA), langkah-langkah berikut akan dilakukan:
- **Statistik Ringkasan:** Menilai nilai rata-rata, median, standar deviasi, dan sebagainya.
- **Tipe Data:** Memastikan bahwa data bersifat numerik.
- **Nilai yang Hilang:** Mengidentifikasi dan menangani data yang hilang atau tidak konsisten.
- **Visualisasi:** Membuat grafik garis, histogram, dan boxplot untuk mengungkap pola mendasar serta tren musiman.

*Analisis EDA yang komprehensif ini bertujuan untuk memperoleh pemahaman mendalam mengenai evolusi tren di industri fashion.*

---

## 3. Persiapan Data

### 3.1 Pembersihan Data
- **Menangani Nilai yang Hilang:** Setiap nilai yang hilang yang teridentifikasi selama EDA akan diisi (atau dihapus) untuk memastikan integritas data.
- **Format Data:** Mengonversi kolom tanggal ke format _datetime_ agar analisis deret waktu dapat dilakukan dengan tepat.
- **Deteksi Outlier:** Menggunakan metode seperti boxplot untuk mendeteksi dan, jika perlu, membatasi atau menghapus outlier, karena nilai ekstrem dapat mendistorsi hasil peramalan.

### 3.2 Rekayasa Fitur
- **Fitur Berbasis Waktu:** Mengekstrak informasi hari, bulan, dan tahun; serta membuat fitur musiman jika diperlukan.
- **Agregasi:** Jika dataset terlalu rinci (misalnya data harian) dan berisik, data bisa diagregasi menjadi mingguan atau bulanan agar tren lebih mudah diidentifikasi.
- **Normalisasi:** Menskalakan kolom numerik agar perbedaan rentang nilai tidak mempengaruhi proses pelatihan model.

*Setiap langkah dalam persiapan data didokumentasikan dengan jelas lengkap dengan visualisasi penjelasan terkait pilihan yang diambil.*

---

## 4. Pemodelan

### 4.1 Rasional Pemilihan Model
Untuk peramalan data deret waktu, terdapat dua pendekatan populer:
- **Model Tradisional:** Seperti ARIMA/SARIMA yang terbukti andal untuk data dengan pola tren dan musiman yang jelas.
- **Model Deep Learning:** Seperti jaringan LSTM yang memiliki potensi untuk menangkap ketergantungan temporal non-linear.

Dalam proyek ini, model awal yang digunakan adalah **ARIMA** sebagai baseline peramalan, dengan kemungkinan untuk mengembangkan model deep learning jika diperlukan.

### 4.2 Pengembangan Model
- **Pengujian Stasioneritas:** Menggunakan uji statistik (misalnya, Augmented Dickey-Fuller) untuk memeriksa apakah data bersifat stasioner. Jika tidak, dilakukan transformasi (misalnya, differencing) untuk mencapai stasioneritas.
- **Pengaturan Hiperparameter:** Menentukan urutan ARIMA (p, d, q) menggunakan kriteria AIC/BIC.
- **Pelatihan Model:** Melatih model ARIMA pada data historis.
- **Generasi Peramalan:** Memprediksi nilai tren fashion di masa depan menggunakan model yang telah dilatih.

### 4.3 Pertimbangan Tambahan
Apabila model ARIMA menunjukkan hasil yang menjanjikan tetapi masih memiliki keterbatasan (misalnya dalam menangkap dinamika non-linear yang kompleks), pendekatan deep learning menggunakan LSTM dapat dieksplorasi lebih mendalam.

---

## 5. Evaluasi

### 5.1 Metrik Evaluasi
Metrik berikut akan digunakan untuk mengevaluasi kinerja peramalan:
- **Mean Absolute Error (MAE):** Untuk mengukur besaran rata-rata kesalahan peramalan.
- **Root Mean Squared Error (RMSE):** Untuk memberikan penalti pada kesalahan peramalan yang lebih besar.
- **Inspeksi Visual:** Membandingkan grafik tren aktual dengan hasil prediksi guna menilai kualitas peramalan secara visual.

### 5.2 Validasi Model
- **Pembagian Data:** Dataset akan dibagi secara kronologis menjadi data pelatihan dan pengujian.
- **Cross-Validation:** Jika memungkinkan, teknik seperti cross-validation berbasis deret waktu akan diterapkan.
- **Analisis Residual:** Menganalisis residual untuk memastikan tidak ada pola yang tersisa, yang menunjukkan kecocokan model yang baik.

*Fase evaluasi sangat penting untuk memastikan keandalan model serta untuk membandingkan alternatif pendekatan peramalan.*

---

## 6. Simpulan dan Pekerjaan Selanjutnya

---

### 6.1 Ringkasan Temuan
- Proyek ini berhasil melewati seluruh siklus hidup machine learning—mulai dari eksplorasi data hingga evaluasi model.
- Hasil awal menggunakan ARIMA menunjukkan bahwa model memiliki potensi untuk meramalkan nilai tren fashion dengan akurasi yang memadai.

Dalam proyek peramalan tren fashion ini, saya telah mengimplementasikan dua pendekatan peramalan deret waktu, yaitu:

1. **Model ARIMA:**  
   - ARIMA menggunakan pendekatan statistik yang efektif untuk menangkap pola musiman dan tren linier.  
   - Hasil evaluasi pada data test menunjukkan bahwa model ARIMA menghasilkan MAE sekitar 21,67 dan RMSE sekitar 27,40, yang mengindikasikan ketepatan prediksi yang cukup baik untuk dataset ini.

2. **Model LSTM:**  
   - LSTM merupakan metode deep learning yang mampu menangkap pola non-linear dalam data deret waktu.  
   - Evaluasi pada data validasi memberikan MAE sekitar 25,75 dan RMSE sekitar 31,08.  
   - Meskipun secara kuantitatif hasil LSTM sedikit kurang dibandingkan dengan ARIMA, model ini memiliki potensi yang sangat baik dan dapat dioptimalkan lebih lanjut (misalnya dengan tuning hyperparameter atau penambahan fitur tambahan).

**Hasil Peramalan:**  
Kedua model telah digunakan untuk meramalkan _sales count_ hingga 5 tahun ke depan (60 bulan), dan hasil peramalan divisualisasikan bersama data aktual. Visualisasi tersebut menunjukkan bahwa kedua model mampu mengidentifikasi tren peningkatan yang konsisten, meskipun terdapat perbedaan kecil dalam nilai prediksi.

---

### 6.2 **Rekomendasi dan Pengembangan Selanjutnya:**  
- **Tuning Model:** Lakukan pencarian parameter optimal (misalnya dengan auto_arima untuk ARIMA dan eksperimen arsitektur untuk LSTM) guna meningkatkan akurasi.
- **Integrasi Variabel Eksternal:** Pertimbangkan untuk menambahkan fitur eksternal yang berkaitan dengan tren fashion, seperti data ekonomi atau sentimen media sosial, untuk memperkaya model.
- **Optimalisasi Model LSTM:** Eksplorasi penggunaan dropout, penambahan layer, atau modifikasi konfigurasi LSTM untuk meningkatkan generalisasi model.

Secara keseluruhan, proyek ini menunjukkan bahwa analisis deret waktu dapat diterapkan secara efektif dalam peramalan tren fashion. Hasil dari kedua pendekatan digunakan sebagai dasar untuk pengambilan keputusan di bidang manajemen rantai pasok, strategi pemasaran, dan perencanaan inventaris produk fashion. Meskipun model ARIMA saat ini memberikan hasil yang lebih baik secara kuantitatif, peluang optimasi lebih lanjut pada LSTM menjadikannya kandidat yang menjanjikan untuk eksplorasi mendalam di masa depan.

Proyek ini memberikan dasar yang kuat untuk peningkatan berkelanjutan dan membuka jalur bagi penelitian lanjutan.
---

## 7. Dokumentasi dan Kepatuhan Format Laporan

Draft laporan ini mengikuti format laporan pengajuan yang telah disediakan pada [contoh laporan Dicoding](https://raw.githubusercontent.com/dicodingacademy/contoh-laporan-mlt/refs/heads/main/format_laporan_submission_1.md). Setiap tahap dari proyek didokumentasikan secara mendetail dalam sel teks, sehingga alur kerja serta proses pengambilan keputusan dapat ditelusuri dengan jelas.

---
