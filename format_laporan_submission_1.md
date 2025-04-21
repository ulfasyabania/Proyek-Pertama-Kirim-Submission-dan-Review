# Laporan Proyek Machine Learning - Tb Ulfah Nur Sya'baniah

## Domain Proyek

DKI Jakarta adalah pusat kebudayaan dan seni Indonesia yang memiliki keragaman organisasi kesenian, mencakup seni tradisional, kontemporer, dan seni pertunjukan. Organisasi kesenian ini tidak hanya melestarikan budaya tradisional, tetapi juga berperan penting dalam memperkuat identitas seni modern. Namun, distribusi organisasi kesenian di tingkat kecamatan sering kali dipengaruhi oleh berbagai faktor, seperti populasi penduduk, tingkat partisipasi masyarakat, hingga fasilitas seni di wilayah tersebut.

Melalui proyek ini, kita akan memprediksi pertumbuhan organisasi kesenian di DKI Jakarta untuk memberikan wawasan bagi pembuat kebijakan, penggiat seni, dan komunitas lokal dalam mendukung pelestarian budaya.

**Rubrik/Kriteria Tambahan**:
Mengapa masalah ini penting:
- Memahami faktor-faktor yang memengaruhi pertumbuhan organisasi kesenian dapat membantu menyusun program yang mendukung komunitas seni.
- Prediksi jumlah organisasi seni dapat mempermudah pemerintah dalam mengalokasikan sumber daya, seperti pembiayaan dan pengembangan fasilitas kesenian.

Referensi:
- Statistik Sosial Budaya 2021 oleh Badan Pusat Statistik (BPS) (https://www.bps.go.id/id/publication/2022/06/30/6a2dabc16d556ab9d075f918/statistik-sosial-budaya-2021.html)
- Buku tentang pentingnya seni dalam identitas budaya (https://books.google.co.id/books?hl=id&lr=&id=LXgyEQAAQBAJ&oi=fnd&pg=PR6&dq=seni+dalam+identitas+budaya&ots=6TXszn6yqt&sig=drWyIlXxSecWKJDU3od2nEEgUI0&redir_esc=y#v=onepage&q=seni%20dalam%20identitas%20budaya&f=false)

## Business Understanding

Untuk memahami proyek ini secara menyeluruh, berikut adalah identifikasi masalah, tujuan, dan solusi yang diusulkan:

### Problem Statements

- Bagaimana distribusi jumlah organisasi kesenian di berbagai kecamatan di DKI Jakarta saat ini?
- Apa saja faktor utama yang memengaruhi pertumbuhan organisasi kesenian di DKI Jakarta?
- Dapatkah kita memprediksi jumlah organisasi kesenian di masa depan berdasarkan data historis dan faktor lain seperti rasio penduduk terhadap grup kesenian?

### Goals

- Memahami pola distribusi organisasi kesenian di tingkat kecamatan di DKI Jakarta.
- Mengidentifikasi hubungan antara jumlah penduduk, jenis seni, dan pertumbuhan organisasi kesenian.
- Membangun model prediktif untuk memproyeksikan jumlah organisasi kesenian hingga beberapa tahun ke depan.

**Rubrik/Kriteria Tambahan**:
- Dua solusi utama: Model baseline akan dibandingkan dengan model yang telah diimprovisasi (misalnya, dengan hyperparameter tuning) untuk memastikan prediksi yang optimal.

    ### Solution statements
    - Menggunakan algoritma seperti K-Nearest Neighbor (KNN), Random Forest, dan Boosting untuk memodelkan hubungan antar fitur data dan memprediksi jumlah organisasi kesenian.
    - Melakukan hyperparameter tuning untuk meningkatkan performa model dan memastikan hasil prediksi lebih akurat.
    - Membandingkan performa algoritma berdasarkan metrik evaluasi seperti Mean Squared Error (MSE) untuk memilih model terbaik.

## Data Understanding
Pada bagian ini, kita akan menganalisis dataset yang digunakan untuk proyek prediksi jumlah organisasi kesenian di DKI Jakarta.

**Deskripsi Dataset**
1. **Dataset Pertama**:
   - *Filedata Data Jumlah Organisasi Kesenian Menurut Kecamatan dan Bidang Kegiatan yang Dibina* (https://data.go.id/dataset/dataset/data-jumlah-organisasi-kesenian-menurut-kecamatan-dan-bidang-kegiatan-yang-dibina).
   - Fitur utama:
     - **Kecamatan**: Nama kecamatan di DKI Jakarta.
     - **Bidang Kegiatan Seni**: Jenis seni yang dibina (tradisional, kontemporer, pertunjukan).
     - **Jumlah Organisasi**: Jumlah organisasi kesenian di setiap kecamatan.
     - **Tahun**: Tahun pencatatan data.

2. **Dataset Kedua**:
   - *Filedata Data Grup Kesenian per 10.000 Penduduk* (https://data.go.id/dataset/dataset/data-grup-kesenian-per-10-000-penduduk).
   - Fitur utama:
     - **Kecamatan**: Nama kecamatan di DKI Jakarta.
     - **Jumlah Penduduk**: Total populasi di kecamatan tersebut.
     - **Jumlah Grup Kesenian**: Jumlah grup kesenian di kecamatan tersebut.
     - **Rasio Grup Kesenian**: Jumlah grup kesenian per 10.000 penduduk.

**Eksplorasi Awal Data**
1. **Distribusi Data**:
   - Dataset pertama menunjukkan distribusi jumlah organisasi kesenian berdasarkan kecamatan dan jenis seni.
   - Dataset kedua memberikan informasi tentang kepadatan grup kesenian relatif terhadap populasi.

2. **Visualisasi Data**:
   - **Bar Chart**: Untuk melihat distribusi jumlah organisasi kesenian berdasarkan jenis seni.
   - **Line Chart**: Untuk melihat tren jumlah organisasi kesenian dari tahun ke tahun.
   - **Scatter Plot**: Untuk menganalisis hubungan antara jumlah penduduk dan rasio grup kesenian.

3. **Identifikasi Masalah Data**:
   - **Missing values**: Jika ada data yang hilang, akan diisi dengan nilai median atau rata-rata.
   - **Outliers**: Data yang tidak konsisten akan dianalisis dan ditangani.
   - 
**Rubrik/Kriteria Tambahan**:
- Menambahkan visualisasi tambahan seperti heatmap untuk melihat korelasi antar fitur.
- Menyertakan tabel deskriptif untuk memberikan ringkasan statistik setiap fitur.
  
## Data Preparation
Pada tahap ini, kita akan mempersiapkan dataset agar siap digunakan untuk membangun model prediksi. Berikut langkah-langkah yang dilakukan:

**Langkah 1: Penggabungan Dataset**
- Kedua dataset digabungkan berdasarkan fitur **kecamatan**.
- Dataset pertama berisi jumlah organisasi kesenian berdasarkan jenis seni yang dibina, sementara dataset kedua berisi jumlah grup kesenian dan rasio grup kesenian per 10.000 penduduk.
- Penggabungan dilakukan untuk memperoleh informasi lengkap setiap kecamatan.
```
# Menggabungkan kedua dataset berdasarkan fitur 'kecamatan'
merged_data = pd.merge(dataset_organisasi, dataset_grup_kesenian, on='kecamatan', how='inner')
```

**Langkah 2: Penanganan Missing Values**
- Missing values dianalisis dan diisi dengan rata-rata atau median untuk fitur numerik.
- Fitur kategori yang kosong akan diisi dengan nilai "tidak diketahui".
```
# Mengisi missing values dengan median untuk fitur numerik
merged_data.fillna(merged_data.median(), inplace=True)
```

**Langkah 3: Encoding Fitur Kategori**
- Fitur seperti **bidang kegiatan seni** diubah menjadi variabel numerik menggunakan teknik One-Hot Encoding.
- Hal ini memungkinkan algoritma machine learning memproses data kategori dengan baik.
```
# One-Hot Encoding pada fitur kategori
encoded_data = pd.get_dummies(merged_data, columns=['bidang_kegiatan_seni'])
```

**Langkah 4: Standarisasi Fitur Numerik**
- Fitur numerik seperti **jumlah organisasi**, **jumlah penduduk**, dan **rasio grup kesenian** dinormalisasi agar skala antar fitur seragam.
```
# Normalisasi fitur numerik
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(encoded_data[['jumlah_organisasi', 'jumlah_penduduk', 'rasio_grup_kesenian']])
```

**Langkah 5: Pembagian Dataset**
- Dataset dibagi menjadi **training set** dan **testing set** dengan rasio 80:20 untuk memastikan evaluasi model dilakukan dengan data yang belum pernah digunakan selama pelatihan.
```
# Membagi data menjadi training dan testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_data, encoded_data['target'], test_size=0.2, random_state=42)
```

**Rubrik/Kriteria Tambahan**: 
- Penjelasan tentang alasan mengapa langkah tertentu dilakukan, seperti pentingnya encoding fitur kategori dan normalisasi fitur numerik.
- Menyertakan tabel deskripsi statistik untuk menunjukkan distribusi data setelah preprocessing.

## Modeling
Pada tahap ini, kita akan membangun model prediktif menggunakan dataset yang telah diproses sebelumnya. Model ini bertujuan untuk memprediksi jumlah organisasi kesenian di kecamatan-kecamatan DKI Jakarta berdasarkan fitur-fitur yang relevan.

**Algoritma yang Digunakan**
1. **K-Nearest Neighbor (KNN)**:
   - Algoritma ini memprediksi berdasarkan kesamaan antar data (tetangga terdekat). Cocok untuk dataset kecil dengan pola sederhana.
  
2. **Random Forest**:
   - Memanfaatkan ensemble learning dengan banyak pohon keputusan. Algoritma ini mampu menangkap pola kompleks dan memberikan hasil yang lebih stabil.

3. **Boosting Algorithm**:
   - Menggunakan iterasi untuk meningkatkan akurasi dengan fokus pada error sebelumnya. Cocok untuk meningkatkan performa secara signifikan.

**Hyperparameter Tuning**
Setiap algoritma diterapkan dengan pengaturan hyperparameter optimal untuk meningkatkan akurasi prediksi:
1. **KNN**:
   - Jumlah tetangga (K) divariasikan antara 5–15 untuk menemukan nilai optimal.

2. **Random Forest**:
   - Jumlah pohon (n_estimators) disesuaikan antara 50–200.
   - Kedalaman maksimum (max_depth) divariasikan antara 5–20.

3. **Boosting Algorithm**:
   - Learning rate divariasikan antara 0.01–0.1.
   - Jumlah estimator (n_estimators) disesuaikan antara 100–300.
   - 
**Implementasi Model**
Berikut contoh potongan kode implementasi untuk algoritma Random Forest:
```
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Membuat model Random Forest
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluasi model
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# Menghitung MSE
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

print("MSE Train:", mse_train)
print("MSE Test:", mse_test)
```
**Evaluasi Model**
Setiap algoritma dievaluasi berdasarkan nilai Mean Squared Error (MSE) untuk dataset training dan testing. Model dengan nilai MSE terendah dipilih sebagai model terbaik untuk prediksi.

**Rubrik/Kriteria Tambahan**: 
- Jelaskan kelebihan dan kekurangan setiap algoritma berdasarkan hasil evaluasi.
- Sertakan visualisasi seperti grafik perbandingan hasil prediksi dan nilai sebenarnya untuk dataset testing.

## Evaluation
Pada tahap ini, kita akan mengevaluasi performa model yang telah dibuat menggunakan dataset testing. Evaluasi dilakukan dengan menghitung nilai error menggunakan metrik **Mean Squared Error (MSE)** dan membandingkan hasil prediksi dengan data aktual.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

