---

## Laporan Proyek Machine Learning – Tb Ulfah Nur Sya'baniah

---

## **Prediksi Hasil Panen Optimal melalui Machine Learning**

## 1. Latar Belakang

Di tengah transformasi digital yang merambah ke seluruh sektor, pertanian modern kini semakin mengandalkan data dan teknologi untuk mengambil keputusan strategis. Peningkatan produktivitas lahan dan efisiensi penggunaan input (seperti pupuk) merupakan tantangan besar, terutama ketika banyak faktor eksternal seperti kondisi iklim dan jenis tanah berperan secara simultan dalam menentukan hasil panen. Proyek ini hadir untuk menjawab tantangan tersebut melalui pengembangan model machine learning yang mampu memprediksi crop yield secara akurat.

Dataset yang digunakan mencakup informasi mendalam seperti Temperature, Humidity, Moisture yang menggambarkan kondisi iklim mikro selama pertumbuhan tanaman, serta atribut agronomi seperti Soil Type, Crop Type, Nitrogen, Potassium, Phosphorous, dan Fertilizer Name. Kombinasi informasi ini mencerminkan variabel-variabel esensial yang mempengaruhi produktivitas tanaman. Dengan memanfaatkan data tersebut, model prediktif diharapkan tidak hanya mampu memperkirakan hasil panen, tetapi juga memberikan insight untuk mengoptimalkan penggunaan pupuk, penyesuaian teknik irigasi, dan perbaikan manajemen lahan.

Selain memberikan nilai prediktif secara numerik (misalnya, prediksi crop yield dalam ton per hektar), analisis ini juga berkontribusi dalam memahami secara mendalam peran masing-masing faktor. Misalnya, bagaimana variasi suhu atau kelembapan mempengaruhi ketersediaan nutrisi dalam tanah, serta interaksi antara jenis pupuk dengan tipe tanah dan tanaman. Insight semacam ini sangat vital untuk petani dan pemangku kebijakan, guna mengimplementasikan strategi pertanian yang ramah lingkungan sekaligus meningkatkan hasil produksi.

Selain aspek teknis, proyek ini juga mendukung prinsip precision agriculture, suatu pendekatan modern yang mengintegrasikan input data real-time dan historis guna memberikan rekomendasi penanganan lahan yang paling optimal. Di era pertanian yang menghadapi dinamika perubahan iklim dan keterbatasan sumber daya, solusi berbasis machine learning menjadi alat bantu penting dalam pengambilan keputusan agroekonomi yang berkelanjutan.

Dengan latar belakang inilah, proyek ini bertujuan menggabungkan data variabel iklim, tanah, dan input pupuk ke dalam satu model komprehensif yang tidak hanya meningkatkan akurasi prediksi hasil panen, tetapi juga memberikan dasar bagi strategi optimasi pertanian di masa depan.

---

## Referensi

**Bhagat, D., Shah, S., & Gupta, R. K. (2024).**  
   *Crop Yield Prediction Using Machine Learning Approaches.*  
   Makalah konferensi ini mengeksplorasi penggunaan beberapa model regresi seperti Random Forest Regressor dan Linear Regression dalam memprediksi hasil panen. Studi ini menguraikan langkah-langkah pengolahan data, proses seleksi fitur, dan evaluasi metrik performa (misalnya MAE, MSE, RMSE, dan R²), yang membantu mengoptimalkan model prediksi untuk aplikasi praktis di lapangan.  
   *(Sumber: SpringerLink, [Link prosiding](https://link.springer.com/chapter/10.1007/978-3-031-62217-5_6))*

---

## 2. Business Understanding
Dalam tahap ini, kita mendefinisikan tujuan bisnis dan konteks strategis di mana model machine learning akan diaplikasikan. Proyek prediksi hasil panen ini bertujuan menyelaraskan analisis data agronomi dengan kebutuhan nyata di sektor pertanian, guna meningkatkan efisiensi penggunaan input dan mengoptimalkan hasil produksi. Berikut adalah beberapa poin kunci yang mendasari pemahaman bisnis proyek ini:

1. **Tujuan dan Manfaat Bisnis**  
   - Meningkatkan hasil panen dengan memberikan rekomendasi penggunaan pupuk dan teknik manajemen lahan yang tepat sehingga sumber daya dapat dimanfaatkan secara optimal.  
   - Memberikan insight berbasis data yang dapat membantu petani, pengusaha pertanian, dan stakeholder lainnya dalam merencanakan strategi operasional misalnya, pemilihan jenis pupuk atau adaptasi terhadap kondisi iklim mikro yang berubah.  
   - Mengurangi biaya operasional melalui optimalisasi input, serta mendukung praktik pertanian berkelanjutan di tengah dinamika perubahan iklim.

2. **Identifikasi Stakeholders**  
   - Mereka Petani dan Operator Lahan sebagai pengguna utama yang membutuhkan rekomendasi praktis untuk meningkatkan produktivitas.  
   - Pihak-pihak yang berkepentingan Distributor Pupuk dan Lembaga Agrikultur dalam memastikan penggunaan input pertanian yang efisien dan tepat guna.  
   - Pemerintah dan Pembuat Kebijakan untuk pengembangan program-program ketahanan pangan dan strategi penanganan dampak perubahan iklim.  
   - Investor dan Perusahaan Teknologi Agrikultur yang berpotensi memanfaatkan solusi prediktif ini dalam skala yang lebih luas.

3. **Problem Statement**  
   - Bagaimana menentukan dosis dan jenis pupuk yang optimal agar setiap lahan mendapatkan perlakuan yang sesuai dengan kondisi tanah, iklim, dan jenis tanaman?
   - Bagaimana memanfaatkan variabel seperti temperatur, kelembapan, moisture, tipe tanah, dan kandungan nutrisi untuk memprediksi hasil panen secara akurat dan mendukung keputusan agronomi?

4. **Goals**  
   - **Model Prediktif Berkinerja Tinggi:** Mencapai metrik evaluasi yang memadai (misalnya, R² score tinggi dengan MAE/MSE rendah) sehingga model benar-benar representatif dalam memproyeksikan hasil panen.  
   - **Rekomendasi Operasional:** Mendapatkan insight tentang kontribusi masing-masing variabel (seperti pengaruh pupuk, kondisi iklim, dan tipe tanah) guna memberikan rekomendasi yang dapat diimplementasikan secara langsung oleh para praktisi pertanian.  
   - **Dampak Bisnis Nyata:** Misalnya, pengurangan biaya operasional, peningkatan output pertanian, dan peningkatan profitabilitas berkat keputusan yang lebih terinformasi.

---  

**Solution Statements:**  
1. **Data Preprocessing dan Feature Engineering**  
   - **Pembersihan dan Transformasi Data:**  
     Langkah pertama adalah memastikan kualitas data dengan menangani nilai yang hilang dan outlier. Variabel kategorikal seperti *Soil Type, Crop Type,* dan *Fertilizer Name* di-encode menjadi format numerik, sedangkan fitur numerik (seperti *Temperature, Humidity, Moisture, Nitrogen, Potassium,* dan *Phosphorous*) dinormalisasi agar skala nilainya konsisten.  
   - **Pembuatan Fitur Tambahan:**  
     Jika diperlukan, dapat dikembangkan fitur baru (misalnya, indeks kesuburan tanah yang merupakan kombinasi nilai nutrisi) guna menangkap informasi yang lebih mendalam dari data mentah.  
   
2. **Analisis Data (Exploratory Data Analysis - EDA)**  
   - **Visualisasi dan Korelasi:**  
     Dilakukan visualisasi distribusi fitur dan hubungan antar variabel untuk mengidentifikasi pola dan korelasi yang potensial antara parameter iklim, karakteristik tanah, input pupuk, dan hasil panen.  
   - **Insight Awal:**  
     EDA juga memberikan dasar untuk menentukan fitur mana yang berperan signifikan dalam mempengaruhi *Crop Yield*, yang kemudian menjadi fokus utama dalam proses pemodelan.

3. **Pengembangan dan Evaluasi Model Baseline**  
   - **Baseline Model:**  
     Mulai dengan Linear Regression sebagai model dasar. Evaluasi performa menggunakan metrik seperti Mean Absolute Error (MAE), Mean Squared Error (MSE), dan R² score pada data uji (test set).  
   - **Pemetaan Performa:**  
     Hasil evaluasi baseline memberikan gambaran tentang efektivitas model awal dan menentukan apakah diperlukan model yang lebih kompleks untuk mencapai akurasi yang optimal.

4. **Eksperimen dengan Model Lanjutan dan Hyperparameter Tuning**  
   - **Model Alternatif:**  
     Selanjutnya, uji model-model lain seperti Random Forest Regression, Decision Tree Regression, atau algoritma ensemble seperti Gradient Boosting untuk meningkatkan performa prediksi.  
   - **Pengoptimalan Parameter:**  
     Lakukan tuning hyperparameter menggunakan teknik Grid Search atau Randomized Search bersama dengan cross-validation, guna memastikan model tidak overfitting dan memiliki generalisasi yang baik.

5. **Interpretabilitas dan Penyampaian Insight**  
   - **Analisa Fitur Penting:**  
     Dengan menggunakan teknik interpretabilitas seperti feature importance, SHAP (SHapley Additive exPlanations) atau LIME, analisis akan dilakukan untuk mengidentifikasi dan mengkuantifikasi kontribusi masing-masing fitur terhadap prediksi yield.  
   - **Rekomendasi Praktis:**  
     Insight ini akan diterjemahkan menjadi rekomendasi bagi petani atau pihak terkait, seperti strategi penggunaan pupuk optimal atau penyesuaian teknik irigasi, sehingga keputusan operasional dapat didasarkan pada bukti empiris.

6. **Implementasi dan Deployment**  
   - **Menyimpan Model:**  
     Model terbaik yang telah di-tuning dan tervalidasi akan disimpan menggunakan `pickle` atau `joblib` sehingga dapat digunakan kembali untuk prediksi di masa depan.  
   - **Integrasi dalam Sistem Produksi:**  
     Pada tahap akhir, solusi dapat diintegrasikan ke dalam aplikasi berbasis web atau dashboard interaktif (misalnya dengan Streamlit atau Flask) untuk memberikan akses real-time kepada pengguna akhir.

---

## 3. Data Understanding

Dataset yang digunakan dalam proyek ini merupakan basis data untuk prediksi hasil panen tanaman (crop yield). Dataset ini mencakup 10 fitur utama yang mewakili kondisi lingkungan, karakteristik tanah, dan input pertanian, yang semuanya berpotensi memengaruhi produktivitas tanaman.

**Detail struktur data:**

- **Jumlah Baris:** 8000  
- **Jumlah Kolom:** 30 

Proses verifikasi jumlah data dapat dilakukan dengan cara berikut:

```python
import pandas as pd

url = "https://raw.githubusercontent.com/ulfasyabania/Proyek-Pertama-Kirim-Submission-dan-Review/refs/heads/main/data_core_with_yield.csv"

print("Dataset shape:", df.shape)
```

**Fitur-Fitur Utama dalam Dataset:**
- **Temparature:**  
  Mewakili suhu lingkungan (dalam °C) yang dapat memengaruhi proses metabolisme dan pertumbuhan tanaman.

- **Humidity:**  
  Menunjukkan tingkat kelembapan udara; kondisi kelembapan yang tepat penting untuk proses evapotranspirasi dan kesehatan tanaman.

- **Moisture:**  
  Menggambarkan kandungan kelembapan dalam tanah. Ketersediaan air adalah faktor kritis dalam penyerapan nutrisi serta mendukung fotosintesis.

- **Soil Type:**  
  Mengklasifikasikan jenis tanah. Sifat fisik dan kimia tanah, seperti struktur dan kadar bahan organik, menentukan ketersediaan nutrisi dan kemampuan tanah mengalirkan air.

- **Crop Type:**  
  Menyatakan jenis tanaman yang ditanam, misalnya padi, jagung, atau gandum. Setiap jenis tanaman memiliki kebutuhan nutrisi dan toleransi yang berbeda terhadap kondisi lingkungan.

- **Nitrogen, Potassium, Phosphorous:**  
  Merupakan unsur hara utama yang sangat penting untuk pertumbuhan tanaman. Variasi jumlah dari unsur-unsur ini mencerminkan kekayaan dan dosis pemupukan yang diaplikasikan.

- **Fertilizer Name:**  
  Menunjukkan jenis atau merek pupuk yang digunakan. Informasi ini dapat membantu mengidentifikasi dampak spesifik dari komposisi pupuk terhadap hasil panen.

- **Crop Yield:**  
  Merupakan variabel target yang mengukur hasil panen (biasanya dalam satuan ton/ha). Outcome ini adalah indikator utama keberhasilan praktik pertanian dan menjadi fokus model prediksi.

**Langkah-langkah Eksplorasi Data:**

1. **Pemeriksaan Struktur Data dan Statistik Deskriptif:**  
   Memuat dataset menggunakan pustaka seperti Pandas, melakukan pengecekan jumlah baris dan kolom, tipe data masing-masing fitur, serta menghitung statistik dasar (rata-rata, median, standar deviasi, dsb).

2. **Identifikasi Missing Values dan Outlier:**  
   Memeriksa apakah terdapat nilai yang hilang atau data yang tidak konsisten yang perlu dibersihkan agar hasil analisis tidak bias.

3. **Visualisasi Distribusi dan Korelasi:**  
   Menggunakan histogram, box plot, dan heatmap untuk memahami distribusi masing-masing variabel serta menganalisis korelasi antara fitur-fitur dan target *Crop Yield*. Hal ini memberikan insight awal mengenai fitur apa yang memiliki pengaruh kuat terhadap hasil panen.

4. **Analisis Kategori:**  
   Melakukan eksplorasi pada fitur kategorikal seperti *Soil Type, Crop Type,* dan *Fertilizer Name* untuk memastikan distribusinya cukup representatif dan tidak menyebabkan bias dalam pemodelan.

Melalui pemahaman mendalam terhadap struktur dan karakteristik data ini, kita dapat merancang strategi untuk tahap pre-processing dan feature engineering yang tepat. Hal ini akan memaksimalkan potensi model prediktif dalam mengoptimalkan hasil panen dan memberikan basis rekomendasi yang actionable bagi para praktisi pertanian.

**Sumber Data:**  
Dataset diperoleh dari Kaggle, yaitu "Crop Yield Prediction Dataset" oleh anshsinghal3107.  
[Link ke dataset](https://www.kaggle.com/datasets/anshsinghal3107/crop-yield-prediction-dataset?select=data_core_with_yield.csv)

---

## 4. Data Preparation
Tujuan utama pada tahap Data Preparation adalah untuk membersihkan, mentransformasikan, dan menyiapkan dataset sehingga siap untuk proses pemodelan. Pada tahap ini, kita memastikan bahwa data yang akan digunakan adalah bersih, terstruktur, dan konsisten. Perlu dicatat bahwa proses-proses seperti memuat dataset dan pemeriksaan eksploratori dasar telah dilakukan pada tahap Data Understanding. Di sini, fokus kita adalah pada transformasi dan penyempurnaan data sehingga dapat mendukung pembangunan model secara efektif.

Dataset yang digunakan (yaitu, *data_core_with_yield.csv*) memiliki **8000 baris dan 30 kolom**. Informasi ini penting karena menunjukkan skala data yang dihadapi dan memastikan bahwa langkah-langkah preparasi diterapkan secara konsisten pada seluruh dataset.

### Langkah-Langkah Data Preparation

1. **Pemeriksaan dan Penanganan Data Hilang / Outlier**  
   Nilai yang hilang atau tidak konsisten dapat menyebabkan bias saat pelatihan model dan mengganggu algoritma pembelajaran.  
   Langkah:  
   ```python
   # Melihat jumlah nilai null pada masing-masing kolom
   print(df.isnull().sum())
   # Jika ada missing value, misalnya dapat di-impute menggunakan metode forward fill:
   # df = df.fillna(method='ffill')
   ```
   
2. **Pengecekan Tipe Data dan Konversi**  
   Pastikan setiap kolom memiliki tipe data yang sesuai agar proses numerik (misalnya scaling) dan analisis berjalan lancar.  
   Langkah:  
   ```python
   print(df.dtypes)
   # Jika diperlukan, misalnya:
   # df['Crop Yield'] = pd.to_numeric(df['Crop Yield'], errors='coerce')
   ```

3. **Encoding Variabel Kategorikal**  
   Variabel kategorikal (seperti *Soil Type*, *Crop Type*, dan *Fertilizer Name*) perlu dikonversi ke format numerik agar model dapat memprosesnya tanpa mengasumsikan adanya urutan (ordinalitas) yang tidak relevan.  
   Langkah:  
   ```python
   df = pd.get_dummies(df, columns=['Soil Type', 'Crop Type', 'Fertilizer Name'])
   ```
   
4. **Normalisasi/Standarisasi Fitur Numerik**  
   Fitur numerik seperti *Temperature*, *Humidity*, *Moisture*, dan nilai nutrisi memiliki skala yang berbeda-beda. Menggunakan StandardScaler membantu mengubahnya ke rentang yang serupa, sehingga model tidak condong pada fitur-fitur dengan nilai yang lebih besar.  
   Langkah:
   ```python
   from sklearn.preprocessing import StandardScaler

   num_cols = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
   scaler = StandardScaler()
   df[num_cols] = scaler.fit_transform(df[num_cols])
   ```

5. **Pembagian Dataset Menjadi Training dan Testing Set**  
   Untuk mengevaluasi generalisasi model, dataset dibagi menjadi data pelatihan (untuk melatih model) dan data pengujian (untuk mengecek performa pada data yang belum terlihat oleh model).  
   Langkah:
   ```python
   from sklearn.model_selection import train_test_split

   X = df.drop('Crop Yield', axis=1)
   y = df['Crop Yield']

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   print("Training set shape:", X_train.shape)
   print("Test set shape:", X_test.shape)
   ```
---

### Alasan Teknik Data Preparation

- **Penanganan Missing Value / Outlier:**  
  Untuk memastikan bahwa model tidak mendapatkan input yang tidak lengkap atau ekstrem yang dapat mengganggu proses pembelajaran, sehingga validitas dan keakuratan prediksi dapat dijamin.

- **Konversi Tipe Data:**  
  Transformasi tipe data diperlukan agar fungsi-fungsi matematis dan algoritma pembelajaran dapat dijalankan tanpa error, serta memastikan konsistensi data.

- **Encoding Variabel Kategorikal:**  
  One-Hot Encoding digunakan untuk menghindari kesalahan dalam interpretasi model terhadap variabel nominal. Dengan mengonversi ke format numerik, setiap kategori diperlakukan secara independen.

- **Normalisasi/Standarisasi:**  
  Mengurangi perbedaan skala antar fitur memudahkan model dalam menemukan pola dan mengoptimalkan proses training.

- **Pembagian Dataset:**  
  Pemisahan data secara eksplisit memastikan bahwa evaluasi model dilakukan terhadap data baru yang belum pernah dilihat sebelumnya, sehingga mencerminkan performa model di dunia nyata.

Tahapan-tahapan di atas memastikan bahwa dataset siap digunakan untuk proses pemodelan, sehingga model yang dikembangkan dapat memberikan prediksi yang akurat dan reliable. Proses ini juga mendokumentasikan langkah-langkah penting untuk menjamin bahwa data yang masuk ke model terjaga konsistensinya, yang pada akhirnya mendukung keputusan strategis berbasis data.

---

## 5. Modeling
Pada tahap ini, membangun model prediktif untuk memprediksi *Crop Yield* berdasarkan variabel-variabel input (misalnya, temperatur, kelembapan, moisture, tipe tanah, dan kandungan nutrisi). Proses modeling dimulai dari pembuatan model baseline hingga eksperimen dengan model yang lebih kompleks menggunakan pendekatan ensemble. Di bawah ini dijelaskan cara kerja masing-masing algoritma dan parameter yang digunakan.

#### 1. Baseline Model: Linear Regression

**Cara Kerja:**  
Model Linear Regression berupaya menemukan hubungan linear antara variabel independen (input) dan variabel dependen (target) dengan meminimalkan jumlah kesalahan kuadrat (least squares). Model ini menghitung koefisien (bobot) untuk setiap fitur sehingga prediksi merupakan kombinasi linear dari fitur-fitur tersebut. Dalam implementasi scikit-learn, algoritma ini berjalan dengan parameter default tanpa regularisasi tambahan.

**Parameter:**  
- **fit_intercept** (default: `True`): Model secara otomatis menghitung intercept (bias).  
- **normalize** (default: `False`): Skalasi input tidak dilakukan secara otomatis karena sudah melakukan scaling sebelumnya.  
- **n_jobs** (default: `None`): Tidak menggunakan paralelisasi.

**Kode Implementasi Linear Regression:**

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Membuat dan melatih model Linear Regression dengan parameter default
lr = LinearRegression()  # fit_intercept=True, normalize=False by default
lr.fit(X_train, y_train)

# Melakukan prediksi pada data testing
y_pred_lr = lr.predict(X_test)

# Evaluasi performa model (ini mencakup perhitungan MAE, MSE, dan R²)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("Linear Regression Performance:")
print(f"MAE: {mae_lr:.2f}, MSE: {mse_lr:.2f}, R2 Score: {r2_lr:.2f}")
```
## **Insights:**

Berikut beberapa insight yang dapat diambil:

- **R² Score 0.93:**  
  Nilai ini menunjukkan bahwa model Linear Regression mampu menjelaskan sekitar 93% variasi (variance) dalam data hasil panen. Artinya, sebagian besar variabilitas output (Crop Yield) berhasil di capture oleh model dari kombinasi fitur-fitur input yang tersedia. Ini merupakan indikasi performa yang sangat baik dalam konteks model regresi.

- **Mean Absolute Error (MAE) 420.32:**  
  MAE mengindikasikan bahwa rata-rata kesalahan prediksi model adalah sekitar 420 unit (misalnya, dalam satuan panen tertentu seperti kg/ha atau ton/ha, tergantung definisi dataset). Dengan mempertimbangkan skala nilai hasil panen dalam dataset, nilai ini bisa dianggap cukup baik jika selisih prediksi rata-rata tidak terlalu signifikan secara proporsional.

- **Mean Squared Error (MSE) 296193.51:**  
  MSE memberikan penalti lebih berat terhadap kesalahan yang besar karena error dikuadratkan. Meskipun nilainya tampak tinggi, hal ini biasa terjadi karena sensitivitas MSE terhadap outlier. Untuk interpretasi yang lebih intuitif, bisa menghitung Root Mean Squared Error (RMSE) yang kira-kira sebesar √296193.51 ≈ 543. Jika RMSE relatif sebanding dengan MAE, hal ini dapat mengindikasikan bahwa kesalahan prediksi tidak memiliki outlier yang ekstrem, atau setidaknya tidak dominan.

**Kesimpulan Insight:**

- **Model Fit yang Baik:** R² Score sebesar 0.93 mengindikasikan bahwa model Linear Regression memberikan fitting yang sangat baik untuk data ini.
- **Kesalahan Prediksi:** MAE dan RMSE menunjukkan bahwa secara rata-rata prediksi model menyimpang sekitar 420 hingga 543 unit dari nilai aktual, tergantung metrik yang digunakan.
- **Baseline yang Kuat:** Dengan performa seperti ini, Linear Regression menjadi baseline yang kuat untuk prediksi hasil panen. Meskipun demikian, perlu dipertimbangkan pula untuk mengeksplorasi model non-linear (misalnya, Random Forest atau Gradient Boosting) untuk melihat apakah peningkatan kinerja lebih lanjut dapat dicapai, terutama jika terdapat pola non-linear yang kompleks dalam data.

---
Dalam model ini, parameter default sudah mencukupi untuk membangun baseline. Karena model linear relatif sederhana, hasilnya juga memberikan gambaran awal kapan perlu berpindah ke model yang lebih kompleks.

#### 2. Model Ensemble: Random Forest Regressor

**Cara Kerja:**  
Random Forest Regression merupakan algoritma ensemble yang menggabungkan banyak pohon keputusan (decision trees) dan mengambil rata-rata (untuk regresi) dari prediksi masing-masing pohon. Setiap pohon dibangun dengan subset data acak (bagging) dan subset fitur acak, sehingga membantu mengurangi overfitting dan meningkatkan generalisasi. Model ini dapat menangkap hubungan non-linear antar fitur secara efektif.

**Parameter Utama dan Nilai Default:**
- **n_estimators** (default: `100`): Jumlah pohon yang dibangun. Dalam eksperimen, nilai ini dapat dioptimalkan (misalnya, 100 atau 200).
- **max_depth** (default: `None`): Tidak ada batasan kedalaman pohon, namun, membatasi kedalaman (misalnya, 10 atau 20) dapat mencegah overfitting.
- **min_samples_split** (default: `2`): Jumlah minimum sampel untuk membagi node.
- **min_samples_leaf** (default: `1`): Jumlah minimum sampel di setiap daun.
- **random_state**: Digunakan untuk reproduksibilitas (misalnya, `42`).

**Kode Implementasi Random Forest (dengan parameter default dan tuning):**

```python
from sklearn.ensemble import RandomForestRegressor

# Membuat dan melatih model Random Forest dengan parameter default (kecuali random_state)
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Prediksi dan evaluasi Random Forest
y_pred_rf = rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Performance (Default Parameters):")
print(f"MAE: {mae_rf:.2f}, MSE: {mse_rf:.2f}, R2 Score: {r2_rf:.2f}")
```

Untuk memastikan model terbaik, dilakukan hyperparameter tuning guna mencari konfigurasi optimal.

**Hyperparameter Tuning dengan GridSearchCV:**

```python
from sklearn.model_selection import GridSearchCV

# Menentukan grid parameter untuk Random Forest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_rf.fit(X_train, y_train)

print("Best parameters for Random Forest:", grid_rf.best_params_)

# Menggunakan model terbaik dari hasil tuning
best_rf = grid_rf.best_estimator_
y_pred_best = best_rf.predict(X_test)
mae_best = mean_absolute_error(y_test, y_pred_best)
mse_best = mean_squared_error(y_test, y_pred_best)
r2_best = r2_score(y_test, y_pred_best)

print("Best Random Forest Performance:")
print(f"MAE: {mae_best:.2f}, MSE: {mse_best:.2f}, R2 Score: {r2_best:.2f}")
```

**Penjelasan Tuning:**  
Pada tahap tuning, grid parameter digunakan untuk menguji kombinasi yang berbeda dari *n_estimators*, *max_depth*, *min_samples_split*, dan *min_samples_leaf*.  
- **n_estimators:** Menentukan banyaknya pohon dalam ensemble, yang dapat mempengaruhi kestabilan prediksi.  
- **max_depth:** Membatasi kedalaman pohon membantu mencegah overfitting.  
- **min_samples_split** dan **min_samples_leaf:** Menjamin bahwa setiap pembagian decision tree tidak terlalu spesifik terhadap data pelatihan sehingga meningkatkan generalisasi.

## **Insights:**

Berikut adalah beberapa insights yang dapat diambil dari output tuning model Random Forest dengan GridSearchCV:

1. **Optimalisasi Hyperparameter:**  
   - **Best parameters:**  
     `{'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 200}`  
     GridSearchCV telah mengidentifikasi bahwa model dengan maksimum kedalaman (max_depth) sebesar 10, minimal 5 sampel untuk melakukan split (min_samples_split = 5), dan minimal 2 sampel di setiap leaf (min_samples_leaf = 2) bersama dengan 200 pohon (n_estimators = 200) menghasilkan performa terbaik.  
   - **Makna Parameter:**  
     - **max_depth=10:** Membatasi kedalaman pohon sehingga mencegah overfitting dan memastikan bahwa setiap pohon tidak terlalu kompleks.  
     - **min_samples_split=5 & min_samples_leaf=2:** Menjamin bahwa node dalam pohon hanya akan di split jika memiliki cukup data, yang membantu dalam stabilisasi model dan mengurangi variabilitas prediksi.  
     - **n_estimators=200:** Menggunakan 200 pohon membantu meningkatkan robustnes model dengan mengurangi variansi.

2. **Evaluasi Performa Model Terbaik:**  
   - **MAE:** 426.08  
     Rata-rata kesalahan prediksi model adalah sekitar 426 unit.  
   - **MSE:** 311,815.73  
     MSE yang diperoleh mengindikasikan penalti yang lebih besar untuk error yang besar meskipun nilainya sedikit lebih rendah jika dibandingkan dengan pengaturan default.  
   - **R² Score:** 0.92  
     Model Tuned Random Forest mampu menjelaskan sekitar 92% variasi dalam data target (Crop Yield).  
     
   Performa dari model optimum ini menunjukkan sedikit perbaikan (misalnya, MAE menurun dari 430.11 ke 426.08) namun secara keseluruhan metrik seperti R² tetap berada di posisi yang mendekati 0.92.

3. **Bandingkan dengan Model Baseline:**  
   - **Model Linear Regression** sebelumnya menunjukkan R² Score sebesar 0.93 dengan MAE sekitar 420.32.  
   - **Perbandingan:** Meskipun model Random Forest yang dituning menghasilkan performa yang cukup baik (R² 0.92), hasilnya hampir sebanding dengan model Linear Regression. Hal ini mengindikasikan bahwa hubungan prediktif pada dataset ini relatif mudah ditangkap dan mungkin tidak terlalu kompleks sehingga model linier sudah sangat kompetitif.

Dengan demikian, tuning hyperparameter memberikan sedikit perbaikan terhadap model Random Forest, namun perbandingan dengan pendekatan model linier menunjukkan bahwa edge yang didapat dari penerapan ensemble pada dataset ini masih marginal. Langkah selanjutnya bisa mempertimbangkan eksplorasi metode lain atau penambahan data fitur agar model dapat meningkatkan kinerjanya secara signifikan.

---

### Kesimpulan Modeling

- **Linear Regression:**  
  Digunakan sebagai model baseline dengan parameter default. Cara kerjanya yang sederhana dan interpretabilitas tinggi menjadikannya acuan awal untuk mengukur performa model prediksi.
  
- **Random Forest Regression:**  
  Model ensemble yang menangkap hubungan non-linear secara efektif. Dengan mengoptimalkan parameter utama melalui GridSearchCV, model ini dapat meningkatkan akurasi prediksi *Crop Yield* dan dapat mengatasi variabilitas data pertanian.
   
---

## 6. Evaluation
Pada tahap evaluasi, mengukur seberapa baik model prediktif dalam menangkap hubungan antara variabel-variabel input dan target (*Crop Yield*). Evaluasi dilakukan dengan menggunakan metrik regresi yang umum, yaitu Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), dan R² Score. Berikut penjelasan dan formulanya:

#### 1. Metrik Evaluasi dan Formula

- **Mean Absolute Error (MAE):**

  MAE mengukur rata-rata perbedaan absolut antara nilai prediksi dan nilai aktual.  
  **Rumus:**
  
$$
MAE = \frac{1}{n}\sum_{i=1}^{n} \left| y_i - \hat{y}_i \right|
$$

  *Contoh:* Pada model Linear Regression, MAE yang diperoleh adalah 420.32, yang berarti rata-rata kesalahan prediksi adalah sekitar 420.32 unit.

- **Mean Squared Error (MSE):**

  MSE mengukur rata-rata kuadrat selisih antara nilai aktual dan prediksi. Metrik ini memberikan penalti yang lebih tinggi pada error yang besar.  
  **Rumus:**
    
$$
MSE = \frac{1}{n}\sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2
$$

  *Contoh:* Model Linear Regression menghasilkan MSE sekitar 296193.51.

- **Root Mean Squared Error (RMSE):**

  RMSE merupakan akar kuadrat dari MSE sehingga nilai yang dihasilkan berada pada satuan yang sama dengan target.  
  **Rumus:**
  
$$
RMSE = \sqrt{MSE}
$$

- **R² Score:**

  R² Score mengukur proporsi variansi pada target yang berhasil dijelaskan oleh model. Nilai yang mendekati 1 berarti model menjelaskan sebagian besar variasi data.  
  **Rumus:**
  
$$
R^2 = 1 - \frac{\sum_{i=1}^{n}\left(y_i - \hat{y}_i\right)^2}{\sum_{i=1}^{n}\left(y_i - \bar{y}\right)^2}
$$

  *Contoh:* Dengan nilai R² dari 0.93, model Linear Regression dapat menjelaskan 93% variasi data.

#### 2. Contoh Kode Evaluasi

Berikut adalah implementasi kode evaluasi yang digunakan:

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Misalkan y_test adalah nilai aktual dan y_pred_best hasil prediksi dari model terbaik
mae = mean_absolute_error(y_test, y_pred_best)
mse = mean_squared_error(y_test, y_pred_best)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_best)

print("Evaluasi Model Terbaik:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")
```

#### 3. Hasil Evaluasi dan Interpretasi

Sebagai contoh, pada model baseline Linear Regression diperoleh:
- **MAE:** 420.32  
  Rata-rata kesalahan prediksi sebesar 420.32 unit menunjukkan seberapa besar penyimpangan secara absolut antara nilai aktual dengan prediksi.
- **MSE:** 296193.51  
  Nilai MSE yang tinggi menunjukkan bahwa prediksi yang sangat meleset mendapatkan penalti lebih besar.
- **R² Score:** 0.93  
  Ini berarti model mampu menjelaskan 93% variansi pada *Crop Yield*, yang mengindikasikan bahwa model prediktif cukup andal.

Interpretasi metrik ini memberikan dasar yang kuat dalam menilai kualitas model:  
- Error yang masuk akal (MAE dan RMSE) dan nilai R² yang tinggi menunjukkan bahwa model mampu menangkap pola utama pada data.  
- Penjelasan lengkap mengenai rumus dari masing-masing metrik juga memudahkan pembaca untuk memahami bagaimana setiap error dihitung, sehingga meningkatkan kredibilitas analisis.
- 
#### 4. Visualisasi Hasil Prediksi
- **Scatter Plot:**  
  Membandingkan nilai aktual versus nilai prediksi agar bisa melihat apakah model cenderung underestimasi atau overestimasi dalam rentang tertentu.
- **Residual Plot:**  
  Menganalisis sebaran residual (selisih antara nilai aktual dan nilai prediksi) untuk memastikan tidak terdapat pola sistematik. Distribusi residual yang acak dan simetris merupakan indikasi model yang baik.

#### 5. Cross-Validation
Untuk memastikan model yang dibangun memiliki performa yang konsisten dan generalisasi yang baik, lakukan evaluasi dengan cross-validation (misalnya, K-Fold Cross Validation). Teknik ini membagi data menjadi beberapa subset sehingga model diuji secara bergantian pada bagian data yang berbeda.

#### 6. Analisis Hasil dan Interpretasi
Setelah mengumpulkan metrik evaluasi dan visualisasi, analisis hasil yang didapat untuk:
- Memastikan error yang dihasilkan masuk akal secara domain (misalnya, apakah error masuk akal jika dibandingkan dengan variasi alami hasil panen).
- Menentukan apakah terdapat bias tertentu atau pola error yang menunjukkan kebutuhan untuk penyempurnaan fitur atau model.
- Menyajikan rekomendasi perbaikan di samping temuan yang mendukung efektivitas model sebagai alat bantu pengambilan keputusan.

---

## 7. Kesimpulan
Proyek **Prediksi Hasil Panen Optimal** telah berhasil mengembangkan dan mengevaluasi model machine learning menggunakan dua pendekatan utama: **Linear Regression** dan **Random Forest Regression**, dengan optimasi melalui **GridSearchCV**.

**Hasil Evaluasi Model:**
- **Linear Regression:**  
  - **MAE:** 420.32  
  - **MSE:** 296193.51  
  - **R² Score:** 0.93  

- **Random Forest Regression (Baseline):**  
  - **MAE:** 430.11  
  - **MSE:** 324300.15  
  - **R² Score:** 0.92  

- **Random Forest Regression (Optimized with GridSearchCV):**  
  - **Best Parameters:** `{'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 200}`  
  - **MAE:** 426.08  
  - **MSE:** 311815.73  
  - **R² Score:** 0.92  

Dari hasil tersebut, **Linear Regression** menunjukkan performa sedikit lebih baik dibandingkan model **Random Forest**, terutama dari segi **MSE yang lebih rendah dan R² score yang lebih tinggi**. Model regresi linier cukup mampu menangkap pola prediksi dengan baik, tetapi model ensemble seperti Random Forest lebih fleksibel dalam menangani hubungan non-linear dan interaksi fitur.

**Insight dan Rekomendasi:**
- Model **Linear Regression** memberikan hasil yang baik, namun mungkin terbatas dalam menangkap pola hubungan kompleks antar variabel agronomi.  
- **Random Forest Regression** memberikan hasil yang stabil dan dapat ditingkatkan dengan lebih banyak data atau fitur tambahan seperti data historis cuaca atau sistem irigasi.  
- **Optimasi Hyperparameter pada Random Forest** membantu meningkatkan akurasi, tetapi belum mampu melampaui hasil dari model linier.  
- Perlu dilakukan **eksplorasi lebih lanjut terhadap fitur yang paling berpengaruh**, seperti kandungan nitrogen dan kelembapan tanah, untuk meningkatkan model prediksi.  

Dari evaluasi model dan analisis residual, dapat disimpulkan bahwa meskipun model mampu menangkap pola prediksi dengan cukup baik, masih terdapat beberapa **area untuk penyempurnaan**, seperti:
- **Penambahan fitur lain yang lebih spesifik**, seperti pola cuaca historis atau kondisi irigasi untuk meningkatkan akurasi prediksi.
- **Peningkatan jumlah data training**, agar model lebih generalizable terhadap variasi lingkungan pertanian di berbagai wilayah.
- **Integrasi model dengan sistem digital**, seperti dashboard prediksi berbasis web, guna mempermudah akses informasi bagi petani dan pemangku kepentingan.

Dengan pengembangan lebih lanjut, model ini berpotensi menjadi **alat strategis dalam precision agriculture**, membantu petani dalam pengambilan keputusan berbasis data untuk meningkatkan produktivitas dan efisiensi penggunaan sumber daya pertanian.

---

## 8. Pengembangan Lebih Lanjut
Setelah proyek **Crop Yield Prediction** ini berhasil dikembangkan dan dievaluasi, ada beberapa area yang dapat diperluas untuk meningkatkan akurasi, skalabilitas, dan penerapan nyata dalam sektor pertanian. Berikut beberapa arah pengembangan lebih lanjut yang dapat dilakukan:

#### **1. Peningkatan Model dan Teknik Machine Learning**
- **Eksperimen dengan Model Lebih Kompleks:**  
  Mengembangkan model berbasis *Deep Learning* seperti **Artificial Neural Networks (ANN)** atau **Long Short-Term Memory (LSTM)** untuk menangkap pola non-linear dan hubungan kompleks antar variabel.
- **Transfer Learning untuk Pertanian:**  
  Menggunakan model pre-trained dari dataset agronomi lainnya untuk meningkatkan akurasi dalam memprediksi hasil panen berdasarkan kondisi lingkungan dan input pertanian.
- **Automated Feature Engineering:**  
  Menggunakan metode otomatis seperti **Feature Selection dengan Recursive Feature Elimination (RFE)** atau **AutoML** untuk menemukan kombinasi fitur terbaik tanpa eksplorasi manual.

#### **2. Integrasi dengan Data Real-Time dan IoT**
- **Penggunaan Sensor dan IoT untuk Monitoring Lahan:**  
  Menghubungkan model prediksi dengan **sensor tanah dan cuaca** yang mengumpulkan data real-time mengenai kelembapan, temperatur, dan kadar nutrisi. Data ini dapat dimasukkan ke dalam model untuk prediksi yang lebih akurat.
- **Integrasi dengan API Cuaca dan Data Satelit:**  
  Menggunakan API seperti **Google Weather API** atau data satelit dari **NASA** dan **Sentinel-2** untuk mengakses informasi iklim dan vegetasi, yang akan membantu prediksi hasil panen secara lebih dinamis.

#### **3. Pengembangan Dashboard Interaktif dan Aplikasi Web**
- **Membuat Dashboard Visualisasi dengan Streamlit atau Flask:**  
  Mengembangkan antarmuka visual yang memungkinkan pengguna (misalnya petani atau pemangku kebijakan) menginput variabel pertanian dan mendapatkan prediksi hasil panen secara langsung.
- **Implementasi Model sebagai API:**  
  Membuat REST API menggunakan **FastAPI** atau **Flask**, sehingga prediksi bisa digunakan dalam aplikasi web atau sistem manajemen pertanian.

#### **4. Optimasi Penggunaan Pupuk dan Sumber Daya**
- **Rekomendasi Dosis Pupuk Optimal:**  
  Berdasarkan hasil prediksi, model dapat diperluas menjadi sistem rekomendasi untuk **pemilihan jenis pupuk terbaik** dan **dosis yang optimal** berdasarkan kondisi tanah dan tanaman.
- **Integrasi dengan Sistem Manajemen Pertanian (Agriculture ERP):**  
  Model dapat diintegrasikan dengan **Enterprise Resource Planning (ERP) khusus pertanian**, sehingga rekomendasi pertanian berbasis data dapat digunakan untuk meningkatkan efisiensi operasional.

#### **5. Penerapan dalam Precision Agriculture dan Keberlanjutan**
- **Skalabilitas untuk Berbagai Wilayah:**  
  Mengadaptasi model agar dapat digunakan pada berbagai wilayah pertanian dengan karakteristik tanah dan iklim yang berbeda.
- **Pemetaan Risiko dan Adaptasi Perubahan Iklim:**  
  Menggunakan prediksi hasil panen untuk menentukan risiko gagal panen akibat cuaca ekstrem, serta menyusun strategi adaptasi agar produktivitas pertanian tetap stabil.

---

Dengan pengembangan lebih lanjut ini, proyek prediksi hasil panen tidak hanya akan membantu petani dalam meningkatkan efisiensi dan produktivitas, tetapi juga berkontribusi dalam **ketahanan pangan, keberlanjutan ekosistem pertanian, dan adaptasi terhadap perubahan iklim**. 

---
