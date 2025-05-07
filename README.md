---

# Laporan Proyek Machine Learning - Tb Ulfah Nur Sya'baniah

---

## 1. Domain Proyek

**Latar Belakang:**  
Proyek ini mengambil domain **Predictive Maintenance (Pemeliharaan Prediktif)** pada mesin produksi. Di industri manufaktur dan produksi, kerusakan mesin dapat mengakibatkan downtime yang signifikan, menurunkan produktivitas, dan meningkatkan biaya operasional. Dengan menganalisis parameter operasional seperti suhu udara, suhu proses, kecepatan rotasi, torsi, dan keausan alat, kita dapat memprediksi potensi kegagalan mesin lebih dini dan mengoptimalkan jadwal perawatan.

**Alasan dan Riset Terkait:**  
- Dengan menerapkan teknik machine learning, perusahaan dapat mengurangi biaya pemeliharaan mendadak (unscheduled maintenance) dan meningkatkan efisiensi proses produksi.
- Studi-studi telah menunjukkan bahwa penggunaan algoritma prediktif dalam predictive maintenance dapat meningkatkan akurasi prediksi dan mengurangi downtime.  
(Sumber data: UCI Machine Learning Repository: (https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset))

---

## 2. Business Understanding

**Problem Statements:**  
1. **Pernyataan Masalah 1:** Bagaimana mengidentifikasi potensi kegagalan mesin secara tepat waktu sehingga dapat dilakukan perawatan sebelum terjadi kerusakan total?  
2. **Pernyataan Masalah 2:** Bagaimana memanfaatkan data operasional mesin untuk meningkatkan efektivitas perawatan preventif dan mengurangi biaya operasional?

**Goals:**  
1. **Goal 1:** Memprediksi kegagalan mesin dengan akurasi yang tinggi berdasarkan parameter sensor dan data historis.  
2. **Goal 2:** Mengoptimalkan jadwal perawatan (maintenance scheduling) sehingga dapat mencegah downtime yang tidak diinginkan dan meminimalkan biaya perawatan.

**Solution Statement (Opsional):**  
Untuk mencapai goals tersebut, kami mengusulkan:
- **Solusi 1:** Membangun baseline model klasifikasi menggunakan algoritma seperti Random Forest dan Logistic Regression untuk memprediksi kegagalan mesin.
- **Solusi 2:** Melakukan hyperparameter tuning dan teknik ensemble untuk meningkatkan performa model, serta mengukur model dengan metrik evaluasi seperti akurasi, precision, recall, dan F1 score.

---

## 3. Data Understanding

**Deskripsi Dataset:**  
Dataset yang digunakan adalah *AI4I 2020 Predictive Maintenance Dataset* yang tersedia pada [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset).  
Beberapa variabel utama dalam data ini antara lain:
- **UDI:** Unique Identifier untuk setiap sampel.
- **Product ID and Type:** Identifikasi dan tipe mesin.
- **Air temperature [K]:** Suhu udara di area tempat mesin bekerja.
- **Process temperature [K]:** Suhu pada saat proses produksi berlangsung.
- **Rotational speed [rpm]:** Kecepatan rotasi mesin.
- **Torque [Nm]:** Torsi yang dihasilkan.
- **Tool wear [min]:** Lama keausan alat yang digunakan.
- **Machine failure:** Label target, menunjukkan apakah mesin mengalami kegagalan.
- **TWF, HDF, PWF, OSF, RNF:** Indikator kegagalan spesifik.

**Exploratory Data Analysis (EDA):**  
Beberapa tahapan EDA yang telah dilakukan meliputi:
- **Distribusi Data:**  
  - *Histogram:* Menggunakan `sns.histplot()` untuk melihat sebaran tiap fitur numerik dan mendeteksi pola seperti skewness atau multimodalitas.  
  - *Boxplot:* Menggunakan `sns.boxplot()` untuk mendeteksi outlier dan memahami ringkasan statistik dari masing-masing fitur.
- **Analisis Korelasi:**  
  Matriks korelasi dihitung dengan `df.corr()` dan divisualisasikan dengan `sns.heatmap()`, sehingga dapat diidentifikasi fitur mana yang memiliki keterkaitan tinggi dan berpotensi redundant dalam model.

_(Misalnya, kode berikut digunakan untuk EDA visual:)_  

```python
# Visualisasi Distribusi menggunakan Histogram dan Boxplot
import matplotlib.pyplot as plt
import seaborn as sns

# Daftar fitur numerik untuk dianalisis
numeric_cols = ["Air temperature [K]", "Process temperature [K]", 
                "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]

# Histogram
plt.figure(figsize=(15, 10))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[col], kde=True, bins=30)
    plt.title(f"Distribusi {col}")
    plt.xlabel(col)
    plt.ylabel("Frekuensi")
plt.tight_layout()
plt.show()

# Boxplot
plt.figure(figsize=(15, 5))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(1, len(numeric_cols), i)
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot {col}")
    plt.xlabel(col)
plt.tight_layout()
plt.show()

# Matriks Korelasi dan Heatmap
corr_matrix = df[numeric_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Matriks Korelasi Fitur Numerik")
plt.show()
```

---

## 4. Data Preparation

**Proses yang Dilakukan:**  
1. **Pembersihan Data:**  
   - Pemeriksaan dan penanganan missing values menggunakan `SimpleImputer` jika diperlukan.  
   - Memastikan bahwa tipe data sesuai dengan yang dibutuhkan untuk analisis.
2. **Normalisasi Data:**  
   - Menggunakan `StandardScaler` untuk menormalisasi fitur numerik agar semua fitur berada pada skala yang sama. Hal ini penting agar model machine learning tidak bias terhadap fitur dengan rentang nilai yang besar.
3. **Split Dataset:**  
   - Data dibagi menjadi training set dan testing set untuk validasi model.

**Alasan Data Preparation:**  
Tahapan ini penting untuk memastikan bahwa data siap digunakan untuk proses pemodelan. Normalisasi dan penanganan missing values meningkatkan stabilitas dan akurasi model, sedangkan pembagian dataset memungkinkan evaluasi performa secara adil dan mencegah overfitting.

_(Contoh kode Data Preparation:)_

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Misal, kita fokus pada fitur numerik dan label 'Machine failure'
features = df[numeric_cols]
label = df["Machine failure"]

# Mengatasi missing values (jika ada) dengan imputasi menggunakan median
imputer = SimpleImputer(strategy="median")
features_imputed = imputer.fit_transform(features)

# Normalisasi fitur
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_imputed)

# Membagi data menjadi training dan testing set (misalnya 70:30)
X_train, X_test, y_train, y_test = train_test_split(features_scaled, label, test_size=0.3, random_state=42)
```

---

## 5. Modeling

**Algoritma yang Digunakan:**  
Dalam proyek ini, kita menerapkan beberapa algoritma untuk memprediksi kegagalan mesin, contohnya:
- **Logistic Regression**  
- **Random Forest Classifier**

**Tahapan Pemodelan:**  
1. **Baseline Model:**  
   - Menggunakan Logistic Regression sebagai model baseline.
2. **Improvement Model:**  
   - Menerapkan Random Forest yang biasanya bisa menangani data nonlinear dan interaksi antar fitur dengan baik.
3. **Hyperparameter Tuning:**  
   - Melakukan tuning hyperparameter menggunakan teknik seperti Grid Search untuk meningkatkan performa model.

**Contoh Kode Modeling:**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Model Baseline: Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("Classification Report - Logistic Regression:")
print(classification_report(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

# Model Improvement: Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Classification Report - Random Forest:")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
```

**Insight Modeling:**  
- **Kelebihan dan Kekurangan:**  
  *Logistic Regression* bersifat sederhana dan cepat, tetapi mungkin tidak menangkap interaksi nonlinier antar fitur. *Random Forest* biasanya memberikan performa yang lebih baik karena mampu menangani data kompleks, namun membutuhkan waktu komputasi yang lebih besar.  
- **Pemilihan Model Terbaik:**  
  Model terbaik dipilih berdasarkan metrik evaluasi (accuracy, precision, recall, F1 score). Pada proyek ini, jika Random Forest menunjukkan peningkatan performa signifikan dibandingkan Logistic Regression, maka Random Forest dijadikan model akhir.

---

## 6. Evaluation

**Metrik Evaluasi yang Digunakan:**  
Untuk mengukur performa model, digunakan beberapa metrik evaluasi, antara lain:  
- **Accuracy:** Persentase prediksi yang benar secara keseluruhan.  
- **Precision:** Kemampuan model dalam memprediksi dengan tepat label positif.  
- **Recall:** Kemampuan model untuk menangkap semua kasus positif.  
- **F1 Score:** Rata-rata harmonis antara precision dan recall untuk menyeimbangkan trade-off-nya.

**Hasil Evaluasi:**  
- Laporan klasifikasi (classification report) dan confusion matrix memberikan gambaran menyeluruh mengenai performa model pada data testing.  
- Berdasarkan hasil evaluasi ini, kita dapat menentukan apakah model sudah cukup baik untuk diterapkan atau masih memerlukan improvement lebih lanjut seperti penyesuaian hyperparameter, rekayasa fitur, atau eksplorasi algoritma lain.

_(Contoh evaluasi hasil model sudah ditampilkan pada kode modeling di atas.)_

---

## Kesimpulan dan Rekomendasi

**Kesimpulan:**  
Menggunakan pendekatan Predictive Maintenance berbasis Machine Learning memungkinkan perusahaan untuk mendeteksi potensi kegagalan mesin lebih awal. Dengan memanfaatkan data operasional dan menerapkan teknik EDA, data preparation, dan pemodelan yang baik, kita dapat mencapai akurasi prediksi yang tinggi.

**Rekomendasi:**  
- Lanjutkan dengan eksplorasi algoritma lainnya (misalnya, Gradient Boosting atau Neural Networks) jika diperlukan.
- Lakukan perbaikan pada proses rekayasa fitur untuk menangkap pola-pola yang lebih kompleks.
- Terapkan sistem monitoring real-time di lingkungan operasional untuk pemantauan dan perawatan mesin secara proaktif.

---
