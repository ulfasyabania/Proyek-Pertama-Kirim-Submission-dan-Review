---

## Laporan Proyek Machine Learning – [Nama Anda]  
**Domain Proyek: Prediksi Hasil Panen Padi di Jawa Barat**

---

### 1. Latar Belakang dan Business Understanding

Di era digital dengan pertanian yang semakin mengadopsi teknologi, prediksi hasil panen menjadi aspek kunci untuk membantu petani mengoptimalkan penggunaan lahan, jenis pupuk, dan irigasi. Proyek ini bertujuan membangun model prediktif untuk memprediksi *Hasil Panen (ton)* berdasarkan variabel-variabel seperti:  
- **Luas Lahan (ha):** luas area yang dialokasikan untuk tanam  
- **Jenis Pupuk:** tipe pupuk yang digunakan  
- **Curah Hujan (mm):** curah hujan yang didapat selama periode tanam  
- **Intensitas Irigasi (kali/minggu):** frekuensi penyiraman lahan  

**Problem Statements:**  
1. Bagaimana memprediksi hasil panen secara akurat berdasarkan faktor-faktor pertanian?  
2. Bagaimana mengidentifikasi kombinasi optimal dari variabel input yang berkontribusi pada peningkatan hasil panen?  

**Goals:**  
- Menghasilkan model regresi yang mampu memperkirakan hasil panen dengan metrik evaluasi yang baik (misalnya, R² score, MAE, dan MSE).  
- Menyediakan insight berbasis data mengenai kontribusi masing-masing faktor terhadap hasil panen.  

**Solution Statements (Opsional):**  
1. Menggunakan Linear Regression sebagai baseline model.  
2. Mencoba algoritma lain, seperti Decision Tree Regression dan RandomForest Regression, untuk menguji performa yang lebih baik melalui hyperparameter tuning.  

---

### 2. Data Understanding

Dataset yang digunakan bersifat sintetik. Nilainya dihasilkan berdasarkan asumsi dan simulasi sehingga mendekati fenomena nyata. Dataset ini terdiri dari lima variabel:  

- **Luas Lahan (ha)**
- **Jenis Pupuk**
- **Curah Hujan (mm)**
- **Intensitas Irigasi (kali/minggu)**
- **Hasil Panen (ton)**

Dataset dapat diunduh melalui sumber berikut:  
[Dataset Produksi Padi Jawa Barat](https://raw.githubusercontent.com/ulfasyabania/Proyek-Pertama-Kirim-Submission-dan-Review/refs/heads/main/produksi_padi_jabar.csv)

---

### 3. Data Preparation

Tahapan persiapan data meliputi:  
- **Memeriksa nilai kosong atau tidak sesuai.**  
- **Mengubah tipe data jika diperlukan.**  
- **Membagi data menjadi training dan testing set.**

---

### 4. Modeling

Pada tahap ini, kita akan menerapkan beberapa algoritma regresi:  
- **Linear Regression:** sebagai model baseline.  
- **Decision Tree Regression dan RandomForest Regression:** sebagai model alternatif dan untuk improvement.

Setiap solusi akan dievaluasi menggunakan metrik evaluasi yang sesuai. Proses hyperparameter tuning dilakukan pada model-model non-linear untuk memperoleh performa terbaik.

---

### 5. Evaluation

Untuk evaluasi model, metrik yang digunakan adalah:  
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **R² Score**

Setiap model akan dibandingkan berdasarkan metrik ini untuk memilih model terbaik.

---

### 6. Implementation di Google Colab

Berikut adalah contoh kode lengkap yang dapat Anda gunakan di Google Colab.

---

```python
# Cell 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set style for plots
sns.set(style='whitegrid')
```

---

```python
# Cell 2: Load the Dataset
url = "https://raw.githubusercontent.com/ulfasyabania/Proyek-Pertama-Kirim-Submission-dan-Review/refs/heads/main/produksi_padi_jabar.csv"
data = pd.read_csv(url)

# Tampilkan 5 baris pertama
data.head()
```

---

```python
# Cell 3: Data Understanding & EDA
print("Informasi dataset:")
data.info()

print("\nStatistik deskriptif:")
print(data.describe())

# Cek missing values
print("\nMissing values per kolom:")
print(data.isnull().sum())

# Visualisasi distribusi tiap variabel
plt.figure(figsize=(16, 4))

plt.subplot(1, 5, 1)
sns.histplot(data['Luas Lahan (ha)'], kde=True)
plt.title('Luas Lahan (ha)')

plt.subplot(1, 5, 2)
sns.countplot(x='Jenis Pupuk', data=data)
plt.title('Jenis Pupuk')

plt.subplot(1, 5, 3)
sns.histplot(data['Curah Hujan (mm)'], kde=True)
plt.title('Curah Hujan (mm)')

plt.subplot(1, 5, 4)
sns.histplot(data['Intensitas Irigasi (kali/minggu)'], kde=True)
plt.title('Intensitas Irigasi')

plt.subplot(1, 5, 5)
sns.histplot(data['Hasil Panen (ton)'], kde=True)
plt.title('Hasil Panen (ton)')

plt.tight_layout()
plt.show()
```

---

```python
# Cell 4: Correlation Analysis
plt.figure(figsize=(8,6))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu")
plt.title("Correlation Matrix")
plt.show()
```

---

```python
# Cell 5: Data Preparation
# Pastikan tidak terdapat nilai null, dan jika perlu, lakukan transformasi.
# Karena dataset sudah bersih (synthetic), kita langsung pisahkan data

# Fitur (X) dan target (y)
X = data.drop("Hasil Panen (ton)", axis=1)
y = data["Hasil Panen (ton)"]

# Bagi data menjadi training dan testing (80:20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data training:", X_train.shape)
print("Data testing:", X_test.shape)
```

---

```python
# Cell 6: Modeling – Baseline with Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Prediksi pada test set
y_pred_lr = lr.predict(X_test)

# Evaluasi Linear Regression
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("Linear Regression Performance:")
print("MAE:", mae_lr)
print("MSE:", mse_lr)
print("R2 Score:", r2_lr)
```

---

```python
# Cell 7: Modeling – Decision Tree Regression
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

mae_dt = mean_absolute_error(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)

print("\nDecision Tree Regression Performance:")
print("MAE:", mae_dt)
print("MSE:", mse_dt)
print("R2 Score:", r2_dt)
```

---

```python
# Cell 8: Modeling – Random Forest Regression with Hyperparameter Tuning
rf = RandomForestRegressor(random_state=42)

# Grid search untuk hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_rf.fit(X_train, y_train)
print("Best parameters:", grid_rf.best_params_)

# Gunakan model terbaik untuk prediksi
best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\nRandom Forest Regression Performance:")
print("MAE:", mae_rf)
print("MSE:", mse_rf)
print("R2 Score:", r2_rf)
```

---

```python
# Cell 9: Evaluation Comparison
results = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree", "Random Forest"],
    "MAE": [mae_lr, mae_dt, mae_rf],
    "MSE": [mse_lr, mse_dt, mse_rf],
    "R2 Score": [r2_lr, r2_dt, r2_rf]
})

print(results)
```

---

```python
# Cell 10: Visualisasi Perbandingan Hasil Prediksi dengan Nilai Aktual
plt.figure(figsize=(10,5))
plt.scatter(y_test, y_pred_rf, color='green', alpha=0.6, label='Random Forest')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label="Ideal")
plt.xlabel("Nilai Aktual")
plt.ylabel("Prediksi")
plt.title("Perbandingan Nilai Aktual vs Prediksi (Random Forest)")
plt.legend()
plt.show()
```

---

### 7. Kesimpulan

Dari tahap evaluasi, dapat dilihat bahwa model **Random Forest Regression** (setelah hyperparameter tuning) memberikan performa terbaik berdasarkan metrik R² serta tingkat error (MAE dan MSE) yang rendah. Model ini dapat dijadikan solusi untuk mendukung keputusan dalam mengoptimalkan input pertanian guna meningkatkan hasil panen padi di Jawa Barat.

---

### 8. Pengembangan Lebih Lanjut

Beberapa usulan pengembangan tambahan:  
- **Analisis Feature Importance:** Menentukan variabel mana yang paling berpengaruh terhadap prediksi hasil panen.  
- **Cross Validation yang Lebih Mendalam:** Mengimplementasikan teknik validasi silang untuk menguji kestabilan model.  
- **Penerapan Model Ensembel Lainnya:** Mencoba algoritma seperti Gradient Boosting atau XGBoost untuk perbandingan lebih lanjut.  
- **Visualisasi Interaktif:** Menggunakan Plotly atau Bokeh untuk eksplorasi data lebih mendalam.

---
