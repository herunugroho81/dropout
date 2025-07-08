import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Baca dataset
df = pd.read_csv("student_dropout_dataset.csv")

# Encode label target
df['dropout_risk'] = df['dropout_risk'].map({'No': 0, 'Yes': 1})

# Pisahkan fitur dan target, hilangkan kolom 'student_id'
X = df.drop(['dropout_risk', 'student_id'], axis=1)
y = df['dropout_risk']

# Normalisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Buat model KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Prediksi
y_pred = knn.predict(X_test)

# Evaluasi
print("Akurasi:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred))

# Contoh prediksi baru
new_student = [[85, 3.2, 7, 15]]  # [attendance, gpa, participation, study hours]
new_student_scaled = scaler.transform(new_student)
prediction = knn.predict(new_student_scaled)
risk = "Ya" if prediction[0] == 1 else "Tidak"
print(f"\nPrediksi untuk siswa baru: Apakah akan undur diri? → {risk}")
