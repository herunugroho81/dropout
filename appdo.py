import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Prediksi Risiko Dropout", layout="wide")
st.title("🎓 Aplikasi Prediksi Risiko Mahasiswa Mengundurkan Diri")
st.markdown("Upload dataset atau masukkan data secara manual untuk memprediksi apakah mahasiswa berisiko undur diri.")

# Sidebar untuk input manual
with st.sidebar:
    st.header("📊 Masukkan Data Mahasiswa")
    student_id = st.text_input("Student ID")
    attendance_rate = st.slider("Persentase Kehadiran (%)", 0, 100, 75)
    gpa = st.slider("IPK Terakhir", 0.0, 4.0, 3.0, step=0.1)
    participation_score = st.slider("Skor Partisipasi (0–10)", 0, 10, 5)
    study_hours_per_week = st.slider("Jam Belajar Per Minggu", 0, 40, 15)

    submit = st.button("🔍 Prediksi")

# Fungsi untuk melatih model
def train_model(df):
    df['dropout_risk'] = df['dropout_risk'].map({'No': 0, 'Yes': 1})
    X = df.drop(['dropout_risk', 'student_id'], axis=1)
    y = df['dropout_risk']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    return knn, scaler, X_test, y_test, y_pred

# Upload dataset
uploaded_file = st.file_uploader("📁 Unggah file CSV Dataset", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Dataset berhasil dimuat!")

        # Tampilkan pratinjau
        st.subheader("🧾 Pratinjau Dataset")
        st.dataframe(df.head())

        if 'dropout_risk' in df.columns and 'student_id' in df.columns:
            with st.spinner("🔄 Melatih model..."):
                knn, scaler, y_test, y_pred, X_test = train_model(df)

                # Evaluasi
                st.subheader("📊 Hasil Evaluasi Model")
                acc = accuracy_score(y_test, y_pred)
                st.metric("Akurasi", f"{acc:.2%}")

                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.xlabel('Prediksi')
                plt.ylabel('Aktual')
                st.pyplot(fig)

                report = classification_report(y_test, y_pred, output_dict=False)
                st.text("📋 Laporan Klasifikasi:\n" + report)

                # Prediksi manual
                if submit:
                    new_student = [[attendance_rate, gpa, participation_score, study_hours_per_week]]
                    new_student_scaled = scaler.transform(new_student)
                    prediction = knn.predict(new_student_scaled)
                    risk = "Ya" if prediction[0] == 1 else "Tidak"
                    st.success(f"⚠️ Prediksi: Mahasiswa {risk} berisiko mengundurkan diri.")
        else:
            st.warning("⚠️ Kolom 'dropout_risk' atau 'student_id' tidak ditemukan di dataset.")
    except Exception as e:
        st.error(f"❌ Kesalahan saat memproses dataset: {e}")
else:
    st.info("📂 Silakan unggah file CSV untuk memulai atau gunakan form di sidebar untuk prediksi manual.")

    # Prediksi manual tanpa dataset
    if submit:
        new_student = [[attendance_rate, gpa, participation_score, study_hours_per_week]]
        # Dummy scaling jika tidak ada model
        dummy_scaler = StandardScaler()
        dummy_scaler.mean_ = [70, 3.0, 5, 15]
        dummy_scaler.scale_ = [15, 0.5, 2, 5]
        new_student_scaled = dummy_scaler.transform(new_student)
        # Dummy prediksi
        prediction = [1] if new_student[0][0] < 60 or new_student[0][1] < 2.0 or new_student[0][2] < 3 else [0]
        risk = "Ya" if prediction[0] == 1 else "Tidak"
        st.success(f"⚠️ Prediksi: Mahasiswa {risk} berisiko mengundurkan diri.")