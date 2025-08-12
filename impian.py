import streamlit as st
import pandas as pd
import sqlite3
import os
import io
import pydotplus
from PIL import Image
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime

def log_admin_action(action, id=None, nama=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("admin_log.txt", "a") as f:
        f.write(f"[{timestamp}] {action} - ID: {id}, Nama: {nama}\n")


# ==================== LOGGING HASIL PREDIKSI ====================
def simpan_log(nama, prediksi, nilai_akhir):
    with open("log_prediksi.txt", "a") as f:
        f.write(f"{nama},{prediksi},{nilai_akhir}\n")

# ==================== FUNGSI CITA-CITA LENGKAP ====================
def tentukan_cita_cita(minat, bakat):
    mapping = {
        # --- Sains ---
        ("Sains", "Logika"): "Dokter, Peneliti, Ahli Biologi",
        ("Sains", "Analitik"): "Dokter, Ilmuwan Data, Farmasis",
        ("Sains", "Kreatif"): "Ahli Teknologi, Desainer Produk",
        ("Sains", "Kinestetik"): "Dokter Bedah, Ahli Fisioterapi, Atlet Sains",
        ("Sains", "Bahasa"): "Dosen Sains, Penulis Ilmiah, Peneliti Luar Negeri",

        # --- Teknik ---
        ("Teknik", "Logika"): "Teknisi, IT Support, Engineer",
        ("Teknik", "Analitik"): "Insinyur, Arsitek, Programmer",
        ("Teknik", "Kreatif"): "Desainer Otomotif, Inovator Produk",
        ("Teknik", "Kinestetik"): "Teknisi Lapangan, Mekanik, Pilot",
        ("Teknik", "Bahasa"): "Instruktur Teknik, Konsultan Proyek Luar Negeri",

        # --- Desain (pengganti Bahasa) ---
        ("Desain", "Bahasa"): "Guru Desain, Editor Kreatif, Penerjemah Visual",
        ("Desain", "Kreatif"): "Desainer Grafis, Ilustrator, Penulis Naskah",
        ("Desain", "Kinestetik"): "Pemandu Seni, Kurator, MC",
        ("Desain", "Logika"): "Ahli UI/UX, Desainer Produk Digital",
        ("Desain", "Analitik"): "Peneliti Tren Desain, Visual Strategist",

        # --- Sosial ---
        ("Sosial", "Bahasa"): "Politikus, Diplomat, Pengacara",
        ("Sosial", "Kreatif"): "Public Relation, Event Organizer, Motivator",
        ("Sosial", "Kinestetik"): "Pelatih, Psikolog, Konselor",
        ("Sosial", "Logika"): "Ahli Hukum, Manajer SDM, Perencana Kebijakan",
        ("Sosial", "Analitik"): "Peneliti Sosial, Ekonom, Surveyor Opini Publik",

        # --- Seni ---
        ("Seni", "Kreatif"): "Seniman, Desainer, Animator",
        ("Seni", "Kinestetik"): "Aktor, Penari, Koreografer",
        ("Seni", "Logika"): "Desainer Game, Arsitek Interior",
        ("Seni", "Bahasa"): "Penulis Skenario Film, Kritikus Seni, Kurator",
        ("Seni", "Analitik"): "Fotografer, Kritikus Film, Peneliti Seni",
    }
    return mapping.get((minat, bakat), "Belum tersedia, konsultasi lebih lanjut")

    # Tentukan prediksi akhir tanpa konsultasi untuk Kinestetik
    if (prediksi == "IPA" and minat_siswa in ["Sains", "Teknik"] and bakat_siswa in ["Logika", "Analitik", "Kinestetik"]) or \
    (prediksi == "IPS" and minat_siswa in ["Sosial", "Bahasa"] and bakat_siswa in ["Bahasa", "Kreatif", "Kinestetik"]):
        prediksi_final = prediksi
    else:
        prediksi_final = prediksi + " (perlu konsultasi)"


# ==================== KONFIGURASI STREAMLIT ====================
st.set_page_config(page_title="SPK Penjurusan", layout="wide")

# ==================== KONFIGURASI LOGIN ====================
USER_CREDENTIALS = {
    "admin": "admin123",
    "user": "user123"
}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# ==================== FUNGSI DATABASE ====================
def init_db():
    conn = sqlite3.connect("siswa.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS nilai_siswa (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nama TEXT,
            matematika INTEGER,
            binggris INTEGER,
            biologi INTEGER,
            fisika INTEGER,
            kimia INTEGER,
            minat TEXT,
            bakat TEXT
        )
    """)
    conn.commit()
    conn.close()

def insert_data(nama, mtk, bing, bio, fis, kim, minat, bakat):
    conn = sqlite3.connect("siswa.db")
    c = conn.cursor()
    c.execute("INSERT INTO nilai_siswa (nama, matematika, binggris, biologi, fisika, kimia, minat, bakat) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
              (nama, mtk, bing, bio, fis, kim, minat, bakat))
    conn.commit()
    conn.close()

def update_data(id, nama, mtk, bing, bio, fis, kim, minat, bakat):
    conn = sqlite3.connect("data_siswa.db")
    c = conn.cursor()
    c.execute("""
        UPDATE siswa SET
        nama = ?, matematika = ?, binggris = ?, biologi = ?,
        fisika = ?, kimia = ?, minat = ?, bakat = ?
        WHERE id = ?
    """, (nama, mtk, bing, bio, fis, kim, minat, bakat, id))
    conn.commit()
    conn.close()


def get_all_data():
    conn = sqlite3.connect("siswa.db")
    df = pd.read_sql_query("SELECT * FROM nilai_siswa", conn)
    conn.close()
    return df

def delete_data(id):
    conn = sqlite3.connect("siswa.db")
    c = conn.cursor()
    c.execute("DELETE FROM nilai_siswa WHERE id = ?", (id,))
    conn.commit()
    conn.close()

def delete_all_data():
    conn = sqlite3.connect("siswa.db")
    c = conn.cursor()
    c.execute("DELETE FROM nilai_siswa")
    conn.commit()
    conn.close()

# --- Fungsi Normalisasi dan Validasi Silang ---
def evaluate_knn_model():
    df = get_all_data()
    if len(df) < 5:
        st.warning("Data belum cukup untuk validasi silang (minimal 5 siswa).")
        return
    X = df[["matematika", "binggris", "biologi", "fisika", "kimia"]]
    y = df["jurusan"]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    knn = KNeighborsClassifier(n_neighbors=3)
    scores = cross_val_score(knn, X_scaled, y, cv=5)
    st.subheader("📊 Validasi Silang KNN")
    st.write("Akurasi Tiap Lipatan:", scores)
    st.write("Akurasi Rata-rata:", f"{scores.mean() * 100:.2f}%")


def evaluate_dt_model():
    df = get_all_data()
    if len(df) < 5:
        st.warning("Data belum cukup untuk validasi silang.")
        return
    X = df[["matematika", "binggris", "biologi", "fisika", "kimia"]]
    y = df["jurusan"]
    tree = DecisionTreeClassifier()
    scores = cross_val_score(tree, X, y, cv=5)
    st.subheader("🌳 Validasi Silang Decision Tree")
    st.write("Akurasi Tiap Lipatan:", scores)
    st.write("Akurasi Rata-rata:", f"{scores.mean() * 100:.2f}%")

# ==================== FUNGSI HITUNG BOBOT OTOMATIS ====================
def hitung_bobot_otomatis(df):
    if df.empty:
        return {
            "Matematika": 0.2,
            "Bahasa Inggris": 0.2,
            "Biologi": 0.2,
            "Fisika": 0.2,
            "Kimia": 0.2
        }
    fitur = ["matematika", "binggris", "biologi", "fisika", "kimia"]
    variansi = df[fitur].var()
    bobot_normalisasi = variansi / variansi.sum()
    return {
        "Matematika": round(bobot_normalisasi["matematika"], 3),
        "Bahasa Inggris": round(bobot_normalisasi["binggris"], 3),
        "Biologi": round(bobot_normalisasi["biologi"], 3),
        "Fisika": round(bobot_normalisasi["fisika"], 3),
        "Kimia": round(bobot_normalisasi["kimia"], 3)
    }

# ==================== FUNGSI LOGIN ====================
def login():
    st.title("🔐 Login Sistem SPK Penjurusan")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Selamat datang, {username}!")
            st.rerun()
        else:
            st.error("Username atau password salah.")

def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

# ==================== LOGIKA APP ====================
init_db()

if not st.session_state.logged_in:
    login()
else:
    if 'bobot' not in st.session_state:
        st.session_state.bobot = {
            "Matematika": 0.2,
            "Bahasa Inggris": 0.2,
            "Biologi": 0.2,
            "Fisika": 0.2,
            "Kimia": 0.2
        }

    # ✅ Normalisasi bobot jika total != 1
    total_bobot = sum(st.session_state.bobot.values())
    if total_bobot != 1.0:
        for k in st.session_state.bobot:
            st.session_state.bobot[k] /= total_bobot
        st.info("🔄 Bobot dinormalisasi otomatis agar total = 1.0")

    # ==================== NAVIGASI MENU ====================
    menu = st.sidebar.radio("Navigasi", [
        "🏠 Home",
        "🧠 Tips Memilih Jurusan",
        "📝 Input Nilai",
        "📊 Hasil Prediksi",
        "📋 Tabel Perhitungan",
        "✏️ Edit / Update Nilai Siswa",
        "📈 Statistik Nilai",
        "📊 Korelasi Nilai",  # NEW
        "🏆 Ranking Siswa",   # NEW
        "📊 Distribusi Jurusan",
        "⚖️ Atur Bobot",
        "🧪 Uji Akurasi KNN",
        "🧪 Uji Akurasi Decision Tree",
        "🔍 Perbandingan KNN vs Decision Tree",
        "📄 Export Excel",
        "🌳 Visualisasi Decision Tree",
        "📥 Import Excel",
        "💬 Konsultasi",
        "🛠️ Panel Admin",
        "🧬 Tes Minat & Bakat",
        "🚪 Logout"
])   

    if menu == "🏠 Home":
        st.title("📘 Sistem Pendukung Keputusan Penentuan Jurusan")
        st.write(f"Hai **{st.session_state.username}**, selamat datang di aplikasi SPK dengan 5 kriteria utama:")
        st.markdown("""
        - **Matematika**
        - **Bahasa Inggris**
        - **Biologi**
        - **Fisika**
        - **Kimia**
        """)


    elif menu == "🧠 Tips Memilih Jurusan":
        st.header("🧠 Tips Memilih Jurusan")
        st.markdown("""
        Berikut beberapa tips untuk memilih jurusan yang tepat:

        1. **Kenali Minat dan Bakatmu** – Jangan ikut-ikutan teman, kenali apa yang kamu sukai dan bisa kamu lakukan dengan baik.
        2. **Lihat Nilai Akademik** – Gunakan data nilai untuk melihat kekuatan di mata pelajaran tertentu.
        3. **Pertimbangkan Karir ke Depan** – Cari tahu pekerjaan apa yang bisa kamu ambil dari jurusan tersebut.
        4. **Konsultasi dengan Guru BK atau Orang Tua** – Dapatkan masukan dari orang yang mengenalmu.
        5. **Coba Tes Minat Bakat Tambahan** – Banyak tes gratis atau dari sekolah yang bisa membantumu lebih objektif.
        """)

    elif menu == "📝 Input Nilai":
        st.header("📝 Input Nilai Siswa")
        nama = st.text_input("Nama Siswa")
        nilai_mtk = st.number_input("Nilai Matematika", 0, 100, 75)
        nilai_bing = st.number_input("Nilai Bahasa Inggris", 0, 100, 75)
        nilai_bio = st.number_input("Nilai Biologi", 0, 100, 75)
        nilai_fisika = st.number_input("Nilai Fisika", 0, 100, 75)
        nilai_kimia = st.number_input("Nilai Kimia", 0, 100, 75)
        minat = st.selectbox("Minat Siswa", ["Sains", "Bahasa", "Sosial", "Teknik", "Seni"])
        bakat = st.selectbox("Bakat Siswa", ["Logika", "Bahasa", "Analitik", "Kreatif", "Kinestetik"])

        if st.button("Simpan Nilai"):
            insert_data(nama, nilai_mtk, nilai_bing, nilai_bio, nilai_fisika, nilai_kimia, minat, bakat)
            st.success("✅ Nilai berhasil disimpan ke database!")

    elif menu == "📊 Hasil Prediksi":
        st.header("📊 Hasil Prediksi Jurusan")
        df = get_all_data()
        if not df.empty:
            st.subheader("Data Siswa Terbaru")
            latest = df.iloc[-1:]
            st.dataframe(latest)

            nilai_akhir = (
                latest["matematika"].values[0] * st.session_state.bobot["Matematika"] +
                latest["binggris"].values[0] * st.session_state.bobot["Bahasa Inggris"] +
                latest["biologi"].values[0] * st.session_state.bobot["Biologi"] +
                latest["fisika"].values[0] * st.session_state.bobot["Fisika"] +
                latest["kimia"].values[0] * st.session_state.bobot["Kimia"]
            )

            st.write("**Nilai akhir (berdasarkan bobot):**", round(nilai_akhir, 2))
            prediksi = "IPA" if nilai_akhir >= 80 else "IPS"

            minat_siswa = latest["minat"].values[0].title()
            bakat_siswa = latest["bakat"].values[0].title()

            # Penentuan perlu konsultasi atau tidak
            if (prediksi == "IPA" and minat_siswa in ["Sains", "Teknik"] and bakat_siswa in ["Logika", "Analitik"]) or \
            (prediksi == "IPS" and minat_siswa in ["Sosial", "Bahasa"] and bakat_siswa in ["Bahasa", "Kreatif"]):
                prediksi_final = prediksi
            else:
                prediksi_final = prediksi + " (Perlu Konsultasi)"

            st.write(f"**Minat:** {minat_siswa}, **Bakat:** {bakat_siswa}")

            # Cek cita-cita berdasarkan minat dan bakat
            cita = tentukan_cita_cita(minat_siswa, bakat_siswa)
            if "konsultasi" in cita.lower():
                st.warning(f"❌ Kombinasi ({minat_siswa}, {bakat_siswa}) belum tersedia dalam mapping.")
            st.write(f"💡 Cita-Cita yang Cocok: **{cita}**")

            st.success(f"🎓 Prediksi Jurusan Akhir: **{prediksi_final}**")
            st.success(f"🎓 Prediksi Jurusan: **{prediksi}**")

            # ==================== HASIL PREDIKSI TAMBAHAN ====================
            if 78 <= nilai_akhir <= 82:
                st.info("⚠️ Nilai mendekati ambang batas, pertimbangkan jurusan alternatif.")
                st.write("🔄 Alternatif: IPS jika kurang kuat di sains, IPA jika ingin tantangan.")
            simpan_log(latest['nama'].values[0], prediksi_final, nilai_akhir)
            # ==================== HASIL PREDIKSI TAMBAHAN ====================
        else:
            st.warning("⚠️ Belum ada data nilai disimpan.")

    elif menu == "📋 Tabel Perhitungan":
        st.header("📋 Tabel Perhitungan Nilai dan Jurusan")
        df = get_all_data()

        if not df.empty:
            search_nama = st.text_input("🔍 Cari Nama Siswa")
            if search_nama:
                df = df[df["nama"].str.contains(search_nama, case=False, na=False)]

            if not df.empty:
                # Hitung nilai akhir
                df["nilai_akhir"] = (
                    df["matematika"] * st.session_state.bobot["Matematika"] +
                    df["binggris"] * st.session_state.bobot["Bahasa Inggris"] +
                    df["biologi"] * st.session_state.bobot["Biologi"] +
                    df["fisika"] * st.session_state.bobot["Fisika"] +
                    df["kimia"] * st.session_state.bobot["Kimia"]
                ).round(2)

                # Prediksi jurusan
                df["jurusan"] = df["nilai_akhir"].apply(lambda x: "IPA" if x >= 80 else "IPS")

                # Normalisasi kapitalisasi
                df["minat"] = df["minat"].str.title()
                df["bakat"] = df["bakat"].str.title()

                # Tambahkan cita-cita
                df["cita_cita"] = df.apply(lambda row: tentukan_cita_cita(row["minat"], row["bakat"]), axis=1)

                st.dataframe(df[[
                    "id", "nama", "matematika", "binggris", "biologi", "fisika", "kimia",
                    "nilai_akhir", "jurusan", "minat", "bakat", "cita_cita"
                ]])

                # ✅ Export ke Excel
                import io
                buffer = io.BytesIO()
                df.to_excel(buffer, index=False)
                buffer.seek(0)
                st.download_button(
                    label="📅 Download Tabel Perhitungan (.xlsx)",
                    data=buffer,
                    file_name="tabel_perhitungan.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            else:
                st.warning("⚠️ Data tidak ditemukan sesuai pencarian.")
        else:   
            st.warning("⚠️ Tidak ada data siswa yang tersedia.")

    elif menu == "✏️ Edit / Update Nilai Siswa":
        st.header("✏️ Edit / Update Nilai Siswa")
        df = get_all_data()

        if df.empty:
            st.warning("⚠️ Tidak ada data untuk diedit.")
            st.stop()

        selected_id = st.selectbox("Pilih ID Siswa", df["id"])
        siswa = df[df["id"] == selected_id].iloc[0]

        nama = st.text_input("Nama", siswa["nama"])
        mtk = st.number_input("Matematika", 0, 100, int(siswa["matematika"]))
        bing = st.number_input("Bahasa Inggris", 0, 100, int(siswa["binggris"]))
        bio = st.number_input("Biologi", 0, 100, int(siswa["biologi"]))
        fis = st.number_input("Fisika", 0, 100, int(siswa["fisika"]))
        kim = st.number_input("Kimia", 0, 100, int(siswa["kimia"]))
        minat = st.selectbox("Minat", ["Sains", "Bahasa", "Sosial", "Teknik", "Seni"],
                            index=["Sains", "Bahasa", "Sosial", "Teknik", "Seni"].index(siswa["minat"].title()))
        bakat = st.selectbox("Bakat", ["Logika", "Bahasa", "Analitik", "Kreatif", "Kinestetik"],
                            index=["Logika", "Bahasa", "Analitik", "Kreatif", "Kinestetik"].index(siswa["bakat"].title()))

        if st.button("Update"):
            conn = sqlite3.connect("siswa.db")
            c = conn.cursor()
            c.execute("""
                UPDATE nilai_siswa
                SET nama=?, matematika=?, binggris=?, biologi=?, fisika=?, kimia=?, minat=?, bakat=?
                WHERE id=?
            """, (nama, mtk, bing, bio, fis, kim, minat, bakat, selected_id))
            conn.commit()
            conn.close()
            st.success("✅ Data berhasil diperbarui!")
            st.rerun()

    elif menu == "📈 Statistik Nilai":
        st.header("📈 Statistik Nilai Siswa")
        df_stat = get_all_data()
        if not df_stat.empty:
            st.subheader("📊 Jenis Grafik Nilai Angka")
            chart_type = st.selectbox("Pilih jenis grafik:", ["Rata-rata", "Histogram", "Boxplot"])

            fitur = ["matematika", "binggris", "biologi", "fisika", "kimia"]
            if chart_type == "Rata-rata":
                df_mean = df_stat[fitur].mean()
                st.bar_chart(df_mean)
            elif chart_type == "Histogram":
                selected_subject = st.selectbox("Pilih mata pelajaran:", fitur)
                st.bar_chart(df_stat[selected_subject])
            elif chart_type == "Boxplot":
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                df_stat[fitur].plot.box(ax=ax)
                st.pyplot(fig)

            st.subheader("📄 Statistik Deskriptif Nilai")
            st.dataframe(df_stat[fitur].describe())

            # ================= Minat dan Bakat ===================
            st.markdown("---")
            st.subheader("🧠 Statistik Minat dan Bakat Siswa")

            if "minat" in df_stat.columns and "bakat" in df_stat.columns:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**📌 Distribusi Minat**")
                    minat_counts = df_stat["minat"].value_counts()
                    st.bar_chart(minat_counts)
                    st.dataframe(minat_counts.reset_index().rename(columns={"index": "Minat", "minat": "Jumlah"}))

                with col2:
                    st.markdown("**📌 Distribusi Bakat**")
                    bakat_counts = df_stat["bakat"].value_counts()
                    st.bar_chart(bakat_counts)
                    st.dataframe(bakat_counts.reset_index().rename(columns={"index": "Bakat", "bakat": "Jumlah"}))
            else:
                st.info("Kolom 'minat' dan 'bakat' belum tersedia pada data.")
        else:
            st.info("Belum ada data siswa yang dimasukkan.")

    elif menu == "📊 Korelasi Nilai":
        st.header("📊 Korelasi Antar Mata Pelajaran")
        df_data = get_all_data()

        if df_data.empty:
            st.warning("⚠️ Belum ada data siswa untuk analisis korelasi.")
        else:
            import seaborn as sns
            import matplotlib.pyplot as plt
            from io import BytesIO

            # Data asli untuk korelasi
            fitur = ["matematika", "binggris", "biologi", "fisika", "kimia"]
            df_corr_only = df_data[fitur]

            # Hitung korelasi Pearson
            corr = df_corr_only.corr(method="pearson")

            # ---- Tambahan fitur nilai akhir & jurusan (tidak mempengaruhi corr) ----
            df_data["nilai_akhir"] = (
                df_data["matematika"] * st.session_state.bobot["Matematika"] +
                df_data["binggris"] * st.session_state.bobot["Bahasa Inggris"] +
                df_data["biologi"] * st.session_state.bobot["Biologi"] +
                df_data["fisika"] * st.session_state.bobot["Fisika"] +
                df_data["kimia"] * st.session_state.bobot["Kimia"]
            ).round(2)
            df_data["jurusan"] = df_data["nilai_akhir"].apply(lambda x: "IPA" if x >= 80 else "IPS")

            # Filter jurusan (opsional, untuk visualisasi tapi korelasi tetap dari semua data)
            st.subheader("🎯 Filter Data")
            jurusan_filter = st.selectbox("Pilih Jurusan", ["Semua", "IPA", "IPS"])
            if jurusan_filter != "Semua":
                df_corr_only = df_data[df_data["jurusan"] == jurusan_filter][fitur]
                corr = df_corr_only.corr(method="pearson")

            # Tampilkan heatmap
            st.subheader("📊 Heatmap Korelasi")
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax, vmin=-1, vmax=1)
            st.pyplot(fig)

            # Download tombol PNG
            buffer = BytesIO()
            fig.savefig(buffer, format="png", bbox_inches="tight")
            st.download_button(
                label="📥 Download Gambar Korelasi (.png)",
                data=buffer.getvalue(),
                file_name="korelasi_heatmap.png",
                mime="image/png"
            )

            # Tabel korelasi
            st.subheader("📄 Tabel Nilai Korelasi")
            st.dataframe(corr.style.background_gradient(cmap="coolwarm").format("{:.2f}"))

            # Penjelasan singkat
            st.markdown("""
            ℹ️ <small>
            Korelasi bernilai **mendekati +1** menunjukkan hubungan positif yang kuat antara dua mata pelajaran.
            Sebaliknya, **mendekati -1** menunjukkan hubungan negatif. Korelasi **mendekati 0** berarti tidak ada hubungan linear yang jelas.
            </small>
            """, unsafe_allow_html=True)

    elif menu == "🏆 Ranking Siswa":
        st.header("🏆 Ranking Siswa Berdasarkan Nilai Akhir")
        df_rank = get_all_data()

        if df_rank.empty:
            st.warning("⚠️ Belum ada data siswa untuk diranking.")
        else:
            # Hitung nilai akhir
            df_rank["nilai_akhir"] = (
                df_rank["matematika"] * st.session_state.bobot["Matematika"] +
                df_rank["binggris"] * st.session_state.bobot["Bahasa Inggris"] +
                df_rank["biologi"] * st.session_state.bobot["Biologi"] +
                df_rank["fisika"] * st.session_state.bobot["Fisika"] +
                df_rank["kimia"] * st.session_state.bobot["Kimia"]
            ).round(2)

            # Tentukan jurusan
            df_rank["jurusan"] = df_rank["nilai_akhir"].apply(lambda x: "IPA" if x >= 80 else "IPS")
            df_rank["minat"] = df_rank["minat"].str.title()
            df_rank["bakat"] = df_rank["bakat"].str.title()

            # Tentukan cita-cita
            df_rank["cita_cita"] = df_rank.apply(
                lambda row: tentukan_cita_cita(row["minat"], row["bakat"]),
                axis=1
            )

            # Tambahkan filter jurusan
            st.subheader("🎯 Filter dan Pencarian")
            jurusan_filter = st.selectbox("Pilih Jurusan", ["Semua", "IPA", "IPS"])
            if jurusan_filter != "Semua":
                df_rank = df_rank[df_rank["jurusan"] == jurusan_filter]

            # Filter berdasarkan nama
            search_nama = st.text_input("🔍 Cari Nama Siswa")
            if search_nama:
                df_rank = df_rank[df_rank["nama"].str.contains(search_nama, case=False, na=False)]

            # Ranking
            df_rank = df_rank.sort_values(by="nilai_akhir", ascending=False).reset_index(drop=True)
            df_rank.index += 1  # Mulai ranking dari 1

            st.subheader("📋 Tabel Ranking Lengkap")
            st.dataframe(df_rank[[
                "nama", "minat", "bakat", "nilai_akhir", "jurusan", "cita_cita"
            ]], use_container_width=True)

            # Grafik 10 siswa teratas dengan warna berbeda per jurusan
            st.subheader("📊 Grafik 10 Siswa Teratas")

            import matplotlib.pyplot as plt
            import seaborn as sns

            top10 = df_rank.head(10)
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(
                x="nilai_akhir",
                y="nama",
                hue="jurusan",
                data=top10,
                palette={"IPA": "#4CAF50", "IPS": "#2196F3"},
                dodge=False,
                ax=ax
            )
            ax.set_title("Top 10 Ranking Siswa")
            ax.set_xlabel("Nilai Akhir")
            ax.set_ylabel("Nama")
            st.pyplot(fig)

            # Tombol Download Excel
            import io
            buffer = io.BytesIO()
            df_rank.to_excel(buffer, index=True)
            buffer.seek(0)
            st.download_button(
                label="📥 Download Ranking Siswa (.xlsx)",
                data=buffer,
                file_name="ranking_siswa.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    elif menu == "📊 Distribusi Jurusan":
        st.header("📊 Distribusi Jumlah Siswa per Jurusan")
        df = get_all_data()

        if df.empty:
            st.warning("⚠️ Belum ada data.")
        else:
            # Hitung nilai akhir
            df["nilai_akhir"] = (
                df["matematika"] * st.session_state.bobot["Matematika"] +
                df["binggris"] * st.session_state.bobot["Bahasa Inggris"] +
                df["biologi"] * st.session_state.bobot["Biologi"] +
                df["fisika"] * st.session_state.bobot["Fisika"] +
                df["kimia"] * st.session_state.bobot["Kimia"]
            ).round(2)

            # Tentukan jurusan
            df["jurusan"] = df["nilai_akhir"].apply(lambda x: "IPA" if x >= 80 else "IPS")

            # Normalisasi kapitalisasi minat & bakat
            df["minat"] = df["minat"].str.title()
            df["bakat"] = df["bakat"].str.title()

            st.subheader("📊 Grafik Distribusi Jurusan (Bar & Pie)")
            count = df["jurusan"].value_counts()
            col1, col2 = st.columns(2)
            with col1:
                st.bar_chart(count)
            with col2:
                st.pyplot(count.plot.pie(autopct='%1.1f%%', ylabel='', title='Distribusi Jurusan').figure)

            st.write("Jumlah siswa tiap jurusan:")
            st.write(count)

            # Tambahan: Distribusi Jurusan per Minat
            st.subheader("📈 Distribusi Jurusan Berdasarkan Minat")
            minat_jurusan = df.groupby(["minat", "jurusan"]).size().unstack(fill_value=0)
            st.dataframe(minat_jurusan)

            # Tambahan: Distribusi Jurusan per Bakat
            st.subheader("📈 Distribusi Jurusan Berdasarkan Bakat")
            bakat_jurusan = df.groupby(["bakat", "jurusan"]).size().unstack(fill_value=0)
            st.dataframe(bakat_jurusan)


    elif menu == "⚖️ Atur Bobot":
        st.header("⚖️ Atur Bobot Mata Pelajaran")

        st.subheader("🔢 Bobot Saat Ini")
        st.table(pd.DataFrame.from_dict(st.session_state.bobot, orient="index", columns=["Bobot"]))

        st.subheader("🛠️ Ubah Nilai Bobot")
        for key in st.session_state.bobot:
            st.session_state.bobot[key] = st.slider(
                f"Bobot {key}", 0.0, 1.0, st.session_state.bobot[key], 0.05
            )

        total_bobot = sum(st.session_state.bobot.values())

        if total_bobot != 1.0:
            st.warning(f"⚠️ Total bobot saat ini adalah {round(total_bobot, 3)}. Idealnya total = 1.0")
        else:
            st.success("✅ Total bobot valid (1.0)")

        col1, col2 = st.columns(2)

        # Reset ke default
        with col1:
            if st.button("🔁 Reset ke Default (0.2)"):
                st.session_state.bobot = {
                    "Matematika": 0.2,
                    "Bahasa Inggris": 0.2,
                    "Biologi": 0.2,
                    "Fisika": 0.2,
                    "Kimia": 0.2
                }
                st.rerun()

        # Export bobot ke JSON
        with col2:
            import json
            import io
            bobot_json = json.dumps(st.session_state.bobot)
            st.download_button(
                label="💾 Simpan Bobot (.json)",
                data=bobot_json,
                file_name="bobot_siswa.json",
                mime="application/json"
            )
            st.subheader("📤 Impor Bobot dari File JSON")
            uploaded_bobot = st.file_uploader("Unggah file JSON", type=["json"])
        if uploaded_bobot is not None:
            try:
                bobot_baru = json.load(uploaded_bobot)
                if set(bobot_baru.keys()) == set(st.session_state.bobot.keys()):
                    st.session_state.bobot = {k: float(v) for k, v in bobot_baru.items()}
                    st.success("✅ Bobot berhasil diimpor!")
                    st.rerun()
                else:
                    st.error("❌ Struktur JSON tidak sesuai format bobot.")
            except Exception as e:
                st.error(f"❌ Gagal membaca file JSON: {e}")

     
    elif menu == "🌳 Visualisasi Decision Tree":
        st.header("🌳 Visualisasi Pohon Keputusan (Decision Tree)")
        df = get_all_data()

        if df.empty:
            st.warning("⚠️ Tidak ada data siswa untuk divisualisasikan.")
        else:
            
            # Normalisasi teks minat dan bakat, isi default jika kosong
            df["minat"] = df["minat"].fillna("Tidak Ada").str.title()
            df["bakat"] = df["bakat"].fillna("Tidak Ada").str.title()

            # Hitung nilai akhir berdasarkan bobot
            df["nilai_akhir"] = (
                df["matematika"] * st.session_state.bobot["Matematika"] +
                df["binggris"] * st.session_state.bobot["Bahasa Inggris"] +
                df["biologi"] * st.session_state.bobot["Biologi"] +
                df["fisika"] * st.session_state.bobot["Fisika"] +
                df["kimia"] * st.session_state.bobot["Kimia"]
            ).round(2)

            # Tentukan jurusan IPA/IPS
            df["jurusan"] = df["nilai_akhir"].apply(lambda x: "IPA" if x >= 80 else "IPS")

            # Siapkan fitur input: nilai + one-hot encoding minat dan bakat
            fitur_nilai = ["matematika", "binggris", "biologi", "fisika", "kimia"]
            df_encoded = pd.get_dummies(df[["minat", "bakat"]], prefix=["minat", "bakat"])
            X = pd.concat([df[fitur_nilai], df_encoded], axis=1)
            y = df["jurusan"]

            # Latih model Decision Tree (depth dinaikkan ke 5)
            clf = DecisionTreeClassifier(max_depth=5, random_state=42)
            clf.fit(X, y)

            # Visualisasi struktur pohon
            st.subheader("🧠 Struktur Pohon Keputusan")
            fig, ax = plt.subplots(figsize=(20, 10))
            plot_tree(
                clf,
                feature_names=X.columns,
                class_names=clf.classes_,
                filled=True,
                rounded=True,
                fontsize=10,
                ax=ax
            )
            st.pyplot(fig)

            # Tombol download PNG
            buffer = BytesIO()
            fig.savefig(buffer, format="png", bbox_inches="tight")
            st.download_button(
                label="📥 Download Pohon Keputusan (.png)",
                data=buffer.getvalue(),
                file_name="decision_tree.png",
                mime="image/png"
            )

            # Evaluasi akurasi
            y_pred = clf.predict(X)
            acc = accuracy_score(y, y_pred)
            st.subheader("✅ Akurasi Prediksi pada Data Ini")
            st.success(f"Akurasi: {acc * 100:.2f}%")

            # Classification Report
            with st.expander("📄 Lihat Classification Report"):
                report = classification_report(y, y_pred, output_dict=False)
                st.text(report)

            # Visualisasi Pengaruh Fitur
            if st.checkbox("📊 Tampilkan Pengaruh Fitur (Feature Importance)"):
                st.subheader("📈 Pengaruh Masing-Masing Fitur")
                importance = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=True)
                st.bar_chart(importance)


    elif menu == "📥 Import Excel": 
        st.header("📥 Import Data Nilai dari File Excel")

        uploaded_file = st.file_uploader("Unggah file Excel (.xlsx)", type=["xlsx"])
        if uploaded_file is not None:
            try:
                df_excel = pd.read_excel(uploaded_file)

                # Rename kolom agar sesuai standar aplikasi
                df_excel = df_excel.rename(columns={
                    "Nama": "nama",
                    "MAT": "matematika",
                    "BING": "binggris",
                    "BIO": "biologi",
                    "FIS": "fisika",
                    "KIMIA": "kimia",
                    "MINAT": "minat",
                    "BAKAT": "bakat"
                })

                # Kolom yang wajib ada
                required_columns = ["nama", "matematika", "binggris", "biologi", "fisika", "kimia", "minat", "bakat"]
                if all(col in df_excel.columns for col in required_columns):
                    for _, row in df_excel.iterrows():
                        insert_data(
                            row["nama"],
                            int(row["matematika"]),
                            int(row["binggris"]),
                            int(row["biologi"]),
                            int(row["fisika"]),
                            int(row["kimia"]),
                            row["minat"],
                            row["bakat"]
                        )
                    st.success("✅ Semua data dari Excel berhasil dimasukkan ke database.")
                    st.dataframe(df_excel)
                else:
                    st.error("❌ Format Excel salah. Pastikan nama kolom: " + ", ".join(required_columns))
            except Exception as e:
                st.error(f"❌ Terjadi error saat membaca file: {e}")

    
    elif menu == "🧪 Uji Akurasi KNN":
        st.header("🧪 Uji Akurasi Algoritma KNN")
        df = get_all_data()
        if df.shape[0] < 5:
            st.warning("⚠️ Minimal 5 data siswa diperlukan untuk uji akurasi.")
        else:
            from sklearn.model_selection import train_test_split
            from sklearn.neighbors import KNeighborsClassifier
            from sklearn.metrics import accuracy_score, confusion_matrix

            # Tentukan jurusan berdasarkan nilai akhir
            df["jurusan"] = ["IPA" if (
                r["matematika"] * 0.2 + r["binggris"] * 0.2 +
                r["biologi"] * 0.2 + r["fisika"] * 0.2 + r["kimia"] * 0.2
            ) >= 80 else "IPS" for i, r in df.iterrows()]

            fitur = ["matematika", "binggris", "biologi", "fisika", "kimia"]
            X = df[fitur]
            y = df["jurusan"]
            nama_siswa = df["nama"]  # ✅ simpan nama siswa

            # Split data termasuk nama siswa
            X_train, X_test, y_train, y_test, nama_train_split, nama_test_split = train_test_split(
                X, y, nama_siswa, test_size=0.3, random_state=42
            )

            # Latih model KNN
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            # === Tabel K Tetangga ===
            from sklearn.neighbors import NearestNeighbors
            k = 3
            nbrs = NearestNeighbors(n_neighbors=k)
            nbrs.fit(X_train)
            distances, indices = nbrs.kneighbors(X_test)

            st.subheader(f"📋 Tabel {k} Tetangga Terdekat dan Prediksi")
            table_data = []
            y_train = y_train.reset_index(drop=True)
            X_train_reset = X_train.reset_index(drop=True)
            nama_train_reset = nama_train_split.reset_index(drop=True)

            for i, (dist_list, idx_list) in enumerate(zip(distances, indices)):
                tetangga = []
                jurusan_tetangga = []
                for j in range(k):
                    idx_train = idx_list[j]
                    nama_tetangga = nama_train_reset[idx_train]
                    jurusan_tetangga_item = y_train[idx_train]
                    jarak = dist_list[j]
                    tetangga.append(f"{nama_tetangga} (jarak: {jarak:.2f})")
                    jurusan_tetangga.append(jurusan_tetangga_item)

                from collections import Counter
                prediksi = Counter(jurusan_tetangga).most_common(1)[0][0]
                hasil_asli = y_test.iloc[i]
                benar = "✅" if prediksi == hasil_asli else "❌"

                table_data.append({
                    "Data Uji ke": i + 1,
                    "Nama Data Uji": nama_test_split.iloc[i],  # ✅ tampilkan nama siswa uji
                    "Tetangga Terdekat": ", ".join(tetangga),
                    "Prediksi": prediksi,
                    "Asli": hasil_asli,
                    "Status": benar
                })

            df_tetangga = pd.DataFrame(table_data)
            st.dataframe(df_tetangga)

            # ✅ Tombol download ke Excel
            import io
            excel_buffer = io.BytesIO()
            df_tetangga.to_excel(excel_buffer, index=False)
            excel_buffer.seek(0)
            st.download_button(
                label="📥 Download Tabel K Tetangga (.xlsx)",
                data=excel_buffer,
                file_name="tabel_k_tetangga.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # === Tampilkan Akurasi ===
            st.success(f"🎯 Akurasi KNN: {acc * 100:.2f}%")
            st.subheader("📊 Confusion Matrix")
            st.write(cm)


    elif menu == "🧪 Uji Akurasi Decision Tree":
            st.header("🧪 Uji Akurasi Algoritma Decision Tree")
            df = get_all_data()
            if df.shape[0] < 5:
                st.warning("⚠️ Minimal 5 data siswa diperlukan untuk uji akurasi.")
            else:
                from sklearn.model_selection import train_test_split
                from sklearn.tree import DecisionTreeClassifier
                from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

                df["jurusan"] = ["IPA" if (
                    r["matematika"] * 0.2 + r["binggris"] * 0.2 +
                    r["biologi"] * 0.2 + r["fisika"] * 0.2 + r["kimia"] * 0.2
                ) >= 80 else "IPS" for i, r in df.iterrows()]

                fitur = ["matematika", "binggris", "biologi", "fisika", "kimia"]
                X = df[fitur]
                y = df["jurusan"]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
                clf = DecisionTreeClassifier(max_depth=3)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                cm = confusion_matrix(y_test, y_pred)

                st.success(f"🎯 Akurasi Decision Tree: {acc * 100:.2f}%")
                st.subheader("📊 Confusion Matrix")
                st.write(cm)

                report = classification_report(y_test, y_pred, output_dict=True)
                st.subheader("📋 Classification Report")
                st.json(report)

    elif menu == "🔍 Perbandingan KNN vs Decision Tree":
        st.header("🔍 Perbandingan KNN vs Decision Tree")
        
        fitur_nilai = ["MAT", "BING", "BIO", "FIS", "KIMIA"]

        # Data default
        df = pd.read_csv("siswa_knn_data.csv", delimiter=';')

        uploaded_file = st.file_uploader("📤 Unggah file data siswa (.csv, delimiter ';')", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, delimiter=';')
            except Exception as e:
                st.error(f"❌ Gagal membaca file: {e}")
                st.stop()

        # Ubah nama kolom ke huruf besar semua agar konsisten
        df.columns = [col.upper() for col in df.columns]

        # Validasi kolom nilai wajib
        if not all(col in df.columns for col in fitur_nilai):
            st.error("❌ Kolom nilai tidak lengkap. Diperlukan: " + ", ".join(fitur_nilai))
            st.stop()

        # Tambahan fitur minat & bakat jika ada, tanpa peringatan
        if "MINAT" in df.columns and "BAKAT" in df.columns:
            df["MINAT"] = df["MINAT"].astype(str).str.title()
            df["BAKAT"] = df["BAKAT"].astype(str).str.title()
            dummies = pd.get_dummies(df[["MINAT", "BAKAT"]], prefix=["minat", "bakat"])
            fitur_categorical = list(dummies.columns)
            df = pd.concat([df, dummies], axis=1)
        else:
            fitur_categorical = []

        # Hitung jurusan
        df["NILAI_AKHIR"] = (
            df["MAT"] * 0.2 + df["BING"] * 0.2 +
            df["BIO"] * 0.2 + df["FIS"] * 0.2 + df["KIMIA"] * 0.2
        ).round(2)
        df["JURUSAN"] = df["NILAI_AKHIR"].apply(lambda x: "IPA" if x >= 80 else "IPS")

        # Gabungkan fitur
        X = df[fitur_nilai + fitur_categorical]
        y = df["JURUSAN"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # KNN
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        acc_knn = accuracy_score(y_test, y_pred_knn)
        cm_knn = confusion_matrix(y_test, y_pred_knn)
        report_knn = classification_report(y_test, y_pred_knn, output_dict=True)

        # Decision Tree
        tree = DecisionTreeClassifier(max_depth=3, random_state=42)
        tree.fit(X_train, y_train)
        y_pred_tree = tree.predict(X_test)
        acc_tree = accuracy_score(y_test, y_pred_tree)
        cm_tree = confusion_matrix(y_test, y_pred_tree)
        report_tree = classification_report(y_test, y_pred_tree, output_dict=True)

        # Tampilkan hasil
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔵 KNN")
            st.write(f"🎯 Akurasi: {acc_knn*100:.2f}%")
            st.write("📊 Confusion Matrix:")
            st.write(cm_knn)
            st.write("📋 Classification Report:")
            st.json(report_knn)

        with col2:
            st.subheader("🌳 Decision Tree")
            st.write(f"🎯 Akurasi: {acc_tree*100:.2f}%")
            st.write("📊 Confusion Matrix:")
            st.write(cm_tree)
            st.write("📋 Classification Report:")
            st.json(report_tree)

    elif menu == "📄 Export Excel":
        st.header("📄 Export Data Siswa ke Excel")
        df_export = get_all_data()

        if not df_export.empty:
            # Normalisasi kolom minat dan bakat
            df_export["minat"] = df_export["minat"].str.title()
            df_export["bakat"] = df_export["bakat"].str.title()

            # Tambahkan kolom nilai akhir & jurusan
            df_export["nilai_akhir"] = (
                df_export["matematika"] * st.session_state.bobot["Matematika"] +
                df_export["binggris"] * st.session_state.bobot["Bahasa Inggris"] +
                df_export["biologi"] * st.session_state.bobot["Biologi"] +
                df_export["fisika"] * st.session_state.bobot["Fisika"] +
                df_export["kimia"] * st.session_state.bobot["Kimia"]
            ).round(2)

            df_export["jurusan"] = df_export["nilai_akhir"].apply(lambda x: "IPA" if x >= 80 else "IPS")

            # Tentukan cita-cita berdasarkan minat dan bakat
            df_export["cita_cita"] = df_export.apply(
                lambda row: tentukan_cita_cita(row["minat"], row["bakat"]),
                axis=1
            )

            # Tentukan urutan kolom yang diekspor
            kolom_export = [
                "id", "nama", "matematika", "binggris", "biologi", "fisika", "kimia",
                "minat", "bakat", "nilai_akhir", "jurusan", "cita_cita"
            ]

            # Simpan ke Excel
            import io
            buffer = io.BytesIO()
            df_export.to_excel(buffer, index=False, columns=kolom_export)
            buffer.seek(0)

            # Tombol download
            st.download_button(
                label="📥 Download Excel Lengkap",
                data=buffer,
                file_name="data_nilai_siswa_lengkap.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("⚠️ Tidak ada data untuk diekspor.")

    elif menu == "💬 Konsultasi":
        st.header("💬 Konsultasi Penjurusan dan Cita-cita")
        st.markdown("### 🔍 Penasaran dengan jurusan yang cocok?")
        st.write("Berikut adalah **saran cita-cita** berdasarkan _minat_ dan _bakat_ siswa. Jika hasil prediksi sebelumnya tidak jelas atau bertanda 'Perlu Konsultasi', silakan gunakan panduan ini.")

        minat = st.selectbox("Pilih Minat Siswa", ["Sains", "Teknik", "Desain", "Sosial", "Seni"])
        bakat = st.selectbox("Pilih Bakat Siswa", ["Logika", "Analitik", "Kreatif", "Kinestetik", "Bahasa"])

        hasil_cita = tentukan_cita_cita(minat, bakat)

        st.markdown(f"### 🎯 Rekomendasi Cita-Cita:\n**{hasil_cita}**")

        if "konsultasi lebih lanjut" in hasil_cita.lower():
            st.warning("⚠️ Tidak ditemukan saran spesifik untuk kombinasi ini. Silakan konsultasi manual dengan guru BK.")

        st.markdown("---")
        st.subheader("📨 Ajukan Pertanyaan Konsultasi (Opsional)")
        pertanyaan = st.text_area("Tulis pertanyaan Anda seputar jurusan atau masa depan karir:")
        if st.button("Kirim Pertanyaan"):
            with open("konsultasi_pertanyaan.txt", "a") as f:
                f.write(f"{pertanyaan}\n")
            st.success("✅ Pertanyaan konsultasi telah dikirim. Silakan tunggu jawaban dari admin atau guru BK.")

    elif menu == "🛠️ Panel Admin":
        if st.session_state.username != "admin":
            st.error("❌ Akses hanya untuk admin.")
        else:
            st.header("🛠️ Panel Admin - Data Semua Siswa")
            df_all = get_all_data()

            # 🔍 Cari siswa berdasarkan nama
            keyword = st.text_input("Cari nama siswa:")
            if keyword:
                df_all = df_all[df_all["nama"].str.contains(keyword, case=False, na=False)]

            st.dataframe(df_all)

            # ✏️ Edit Data Siswa
            st.subheader("✏️ Edit Data Siswa")
            id_edit = st.number_input("Masukkan ID siswa yang akan diedit:", min_value=1, step=1)
            if id_edit in df_all["id"].values:
                row = df_all[df_all["id"] == id_edit].iloc[0]
                nama_baru = st.text_input("Nama", row["nama"])
                mtk_baru = st.number_input("Matematika", 0, 100, int(row["matematika"]))
                bing_baru = st.number_input("Bahasa Inggris", 0, 100, int(row["binggris"]))
                bio_baru = st.number_input("Biologi", 0, 100, int(row["biologi"]))
                fis_baru = st.number_input("Fisika", 0, 100, int(row["fisika"]))
                kim_baru = st.number_input("Kimia", 0, 100, int(row["kimia"]))
                minat_baru = st.selectbox("Minat", ["Sains", "Sosial", "Bahasa"], index=["Sains", "Sosial", "Bahasa"].index(row["minat"]))
                bakat_baru = st.selectbox("Bakat", ["Logika", "Verbal", "Visual"], index=["Logika", "Verbal", "Visual"].index(row["bakat"]))

                if st.button("💾 Simpan Perubahan"):
                    update_data(
                        id_edit, nama_baru, mtk_baru, bing_baru, bio_baru,
                        fis_baru, kim_baru, minat_baru, bakat_baru
                    )
                    log_admin_action("EDIT", id=id_edit, nama=nama_baru)
                    st.success("✅ Data siswa berhasil diperbarui.")

            # 🗑️ Hapus Berdasarkan ID (dengan konfirmasi)
            st.subheader("🗑️ Hapus Data Siswa")
            delete_id = st.number_input("Masukkan ID untuk menghapus", min_value=1, step=1, key="hapus")
            if delete_id:
                if st.button("Konfirmasi Hapus"):
                    nama_hapus = df_all[df_all["id"] == delete_id]["nama"].values[0] if delete_id in df_all["id"].values else "-"
                    delete_data(delete_id)
                    log_admin_action("HAPUS", id=delete_id, nama=nama_hapus)
                    st.success(f"✅ Data siswa dengan ID {delete_id} telah dihapus.")
        with st.expander("📋 Lihat Log Aktivitas Admin"):
            try:
               with open("admin_log.txt", "r") as log:
                 st.text(log.read())
            except FileNotFoundError:
                   st.info("Belum ada aktivitas admin yang tercatat.")


            # ================== FITUR LIHAT PERTANYAAN KONSULTASI ==================
            st.markdown("---")
            st.subheader("📩 Pertanyaan Konsultasi Siswa")
            konsultasi_file = "konsultasi_pertanyaan.txt"

            if os.path.exists(konsultasi_file):
                with open(konsultasi_file, "r") as f:
                    konsultasi_list = [x.strip() for x in f.readlines() if x.strip()]
                if konsultasi_list:
                    for i, item in enumerate(konsultasi_list, 1):
                        st.markdown(f"**{i}.** {item}")
                else:
                    st.info("📭 Belum ada pertanyaan konsultasi yang masuk.")
            else:
                st.info("📭 File konsultasi belum dibuat.")


    elif menu == "🧬 Tes Minat & Bakat":
        st.header("🧬 Tes Minat dan Bakat Siswa")

        st.markdown("Jawablah pertanyaan berikut untuk mengetahui potensi minat dan bakatmu:")

        # Skor awal
        skor_minat = {"Sains": 0, "Sosial": 0, "Bahasa": 0, "Teknik": 0, "Seni": 0}
        skor_bakat = {"Logika": 0, "Analitik": 0, "Kreatif": 0, "Kinestetik": 0, "Bahasa": 0}

        q1 = st.radio("1. Mana kegiatan yang paling kamu suka?", [
            "Mengerjakan eksperimen", "Menggambar & desain", "Berolahraga", "Membaca dan menulis", "Menganalisis data"
        ])
        q2 = st.radio("2. Mata pelajaran favorit kamu?", [
            "Biologi / Kimia", "Bahasa Indonesia / Inggris", "Matematika", "Seni Budaya", "Penjaskes"
        ])
        q3 = st.radio("3. Saat kerja kelompok, kamu lebih suka jadi:", [
            "Penyusun laporan", "Pemikir strategi", "Pelaksana teknis", "Penggambar / kreator", "Presenter"
        ])
        q4 = st.radio("4. Cita-cita yang kamu impikan?", [
            "Ilmuwan", "Seniman", "Guru Bahasa", "Insinyur", "Atlet / pelatih"
        ])
        q5 = st.radio("5. Aktivitas yang kamu paling semangat melakukannya?", [
            "Menyelesaikan soal logika", "Bercerita atau pidato", "Membuat desain kreatif", "Bergerak dan aktif", "Menganalisis masalah"
        ])

        if st.button("🔍 Lihat Hasil Tes"):
            # Penilaian minat
            if q1 == "Mengerjakan eksperimen": skor_minat["Sains"] += 1
            if q1 == "Menggambar & desain": skor_minat["Seni"] += 1
            if q1 == "Berolahraga": skor_minat["Seni"] += 1
            if q1 == "Membaca dan menulis": skor_minat["Bahasa"] += 1
            if q1 == "Menganalisis data": skor_minat["Teknik"] += 1

            if q2 == "Biologi / Kimia": skor_minat["Sains"] += 1
            if q2 == "Bahasa Indonesia / Inggris": skor_minat["Bahasa"] += 1
            if q2 == "Matematika": skor_minat["Teknik"] += 1
            if q2 == "Seni Budaya": skor_minat["Seni"] += 1
            if q2 == "Penjaskes": skor_minat["Sosial"] += 1

            # Penilaian bakat
            if q3 == "Penyusun laporan": skor_bakat["Bahasa"] += 1
            if q3 == "Pemikir strategi": skor_bakat["Analitik"] += 1
            if q3 == "Pelaksana teknis": skor_bakat["Kinestetik"] += 1
            if q3 == "Penggambar / kreator": skor_bakat["Kreatif"] += 1
            if q3 == "Presenter": skor_bakat["Bahasa"] += 1

            if q4 == "Ilmuwan": skor_bakat["Logika"] += 1
            if q4 == "Seniman": skor_bakat["Kreatif"] += 1
            if q4 == "Guru Bahasa": skor_bakat["Bahasa"] += 1
            if q4 == "Insinyur": skor_bakat["Analitik"] += 1
            if q4 == "Atlet / pelatih": skor_bakat["Kinestetik"] += 1

            if q5 == "Menyelesaikan soal logika": skor_bakat["Logika"] += 1
            if q5 == "Bercerita atau pidato": skor_bakat["Bahasa"] += 1
            if q5 == "Membuat desain kreatif": skor_bakat["Kreatif"] += 1
            if q5 == "Bergerak dan aktif": skor_bakat["Kinestetik"] += 1
            if q5 == "Menganalisis masalah": skor_bakat["Analitik"] += 1

            # Ambil hasil tertinggi
            minat_tertinggi = max(skor_minat, key=skor_minat.get)
            bakat_tertinggi = max(skor_bakat, key=skor_bakat.get)

            st.success(f"🎯 Hasil Minat: **{minat_tertinggi}**, Bakat: **{bakat_tertinggi}**")

            # Tampilkan saran cita-cita
            saran = tentukan_cita_cita(minat_tertinggi, bakat_tertinggi)
            st.markdown(f"💡 **Rekomendasi Cita-Cita:**\n{saran}")

            if "konsultasi" in saran.lower():
                st.info("⚠️ Untuk kombinasi ini, disarankan konsultasi lebih lanjut dengan guru BK.")

    elif menu == "🚪 Logout":
        logout()