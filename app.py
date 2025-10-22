import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime

# ==========================
# Konfigurasi Halaman dengan Tema Dark
# ==========================
st.set_page_config(
    page_title="Deteksi Objek Car & Bike",
    layout="wide",
    theme="dark"  # Tema dark untuk tampilan yang menarik
)

# ==========================
# Load Models (disesuaikan dengan directory Anda)
# ==========================
@st.cache_resource
def load_models():
    try:
        yolo_model = YOLO("model/best.pt")  # Model deteksi objek (sesuai directory Anda)
        classifier = tf.keras.models.load_model("model/classifier_model.h5")  # Model klasifikasi (sesuai directory Anda)
        return yolo_model, classifier
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

yolo_model, classifier = load_models()

# ==========================
# Header dengan Waktu
# ==========================
current_time = datetime.now().strftime("%I:%M %p")
st.markdown(f"""
==================================================================================
| [Aplikasi Streamlit]                                             [{current_time}]  |
==================================================================================
""")

# ==========================
# Sidebar
# ==========================
with st.sidebar:
    st.markdown("⚙️ **SIDEBAR (PENGATURAN)**")
    st.markdown("---")
    
    # Mode Deteksi (disesuaikan dari kode Anda)
    st.markdown("**MODE DETEKSI**")
    mode = st.selectbox("", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
    st.markdown("---")
    
    # Info Model (khusus untuk YOLO)
    if mode == "Deteksi Objek (YOLO)":
        st.markdown("**INFO MODEL**")
        st.write("Revolusi deteksi cepat menggunakan YOLOv8n.")
        st.write("Model: YOLOv8n")
        st.write("Classes: Car, Bike")
        st.write("Path: model/best.pt")  # Sesuai directory Anda
        st.markdown("---")
        
        # Confidence Threshold
        st.markdown("**CONFIDENCE THRESHOLD**")
        confidence = st.slider("", min_value=0.10, max_value=1.00, value=0.25, step=0.01)
        st.markdown("---")
    
    # Upload Gambar
    st.markdown("**UPLOAD GAMBAR**")
    uploaded_file = st.file_uploader("Drag and drop file here", type=["jpg", "png", "jpeg"])
    st.markdown("---")
    
    # Tips Penggunaan
    st.markdown("💡 **TIPS PENGGUNAAN**")
    st.write("- Gambar harus jelas.")
    st.write("- Atur Confidence (untuk YOLO).")
    st.write("- Pastikan objek terlihat.")
    st.markdown("---")
    
    # Tombol Klasifikasi/Deteksi
    if st.button("Klasifikasi Sekarang" if mode == "Klasifikasi Gambar" else "Deteksi Sekarang"):
        st.success("Proses dimulai! (Placeholder)")

# ==========================
# Bagian Utama
# ==========================
st.markdown("🚗🏍️ **Deteksi Objek Car & Bike dengan AI**")
st.markdown("Revolusi deteksi cepat menggunakan YOLOv8n.")

# Tombol Coba Sekarang
if st.button("Coba Sekarang"):
    st.info("Fitur coba sekarang diaktifkan! (Placeholder)")

# Kolom untuk Gambar Asli dan Hasil
col1, col2 = st.columns(2)
with col1:
    st.markdown("**GAMBAR ASLI**")
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Gambar yang diunggah", use_column_width=True)
    else:
        st.write("[ Area untuk menampilkan Gambar yang diunggah ]")

with col2:
    st.markdown("**HASIL**")
    if uploaded_file is not None and yolo_model is not None and classifier is not None:
        if mode == "Deteksi Objek (YOLO)":
            # Deteksi objek menggunakan YOLO
            results = yolo_model(img, conf=confidence)
            result_img = results[0].plot()  # Gambar dengan bounding box
            st.image(result_img, caption="Hasil Deteksi dengan Bounding Box", use_column_width=True)
            
            # Statistik Deteksi (placeholder, sesuaikan dengan hasil YOLO)
            detections = results[0].boxes.data  # Data deteksi
            car_count = sum(1 for det in detections if int(det[5]) == 0)  # Asumsi class 0 = Car
            bike_count = sum(1 for det in detections if int(det[5]) == 1)  # Asumsi class 1 = Bike
            total_count = len(detections)
            
            st.markdown("---")
            st.markdown("📊 **STATISTIK DETEKSI**")
            col_car, col_bike, col_total = st.columns(3)
            with col_car:
                st.metric("🚗 CAR", car_count, f"{(car_count/total_count*100) if total_count > 0 else 0:.0f}% Total")
            with col_bike:
                st.metric("🏍️ BIKE", bike_count, f"{(bike_count/total_count*100) if total_count > 0 else 0:.0f}% Total")
            with col_total:
                st.metric("TOTAL", total_count)
            
            # Detail Deteksi
            st.markdown("---")
            st.markdown("📝 **DETAIL DETEKSI**")
            if total_count > 0:
                data = []
                for det in detections:
                    class_id = int(det[5])
                    class_name = "Car" if class_id == 0 else "Bike"
                    conf = float(det[4])
                    bbox = (float(det[0]), float(det[1]), float(det[2]), float(det[3]))  # x1, y1, x2, y2
                    data.append({"Kelas": class_name, "Confidence": f"{conf:.2f}", "Bounding Box": f"{bbox}"})
                df = pd.DataFrame(data)
                st.table(df)
                
                # Tombol Download Laporan
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Laporan (CSV)",
                    data=csv,
                    file_name="laporan_deteksi.csv",
                    mime="text/csv"
                )
            else:
                st.write("Tidak ada objek terdeteksi.")
        
        elif mode == "Klasifikasi Gambar":
            # Klasifikasi menggunakan TensorFlow
            img_resized = img.resize((224, 224))  # Sesuaikan ukuran dengan model Anda
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0
            
            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            prob = np.max(prediction)
            
            st.write("### Hasil Klasifikasi:")
            st.write(f"**Kelas Prediksi:** {class_index}")
            st.write(f"**Probabilitas:** {prob:.2f}")
            
            # Placeholder untuk statistik jika diperlukan
            st.markdown("---")
            st.markdown("📊 **STATISTIK KLASIFIKASI**")
            st.metric("Probabilitas Tertinggi", f"{prob:.2f}")
    else:
        st.write("[ Area untuk menampilkan Hasil ]")

# ==========================
# CSS Tambahan untuk Tema Dark yang Lebih Menarik (Opsional)
# ==========================
st.markdown("""
<style>
    .stApp {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .stSidebar {
        background-color: #2e2e2e;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stMetric {
        background-color: #333333;
        border-radius: 10px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)
