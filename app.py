import streamlit as st
from ultralytics import YOLO
import pandas as pd
from PIL import Image
from datetime import datetime

# ===========================
# PAGE CONFIGURATION
# ===========================
st.set_page_config(
    page_title="YOLOv8n Detection App",
    page_icon="🚗",
    layout="wide"
)

# ===========================
# CUSTOM STYLING
# ===========================
st.markdown("""
<style>
    .main {background-color: #0d1733;}
    div[data-testid="stSidebar"] {background-color: #0a1027;}
    h1, h2, h3, h4, h5, h6, p, label, span {color: #e0e6f1 !important;}
    .metric {background-color: #111b3a; border-radius: 10px; padding: 15px;}
    .block-container {padding-top: 2rem;}
    .stButton>button {
        background-color: #0ea5e9;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {background-color: #0284c7;}
    footer {visibility: hidden;}
    .footer-container {
        background-color: #0a1027;
        padding: 30px;
        border-top: 1px solid #1e293b;
        margin-top: 50px;
        color: #e0e6f1;
    }
    .footer-container a {color: #38bdf8; text-decoration: none;}
    .footer-container a:hover {text-decoration: underline;}
</style>
""", unsafe_allow_html=True)

# ===========================
# SIDEBAR SECTION
# ===========================
with st.sidebar:
    st.title("⚙️ Settings")

    st.markdown("### 📦 Model Info")
    st.write("**Model:** YOLOv8n")
    st.write("**Classes:** Car, Bike")
    st.write("**Path:** best.pt")

    conf = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
    mode = st.radio("Mode Detection", ["Image", "Video"])

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    st.markdown("""
    ### 💡 Usage Tips
    - Use clear, high-quality images  
    - Adjust confidence threshold for accuracy  
    - Ensure objects are clearly visible  
    - Best results with good lighting
    """)
    classify_btn = st.button("🚀 Classify Now")

# ===========================
# MAIN HEADER
# ===========================
st.markdown("<h1 style='text-align:center;'>🚗🏍️ Car & Bike Object Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Revolutionary fast detection using YOLOv8n technology</p>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;'><button style='background-color:#0ea5e9;color:white;border:none;border-radius:8px;padding:8px 20px;'>Try Now</button></div>", unsafe_allow_html=True)

# ===========================
# IMAGE DISPLAY AREA
# ===========================
col1, col2 = st.columns(2)

if uploaded_file is not None and classify_btn:
    image = Image.open(uploaded_file)
    col1.subheader("🖼️ Original Image")
    col1.image(image, use_column_width=True)

    model = YOLO("best.pt")
    results = model.predict(image, conf=conf)
    boxes = results[0].boxes
    annotated_img = results[0].plot()

    col2.subheader("📸 Detection Result")
    col2.image(annotated_img, use_column_width=True)

    # Extract detection data
    data = []
    for box in boxes:
        cls = model.names[int(box.cls)]
        conf_score = float(box.conf)
        xywh = [round(x, 1) for x in box.xywh[0].tolist()]
        data.append([cls, conf_score, xywh])

    df = pd.DataFrame(data, columns=["Class", "Confidence", "Bounding Box"])

    # ===========================
    # DETECTION STATISTICS
    # ===========================
    st.markdown("### 📊 Detection Statistics")
    car_count = sum(df["Class"] == "Car")
    bike_count = sum(df["Class"] == "Bike")
    total_count = len(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("🚗 CAR", f"{car_count}", f"{(car_count/total_count)*100:.0f}% of Total")
    c2.metric("🏍️ BIKE", f"{bike_count}", f"{(bike_count/total_count)*100:.0f}% of Total")
    c3.metric("📦 TOTAL", f"{total_count}", "Objects Detected")

    # ===========================
    # DETECTION DETAILS TABLE
    # ===========================
    st.markdown("### 📝 Detection Details")
    st.dataframe(df, use_container_width=True)

    # Download CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Report (CSV)",
        data=csv,
        file_name=f"detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

else:
    col1.subheader("🖼️ Original Image")
    col1.markdown("<div style='height:350px;background-color:#111b3a;border-radius:10px;display:flex;align-items:center;justify-content:center;color:#94a3b8;'>Upload an image to start detection</div>", unsafe_allow_html=True)
    col2.subheader("📸 Detection Result")
    col2.markdown("<div style='height:350px;background-color:#111b3a;border-radius:10px;display:flex;align-items:center;justify-content:center;color:#94a3b8;'>Detection result will appear here</div>", unsafe_allow_html=True)

# ===========================
# FOOTER SECTION
# ===========================
st.markdown("""
<hr style="border-color:#1e293b;">
<div style="display:flex;flex-wrap:wrap;justify-content:space-between;gap:20px;">

    <div style="flex:1;min-width:250px;">
        <h4>🚀 YOLOv8n Detection</h4>
        <p>Advanced AI-powered object detection system specializing in car and bike recognition using state-of-the-art YOLOv8n technology.</p>
    </div>

    <div style="flex:1;min-width:250px;">
        <h4>👩‍💻 Creator</h4>
        <p><b>Name:</b> Jilan Putri</p>
        <p><b>Role:</b> AI Developer / Data Scientist (Aamiin ya Allah)</p>
        <p><b>Email:</b> jilanptr06@gmail.com</p>
        <p><b>University:</b> Universitas Syiah Kuala</p>
    </div>

    <div style="flex:1;min-width:250px;">
        <h4>🔗 Connect With Me</h4>
        <p>
            <a href="https://github.com/jilan20/Car-Bike-Detection" target="_blank">GitHub</a> ·
            <a href="https://www.linkedin.com/in/jilan-putri-malisa" target="_blank">LinkedIn</a> ·
            <a href="mailto:jilanptr06@gmail.com">Email</a>
        </p>
        <p style="font-size:13px;color:#64748b;">Built with Streamlit + YOLOv8n</p>
    </div>
</div>

<hr style="border-color:#1e293b;">
<div style="text-align:center;color:#64748b;font-size:13px;">
    © 2025 Jilan Putri. All rights reserved. | Version 1.0.0
</div>
""", unsafe_allow_html=True)
