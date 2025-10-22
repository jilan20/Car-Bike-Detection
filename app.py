import streamlit as st
from ultralytics import YOLO
import pandas as pd
import numpy as np
from PIL import Image
import io
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
# CUSTOM CSS STYLE
# ===========================
st.markdown("""
    <style>
        .main {background-color: #0d1733;}
        div[data-testid="stSidebar"] {background-color: #0a1027;}
        h1, h2, h3, h4, h5, h6, p, label, span {
            color: #e0e6f1 !important;
        }
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
    st.markdown("> 💡 **Tips for usage**\n- Image must be clear and well-lit.\n- Adjust confidence threshold as needed.\n- Ensure objects are clearly visible.")
    
    classify_btn = st.button("🚀 Classify Now")

# ===========================
# MAIN CONTENT
# ===========================
st.markdown(f"<h1 style='text-align:center;'>🚗🏍️ Car & Bike Object Detection with AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Revolutionary fast detection using YOLOv8n technology</p>", unsafe_allow_html=True)
st.markdown("<div style='text-align:center;'><button style='background-color:#0ea5e9;color:white;border:none;border-radius:8px;padding:8px 20px;'>Try Now</button></div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
if uploaded_file is not None and classify_btn:
    image = Image.open(uploaded_file)
    col1.subheader("🖼️ Original Image")
    col1.image(image, use_container_width=True)

    # Load YOLO model
    model = YOLO("best.pt")
    results = model.predict(image, conf=conf)
    boxes = results[0].boxes

    # Annotated image
    annotated_img = results[0].plot()
    col2.subheader("📸 Detection Result")
    col2.image(annotated_img, use_container_width=True)

    # Extract detection info
    data = []
    for box in boxes:
        cls = model.names[int(box.cls)]
        conf_score = float(box.conf)
        xywh = box.xywh[0].tolist()
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
    # DETECTION DETAILS
    # ===========================
    st.markdown("### 📝 Detection Details")
    st.dataframe(df, use_container_width=True)

    # Download CSV report
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Report (CSV)",
        data=csv,
        file_name=f"detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

else:
    # Initial state (before upload)
    col1.subheader("🖼️ Original Image")
    col1.image("https://via.placeholder.com/600x350?text=Upload+an+image+to+start+detection", use_container_width=True)

    col2.subheader("📸 Detection Result")
    col2.image("https://via.placeholder.com/600x350?text=Detection+result+will+appear+here", use_container_width=True)

