"""Streamlit app for YOLOv8 image detection."""

from __future__ import annotations

import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(page_title="YOLOv8 Detector", page_icon="🎯", layout="centered")
st.title("YOLOv8 Real-Time Object Detection")
st.caption("Sadia Liaqat | 2025-MS-AI-005 | MS AI - TUF")

model_choice = st.selectbox(
    "Select a model",
    ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
)
conf = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.05)

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Original Image", use_container_width=True)

    if st.button("Detect Objects"):
        with st.spinner("Running detection..."):
            model = YOLO(model_choice)
            results = model.predict(np.array(image), conf=conf, verbose=False)
            annotated = results[0].plot()
            st.image(annotated, caption="Detected Objects", use_container_width=True)

            boxes = results[0].boxes
            st.success(f"{len(boxes)} objects detected.")

            for box in boxes:
                class_name = results[0].names[int(box.cls)]
                confidence = float(box.conf)
                st.write(f"- {class_name}: {confidence:.2%}")
