import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import os
import tempfile

st.set_page_config(page_title="Helmet Detection", layout="centered")
st.title("Helmet Detection System")

# Sidebar
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox("Select Model", ["YOLOv8s", "YOLOv11s"])
conf_threshold = st.sidebar.slider("Confidence", 0.1, 1.0, 0.45, 0.05)

model_paths = {"YOLOv8s": "model/yolo8s.pt", "YOLOv11s": "model/yolo11s.pt"}


@st.cache_resource
def load_model(path):
    return YOLO(path)


model_path = model_paths[model_choice]
if not os.path.exists(model_path):
    st.error(f"Model not found: {model_path}")
    st.stop()

model = load_model(model_path)
st.sidebar.success(f"Model Loaded: {model_choice}")

input_type = st.sidebar.radio("Input Type", ["Image", "Video"])

# ============================= IMAGE =============================
if input_type == "Image":
    uploaded = st.file_uploader(
        "Upload Image → Auto Detect", type=["jpg", "jpeg", "png"]
    )
    if uploaded:
        with st.spinner("Detecting..."):
            img = Image.open(uploaded)
            res = model(np.array(img), conf=conf_threshold, verbose=False)[0]
            result_img = res.plot()
            st.image(result_img, caption="Result", width=800)

# ============================= VIDEO =============================
else:
    uploaded_video = st.file_uploader(
        "Upload Video → Auto Process", type=["mp4", "mov", "avi", "mkv"]
    )

    if uploaded_video is not None:
        # Use tempfile for cloud compatibility
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_input:
            tmp_input.write(uploaded_video.read())
            input_video_path = tmp_input.name

        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_output:
            output_video_path = tmp_output.name

        try:
            # Processing
            with st.spinner("Processing Video..."):
                cap = cv2.VideoCapture(input_video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Use mp4v codec for better compatibility
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                progress = st.progress(0)
                i = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    result = model(frame, conf=conf_threshold, verbose=False)[0]
                    annotated = result.plot()
                    out.write(annotated)

                    i += 1
                    if total > 0:
                        progress.progress(i / total)

                cap.release()
                out.release()

            progress.empty()
            st.success("Processing complete!")

            # Read the output video as bytes and display
            with open(output_video_path, 'rb') as video_file:
                video_bytes = video_file.read()
                st.video(video_bytes)

        except Exception as e:
            st.error(f"Error processing video: {str(e)}")
        
        finally:
            # Cleanup temporary files
            try:
                if os.path.exists(input_video_path):
                    os.remove(input_video_path)
                if os.path.exists(output_video_path):
                    os.remove(output_video_path)
            except:
                pass
