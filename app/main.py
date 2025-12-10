import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import os
import tempfile
import base64

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
        # Create temp directory in system temp
        temp_dir = tempfile.gettempdir()
        
        # Generate unique filenames
        import time
        timestamp = str(int(time.time()))
        input_video_path = os.path.join(temp_dir, f"input_{timestamp}.mp4")
        output_video_path = os.path.join(temp_dir, f"output_{timestamp}.mp4")

        try:
            # Save uploaded video
            with open(input_video_path, "wb") as f:
                f.write(uploaded_video.read())

            # Processing
            with st.spinner("Processing Video... Please wait"):
                cap = cv2.VideoCapture(input_video_path)
                
                if not cap.isOpened():
                    st.error("Cannot open video file")
                    st.stop()
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Use H264 codec for web compatibility
                fourcc = cv2.VideoWriter_fourcc(*"H264")
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))
                
                # If H264 fails, try mp4v
                if not out.isOpened():
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

                if not out.isOpened():
                    st.error("Cannot create output video")
                    cap.release()
                    st.stop()

                progress = st.progress(0)
                frame_count = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    result = model(frame, conf=conf_threshold, verbose=False)[0]
                    annotated = result.plot()
                    out.write(annotated)

                    frame_count += 1
                    if total > 0:
                        progress.progress(min(frame_count / total, 1.0))

                cap.release()
                out.release()
                progress.empty()

            st.success("✅ Processing complete!")

            # Check if output file exists and has content
            if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
                # Display video
                st.video(output_video_path)
            else:
                st.error("Output video file not created properly")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
        
        finally:
            # Cleanup (but keep files for a bit for st.video to work)
            # Streamlit will handle cleanup automatically
            pass
