import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import os
import tempfile
import subprocess  # New import for video conversion

st.set_page_config(page_title="Helmet Detection", layout="centered")
st.title("Helmet Detection System")

# Sidebar
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox("Select Model", ["YOLOv8s", "YOLOv11s"])
conf_threshold = st.sidebar.slider("Confidence", 0.1, 1.0, 0.45, 0.05)

# Update these paths relative to your project structure
model_paths = {"YOLOv8s": "model/yolo8s.pt", "YOLOv11s": "model/yolo11s.pt"}

@st.cache_resource
def load_model(path):
    return YOLO(path)

model_path = model_paths[model_choice]
if not os.path.exists(model_path):
    st.error(f"Model not found: {model_path}")
    st.info("Please make sure model files are in the 'model/' folder.")
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
        # Intermediate output (OpenCV writes to this)
        output_video_path = os.path.join(temp_dir, f"output_{timestamp}.mp4")
        # Final output (Converted for Browser)
        converted_video_path = os.path.join(temp_dir, f"converted_{timestamp}.mp4")

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

                # LINUX/CLOUD FIX: Use 'mp4v' instead of 'H264' for OpenCV writing
                # This avoids the "Encoder not found" error
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

                if not out.isOpened():
                    st.error("Cannot create output video writer. Check permissions/codecs.")
                    cap.release()
                    st.stop()

                progress = st.progress(0)
                frame_count = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # YOLO Detection
                    result = model(frame, conf=conf_threshold, verbose=False)[0]
                    annotated = result.plot()
                    out.write(annotated)

                    frame_count += 1
                    if total > 0:
                        progress.progress(min(frame_count / total, 1.0))

                cap.release()
                out.release()
                progress.empty()

            # ================= CONVERSION STEP =================
            # OpenCV's mp4v often shows black screen in browsers.
            # We use FFmpeg to convert it to H.264 (libx264) for web compatibility.
            
            st.info("Optimizing video for web playback...")
            
            # FFmpeg command: Input -> output_video_path, Output -> converted_video_path
            # -y overwrites file, -c:v libx264 uses H.264 codec, -preset fast speeds it up
            convert_cmd = f"ffmpeg -y -i {output_video_path} -c:v libx264 -preset fast {converted_video_path}"
            
            conversion_status = subprocess.call(convert_cmd, shell=True)
            
            final_video_path = None
            
            if conversion_status == 0 and os.path.exists(converted_video_path):
                final_video_path = converted_video_path
            elif os.path.exists(output_video_path):
                st.warning("Video conversion failed. Showing original output (might not play in some browsers).")
                final_video_path = output_video_path
            else:
                st.error("Failed to generate video.")

            # Display Video
            if final_video_path:
                st.success("✅ Processing complete!")
                with open(final_video_path, "rb") as video_file:
                    video_bytes = video_file.read()
                st.video(video_bytes)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.text(traceback.format_exc())
