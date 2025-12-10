# main.py (Final Updated Version - No Warnings, Cloud Ready)

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import tempfile
import shutil

st.set_page_config(page_title="Safety Helmet Detection", layout="centered")
st.title("Safety Helmet Detection System")
st.markdown("Upload image or video — helmet detection in seconds!")

# Sidebar
st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox("Select Trained Model", ["YOLOv8s", "YOLOv11s"])

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.10, 1.0, 0.45, 0.05)

# Tumhare custom model ke naam yahan daal do (repo root mein hone chahiye)
model_paths = {"YOLOv8s": "model/yolo8s.pt", "YOLOv11s": "model/yolo11s.pt"}

@st.cache_resource
def load_model(choice):
    path = model_paths[choice]
    if not os.path.exists(path):
        st.error(f"Model not found: {path}")
        st.info("Upload your custom `.pt` file in the repo root.")
        st.stop()
    return YOLO(path)

model = load_model(model_choice)
st.sidebar.success(f"Loaded: {model_choice}")

input_type = st.sidebar.radio("Choose Input", ["Image", "Video"], horizontal=True)

# ============================= IMAGE =============================
if input_type == "Image":
    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpg", "jpeg", "png", "webp"])

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Original Image", width=700)

        with st.spinner("Detecting helmets..."):
            # stream=False rakha hai kyunki single image hai → RAM issue nahi hoga
            results = model(image, conf=conf_threshold, verbose=False)[0]
            annotated = results.plot()

        st.image(annotated, caption="Detection Result", width=700)
        st.success("Detection Complete!")

# ============================= VIDEO =============================
else:
    uploaded_video = st.file_uploader("Upload Video (MP4 recommended)", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_video:
        # Save uploaded video temporarily
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_input.write(uploaded_video.read())
        temp_input.close()

        # Temp folder for YOLO output
        temp_dir = tempfile.mkdtemp()

        try:
            with st.spinner("Processing video... sabr rakho, 20–90 sec lagega"):
                # YOLO ka built-in save — Cloud pe 101% kaam karta hai
                model.predict(
                    source=temp_input.name,
                    conf=conf_threshold,
                    save=True,
                    project=temp_dir,
                    name="output",
                    exist_ok=True,
                    stream=False,        # Video ke liye bhi False rakh sakte hain (safe)
                    verbose=False,
                    imgsz=640,
                    device="cpu"
                )

            # Output video path find karo
            output_folder = os.path.join(temp_dir, "output")
            video_files = [f for f in os.listdir(output_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

            if not video_files:
                st.error("YOLO ne video save nahi kiya. Format issue ho sakta hai.")
                st.stop()

            result_video = os.path.join(output_folder, video_files[0])

            # Show video
            with open(result_video, "rb") as f:
                st.video(f.read())

            st.success("Video processed successfully!")
            st.balloons()

        except Exception as e:
            st.error(f"Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

        finally:
            # Cleanup
            try:
                os.unlink(temp_input.name)
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass
