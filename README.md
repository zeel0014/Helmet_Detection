# 🛡️ Helmet Detection System

<img width="1615" height="700" alt="Capture" src="https://github.com/user-attachments/assets/6375747e-1fb4-427f-9f28-858fc42ff1b4" />

A real-time helmet detection system powered by YOLOv8 and YOLOv11, built with Streamlit for easy deployment and usage.
---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [Dataset & Training](#dataset--training)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Information](#model-information)
- [Deployment](#deployment)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This **Helmet Detection System** uses state-of-the-art YOLO (You Only Look Once) object detection models to identify whether people are wearing helmets in images and videos. The system provides a user-friendly web interface built with Streamlit, making it accessible for both technical and non-technical users.

---

## ✨ Features

- **Real-time Detection**: Process images and videos with instant results
- **Dual Model Support**: Choose between YOLOv8s and YOLOv11s models
- **Adjustable Confidence**: Fine-tune detection sensitivity with a slider
- **Progress Tracking**: Real-time progress bar for video processing
- **Clean UI**: Intuitive Streamlit interface
- **Video Processing**: Upload and process entire videos with annotated output
- **Image Detection**: Quick single-image helmet detection
- **Auto-save Results**: Processed videos are automatically saved and displayed

---

## 🎬 Demo

### Image Detection
Upload an image → Get instant helmet detection results with bounding boxes

### Video Processing
1. Upload a video file (MP4, MOV, AVI, MKV)
2. Watch real-time processing progress
3. View the annotated output video with detected helmets

---

## 📊 Dataset & Training

### Dataset Source

This project uses a **public helmet detection dataset** sourced from **[Roboflow](https://universe.roboflow.com/varad-codemonk-tata/helmet-detection-w1r9b)**:

- 📦 **Dataset Type**: Public Computer Vision Dataset
- 🏷️ **Classes**: Helmet detection (with/without helmet)
- 📸 **Format**: Annotated images with bounding boxes
- 🔗 **Source**: Roboflow Universe - Public Datasets
- ✅ **License**: Public Domain / Open Source

### Training Platforms

The YOLO models were **custom trained** using cloud-based platforms:

#### 🔬 **Kaggle Notebooks**
- Free GPU/TPU access for training
- Used for YOLOv8s model training
- Training time: ~2-4 hours on GPU
- [Kaggle](https://www.kaggle.com/) - Free tier with 30 hours/week GPU

#### 🧪 **Google Colab**
- Free cloud-based Jupyter notebooks
- Used for YOLOv11s model training
- Training time: ~2-4 hours on GPU
- [Google Colab](https://colab.research.google.com/) - Free tier with GPU access

### Training Process

1. **Data Preparation**:
   - Downloaded dataset from Roboflow
   - Preprocessed and augmented images
   - Split into train/validation/test sets

2. **Model Training**:
   - Fine-tuned YOLOv8s and YOLOv11s on helmet dataset
   - Used transfer learning from pre-trained COCO weights
   - Optimized hyperparameters for best performance

3. **Evaluation**:
   - Validated on test set
   - Achieved high mAP (mean Average Precision)
   - Optimized for real-time inference

### Training Configuration

```yaml
Epochs: 100-150
Batch Size: 16-32
Image Size: 640x640
Optimizer: AdamW
Learning Rate: 0.001
Augmentation: Yes (rotation, flip, brightness, contrast)
```

---

## 🛠️ Technologies Used

- **[Python](https://www.python.org/)** - Core programming language
- **[Streamlit](https://streamlit.io/)** - Web application framework
- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)** - Object detection models (YOLOv8 & YOLOv11)
- **[OpenCV](https://opencv.org/)** - Computer vision and video processing
- **[Pillow](https://pillow.readthedocs.io/)** - Image processing
- **[NumPy](https://numpy.org/)** - Numerical computations

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/helmet_detection.git
cd helmet_detection
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv myenv
myenv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv myenv
source myenv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Model Files

Ensure your model files are in the correct location:
```
helmet_detection/
└── model/
    ├── yolo8s.pt
    └── yolo11s.pt
```

---

## 🚀 Usage

### Running Locally

1. **Navigate to project directory:**
   ```bash
   cd helmet_detection
   ```

2. **Run the Streamlit app:**
   ```bash
   streamlit run app/main.py
   ```

3. **Open your browser:**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to the URL shown in the terminal

### Using the Application

#### Image Detection:
1. Select **"Image"** from the sidebar
2. Click **"Upload Image → Auto Detect"**
3. Choose an image file (JPG, JPEG, PNG)
4. View the detection results instantly

#### Video Processing:
1. Select **"Video"** from the sidebar
2. Click **"Upload Video → Auto Process"**
3. Choose a video file (MP4, MOV, AVI, MKV)
4. Wait for processing to complete
5. View the annotated output video

#### Settings:
- **Select Model**: Choose between YOLOv8s or YOLOv11s
- **Confidence Threshold**: Adjust from 0.1 to 1.0 (default: 0.45)

---

## 📁 Project Structure

```
helmet_detection/
├── app/
│   └── main.py              # Streamlit application
├── model/
│   ├── yolo8s.pt            # YOLOv8s trained model
│   └── yolo11s.pt           # YOLOv11s trained model
├── dataset/                 # Training dataset (if applicable)
├── notebooks/               # Jupyter notebooks for experiments
├── myenv/                   # Virtual environment (git-ignored)
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

---

## 🤖 Model Information

### YOLOv8s (Custom Trained)
- **Architecture**: YOLOv8 Small
- **Size**: ~22.5 MB
- **Training Platform**: Kaggle Notebooks
- **Dataset**: Roboflow Public Dataset
- **Speed**: Fast inference (~30-50 FPS on GPU)
- **Accuracy**: High precision for helmet detection
- **Base Weights**: COCO pre-trained
- **Fine-tuning**: Custom trained on helmet dataset

### YOLOv11s (Custom Trained)
- **Architecture**: YOLOv11 Small
- **Size**: ~19.2 MB
- **Training Platform**: Google Colab
- **Dataset**: Roboflow Public Dataset
- **Speed**: Optimized for speed (~40-60 FPS on GPU)
- **Accuracy**: Latest YOLO improvements with enhanced detection
- **Base Weights**: COCO pre-trained
- **Fine-tuning**: Custom trained on helmet dataset

### Model Performance

Both models are **custom trained** specifically for helmet detection using:
- ✅ Public dataset from Roboflow
- ✅ Transfer learning from COCO weights
- ✅ Cloud-based training (Kaggle & Google Colab)
- ✅ Optimized for real-time performance
- ✅ High accuracy on helmet detection tasks

---

## 🌐 Deployment

### Deploy to Streamlit Cloud (FREE)

1. **Push your code to GitHub**
2. **Go to**: [share.streamlit.io](https://share.streamlit.io)
3. **Sign in** with GitHub
4. **Click "New app"**
5. **Fill in the details:**
   - Repository: `yourusername/helmet_detection`
   - Branch: `main`
   - Main file path: `app/main.py`
6. **Click "Deploy"**
7. **Share your app URL** (e.g., `yourapp.streamlit.app`)

### Deploy to Hugging Face Spaces

1. **Create a new Space** at [huggingface.co/spaces](https://huggingface.co/spaces)
2. **Select SDK**: Streamlit
3. **Upload files:**
   - `app/main.py`
   - `model/` folder
   - `requirements.txt`
4. **Add a `README.md`** to your Space
5. **Your app will be live** at `huggingface.co/spaces/youruser/appname`

### Important Notes for Deployment:
- Ensure model files are uploaded (may require Git LFS for large files)
- Use `opencv-python-headless` in requirements.txt for server deployments
- Set appropriate memory limits for video processing

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Areas for Improvement:
- Add more helmet classes (different types)
- Implement real-time webcam detection
- Add database for storing detection logs
- Create REST API for integration
- Improve UI/UX design

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **[Roboflow](https://roboflow.com/)** for providing the public helmet detection dataset
- **[Ultralytics](https://github.com/ultralytics/ultralytics)** for the amazing YOLO models (YOLOv8 & YOLOv11)
- **[Kaggle](https://www.kaggle.com/)** for free GPU resources for model training
- **[Google Colab](https://colab.research.google.com/)** for cloud-based training environment
- **[Streamlit](https://streamlit.io/)** for the easy-to-use web framework
- **[OpenCV](https://opencv.org/)** community for computer vision tools
- All contributors and users of this project

---

## 📧 Contact

For questions, suggestions, or issues:

- **GitHub Issues**: [Create an issue](https://github.com/yourusername/helmet_detection/issues)
- **Email**: your.email@example.com

---

## 📊 Project Stats

- **Models**: 2 (YOLOv8s, YOLOv11s)
- **Input Formats**: Images (JPG, PNG) & Videos (MP4, MOV, AVI, MKV)
- **Framework**: Streamlit
- **Deployment**: Cloud-ready

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

Made with ❤️ using Python & YOLO

</div>
