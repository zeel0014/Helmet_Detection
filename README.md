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

## 📊 Dataset & Training

### Dataset Source

This project uses a **public helmet detection dataset** sourced from **[Roboflow](https://universe.roboflow.com/varad-codemonk-tata/helmet-detection-w1r9b)**:

- **Dataset Type**: Public Computer Vision Dataset
- **Classes**: Helmet detection (with/without helmet)
- **Format**: Annotated images with bounding boxes
- **Source**: Roboflow Universe - Public Datasets
- **License**: Public Domain / Open Source

---

## 🛠️ Technologies Used

- **[Python](https://www.python.org/)** - Core programming language
- **[Streamlit](https://streamlit.io/)** - Web application framework
- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)** - Object detection models (YOLOv8 & YOLOv11)
- **[OpenCV](https://opencv.org/)** - Computer vision and video processing
- **[NumPy](https://numpy.org/)** - Numerical computations

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (optional)

### Step 1: Clone the Repository

```bash
git clone https://github.com/zeel0014/Helmet_Detection.git
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

**Developed by**: [@zeel0014](https://github.com/zeel0014)  
**GitHub Repository**: [Multiple-Disease-Prediction](https://github.com/zeel0014/Multiple-Disease-Prediction)
- **Email**: zeel2659@gmail.com

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

Made with ❤️ using Python & YOLO

</div>
