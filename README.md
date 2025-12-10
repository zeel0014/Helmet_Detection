# 🛡️ Helmet Detection System

https://github.com/user-attachments/assets/f6e513c3-1b05-4d68-8c83-8eaf61f1fbb6

A real-time helmet detection system powered by YOLOv8 and YOLOv11, built with Streamlit for easy deployment and usage.
---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Dataset](#-dataset)
- [Technologies Used](#️-technologies-used)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [License](#-license)

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

## 📊 Dataset

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

## 🙏 Acknowledgments

- **[Roboflow](https://roboflow.com/)** for providing the public helmet detection dataset
- **[Ultralytics](https://github.com/ultralytics/ultralytics)** for the amazing YOLO models (YOLOv8 & YOLOv11)
- **[Kaggle](https://www.kaggle.com/)** for free GPU resources for model training
- **[Google Colab](https://colab.research.google.com/)** for cloud-based training environment
- **[Streamlit](https://streamlit.io/)** for the easy-to-use web framework
- **[OpenCV](https://opencv.org/)** community for computer vision tools
- All contributors and users of this project

---


## ⚠️ Disclaimer

This project is developed for **educational and research purposes only**. It uses publicly available datasets from Roboflow and is provided "as is" without any warranties or guarantees of accuracy. Users are responsible for ensuring compliance with all applicable laws, regulations, and privacy requirements when processing images/videos. The developers are not liable for any misuse or damages arising from the use of this system. **Important**: Always verify detection results manually in critical safety applications - this system should not be the sole method for safety compliance monitoring.

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

**Developed by**: [@zeel0014](https://github.com/zeel0014)  
**GitHub Repository**: [Helmet Detection](https://github.com/zeel0014/Helmet_Detection)  
**Live Demo**: [Hugging Face Space](https://huggingface.co/spaces/zeel0014/Multiple_Disease_Prediction)

---

⭐ **If you find this project helpful, please consider giving it a star!**
