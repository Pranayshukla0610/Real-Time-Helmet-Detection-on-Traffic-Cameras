# Real-Time-Helmet-Detection-on-Traffic-Cameras
An end-to-end deep learning project for detecting helmet and no-helmet riders from images and traffic videos. Includes data preprocessing, model training, real-time inference, and video analytics using OpenCV.


🪖 Helmet Detection System using YOLOv8
📌 Project Overview

This project implements a CNN-based Helmet Detection System using YOLOv8 to detect riders wearing helmets and those without helmets in images and traffic videos. The system supports real-time inference and is suitable for intelligent traffic monitoring and road safety applications.

🎯 Key Features

CNN-based object detection using YOLOv8

Helmet vs No-Helmet classification

Image, video, and real-time detection

Trained on public helmet datasets

Fully implemented using free & open-source tools

Google Colab compatible

🛠️ Tech Stack

Python

YOLOv8 (Ultralytics)

PyTorch

OpenCV

NumPy

Google Colab

📂 Project Structure
Helmet-Detection-YOLOv8/
│
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│
├── helmet.yaml
├── train.py
├── inference_image.py
├── inference_video.py
├── runs/
│   └── detect/
│
├── requirements.txt
└── README.md

📥 Dataset

Public helmet detection datasets from Kaggle

Annotations converted to YOLO format

Classes:

0: Helmet

1: No Helmet

🚀 Model Training
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="helmet.yaml",
    epochs=30,
    imgsz=640,
    batch=16,
    device="cpu"
)

🖼️ Image Inference
model.predict(
    source="test_images/",
    conf=0.25,
    save=True
)

🎥 Video Inference
model.predict(
    source="traffic.mp4",
    conf=0.25,
    save=True
)


Output video is saved as:

runs/detect/predict/traffic.avi

🧠 Real-World Applications

Traffic rule enforcement

Road safety monitoring

Smart city surveillance

Accident prevention systems

📈 Results

High accuracy helmet detection

Real-time performance on CPU

Robust detection in traffic videos

🧪 Future Enhancements

Helmet violation counting

Alert system for no-helmet riders

Deployment using FastAPI

Live CCTV integration

👨‍💻 Author

Pranay Shukla
Data Analyst | Data Science | Deep Learning | Computer Vision
