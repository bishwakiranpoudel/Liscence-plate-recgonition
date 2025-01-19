# License Plate Recognition System

## Introduction

This project is a **License Plate Recognition System** designed to identify and extract license plate information using a combination of deep learning and image processing techniques. It comprises three main components:

1. **YOLO (You Only Look Once)**: Used for detecting the license plate regions in images or videos.
2. **OpenCV with Contour Detection**: Utilized to segment and extract individual characters from the detected license plates.
3. **CNN (Convolutional Neural Network)**: A custom-trained model for recognizing Nepali characters on license plates.

The system is built for detecting and recognizing Nepali license plates, making it specifically tailored for vehicles in Nepal.

## Dataset

### License Plate Region Detection

- Dataset: Vehicle Number Plate Dataset (Nepal)  
  Kaggle Link: [Vehicle Number Plate Dataset](https://www.kaggle.com/datasets/ishworsubedii/vehicle-number-plate-datasetnepal)

### Character Recognition

- Dataset: Nepali Number Plate Characters Dataset  
  Kaggle Link: [Nepali Number Plate Characters Dataset](https://www.kaggle.com/datasets/inspiring-lab/nepali-number-plate-characters-dataset)

Both datasets are preprocessed and split into training and testing sets using an 80/20 ratio.

## Features

1. **End-to-End Pipeline**:

   - Detect license plates in images/videos.
   - Extract characters from the detected license plates.
   - Recognize characters and compile them into text.

2. **Technologies Used**:

   - **YOLOv8**: For license plate detection.
   - **OpenCV**: For image processing and contour-based character segmentation.
   - **CNN**: For recognizing Nepali characters from segmented images.

3. **Hardware Acceleration**:
   - Supports GPU acceleration via CUDA, ensuring fast training and inference.

## Installation

1. Install the required dependencies for YOLO, OpenCV, and the character recognition model.
2. Download the necessary datasets from Kaggle and preprocess them for training and testing.
3. Train the YOLO model for license plate detection and the CNN for character recognition.
4. Use the trained models to perform inference on images or videos containing Nepali license plates.

## Usage

1. **Detect License Plates**: The YOLO model detects license plates in images/videos.
2. **Extract Characters**: OpenCV extracts individual characters from the license plates using contours.
3. **Recognize Characters**: The CNN predicts Nepali characters from segmented images.
