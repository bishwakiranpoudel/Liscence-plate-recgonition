# License Plate Recognition System

## Introduction

This project is a **License Plate Recognition System** designed to identify and extract license plate information using deep learning and image processing techniques. It comprises three main components:

1. **YOLO (You Only Look Once)**: Used for detecting the license plate regions in images or videos.
2. **YOLO for Character Recognition**: A second YOLO model trained specifically to detect and recognize characters within the detected license plates.
3. **Inference Application**: Combines both YOLO models to seamlessly process video input and output the recognized license plate text.

This system is tailored for recognizing Nepali license plates, making it particularly useful for vehicles in Nepal.

## Dataset

### License Plate Region Detection

- **Dataset**: Vehicle Number Plate Dataset (Nepal)  
  Kaggle Link: [Vehicle Number Plate Dataset](https://www.kaggle.com/datasets/ishworsubedii/vehicle-number-plate-datasetnepal)

### Character Recognition

- **Dataset for CNN**: Nepali Number Plate Characters Dataset  
  Kaggle Link: [Nepali Number Plate Characters Dataset](https://www.kaggle.com/datasets/inspiring-lab/nepali-number-plate-characters-dataset)

- **YOLO Dataset**: Character detection dataset obtained via Roboflow API.  
  Code used for downloading:
  ```python
  from roboflow import Roboflow
  rf = Roboflow(api_key="nyTt2Pl7hWZb46DMUKEw")
  project = rf.workspace("monkey-det-eje1y").project("characterdet")
  dataset = project.version(2).download("yolov8")
  ```

Both datasets are preprocessed and split into training and testing sets using an 80/20 ratio.

## Features

1. **End-to-End Pipeline**:

   - Detect license plates in images/videos.
   - Extract and recognize characters from the detected license plates.
   - Compile recognized characters into text.

2. **Technologies Used**:

   - **YOLOv8**: For license plate detection.
   - **YOLOv8**: For character recognition and segmentation.
   - **Flask**: For building an inference application.

3. **Hardware Acceleration**:
   - Supports GPU acceleration via CUDA, ensuring fast training and inference.

## System Evolution

### Initial Approach

Initially, the system used:

1. **YOLO for License Plate Detection**: Successfully detected license plates from input images/videos.
2. **CNN-Based Character Recognition**: Leveraged a pretrained ResNet model for recognizing Nepali characters. Characters were segmented using OpenCV techniques such as grayscale conversion, thresholding, morphological operations, and contour detection. However, this approach yielded poor performance due to inconsistent character segmentation and recognition errors.

### Improved Approach

To address these challenges:

1. **YOLO for Character Recognition**: A second YOLO model was trained specifically to recognize and segment characters within license plates. This approach eliminated the need for traditional image processing methods and significantly improved accuracy and robustness.
2. **Integrated Application**: An end-to-end inference application was developed that combines the two YOLO models (license plate detection and character recognition). The application processes videos and outputs annotated results with recognized license plate text.

## Installation

1. Install dependencies:
   ```bash
   pip install ultralytics opencv-python flask roboflow
   ```
2. Download the required datasets from Kaggle and Roboflow and preprocess them for training.
3. Train the YOLO models for:
   - License plate detection.
   - Character recognition and segmentation.
4. Save the trained models in the appropriate paths (e.g., `./best.pt` and `./runs/detect/train/weights/best.pt`).

## Usage

1. **Inference Application**:

   - Input: A video file containing Nepali license plates.
   - Output: A processed video with detected license plates and recognized characters annotated on each frame.

2. **Steps**:

   - Run the application code (as provided in the `app.py` file).
   - Upload a video through the web interface or provide a direct file path for processing.
   - The processed video will be saved with recognized license plate annotations.

3. **Demo**:
   - Input: `test.MOV`
   - Output: `result.mp4`

---

For further details on the implementation, refer to the source code and documentation.
