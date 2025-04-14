# 🚗 Car Detection with YOLOv11

## 📂 Dataset

- **Link to Kaggle Dataset**: [Kaggle dataset](https://www.kaggle.com/datasets/sshikamaru/car-object-detection/data)
- The dataset contains:
  - Training and testing images
  - A CSV file for training images with bounding box coordinates: `(x_min, y_min, x_max, y_max)`
- Bounding boxes are used to localize the object (in this case, cars) within each image.

## 🎯 Project Goal

- Train a **YOLOv11** model to detect cars in the test image dataset.

## 🧠 YOLO Algorithm Overview

YOLO (You Only Look Once) is a **supervised learning** algorithm.  
It divides an input image into a grid and uses a convolutional neural network (CNN) to:

- Extract features from the image
- Predict bounding boxes and class probabilities for objects within each grid cell

## 🗂️ YOLOv11 Data Structure Requirements

The model expects the following folder structure:


- `images/` contains the raw image files  
- `labels/` contains `.txt` files for each corresponding image in the format:


## 🛠️ Data Preprocessing Steps

1. Split training data into **80% training** and **20% validation**
2. Create new directories: `images/train`, `images/val`, `labels/train`, `labels/val`
3. For each image:
   - Convert bounding box format from `(x_min, y_min, x_max, y_max)` to `(center_x, center_y, width, height)`
   - Normalize values by dividing by the image’s width and height
   - Write bounding box data into a `.txt` file using the required YOLO format

## 📈 Model Training & Evaluation

- After preprocessing, train the model using a **pre-trained YOLOv11** model
- Evaluate performance on the test dataset

## 📚 Resources Used

- [Ultralytics GitHub Repository](https://github.com/ultralytics/ultralytics)
- [Ultralytics Documentation – Export Formats](https://docs.ultralytics.com/modes/benchmark/#export-formats)

### YouTube Tutorials:
- [YOLOv8 Object Detection – Full Course](https://www.youtube.com/watch?v=wM1wn1bZ3S4)
- [Train YOLO on Custom Dataset](https://www.youtube.com/watch?v=ag3DLKsl2vk&t=278s)
- [How to Annotate & Train YOLOv8](https://www.youtube.com/watch?v=iy34dSwfEsY)

### Additional Tools and Packages Used:
- `pandas` for DataFrame operations
- `cv2` / `PIL` / `matplotlib` for image handling and display
- `os`, `shutil` for directory management and file operations
- Kaggle Notebook with **GPU acceleration** for model training
  - (Originally tried on CPU: took 2+ hours and processed data incorrectly)

- **Link to Kaggle Notebook**: [Link to Notebook](https://www.kaggle.com/code/keithbui/car-object-detection-yolov11)
