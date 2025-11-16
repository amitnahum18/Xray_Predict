X-ray Object Detection with YOLOv8

This repository contains a Colab notebook for detecting objects in X-ray images using a YOLOv8 PyTorch model (best.pt). The notebook demonstrates how to run predictions on individual images or a dataset and visualize the results.

Repository Structure
xray_dataset/
│
├── train/
│   ├── images/       # Training images
│   └── labels/       # YOLO-format labels
│
├── valid/
│   ├── images/       # Validation/test images
│   └── labels/       # YOLO-format labels
│
├── data.yaml         # Dataset configuration
└── best.pt           # Trained YOLOv8 model

Dataset

The dataset must follow the YOLO folder structure:

images/ contains all X-ray images (.jpg, .png, etc.).

labels/ contains text files in YOLO format:

<class_id> <x_center> <y_center> <width> <height>


data.yaml defines the dataset paths and class names:

train: xray_dataset/train/images
val: xray_dataset/valid/images

nc: 5
names: ['class0', 'class1', 'class2', 'class3', 'class4']

Model

best.pt is a trained YOLOv8 PyTorch model.

The model predicts bounding boxes and class labels for each object in the images.

Colab Notebook Usage

Load the YOLOv8 model:

from ultralytics import YOLO
model = YOLO('/content/best.pt')


Predict on a single image:

img_path = '/content/xray_dataset/valid/images/sample.jpg'
results = model.predict(img_path, imgsz=640, conf=0.25)

# Display predictions
annotated_img = results[0].plot()
import matplotlib.pyplot as plt
plt.imshow(annotated_img)
plt.axis('off')
plt.show()


Predict on multiple images:

import os
image_folder = '/content/xray_dataset/valid/images'
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]

for img_name in image_files:
    img_path = os.path.join(image_folder, img_name)
    results = model.predict(img_path, imgsz=640, conf=0.25)
    annotated_img = results[0].plot()
    plt.imshow(annotated_img)
    plt.axis('off')
    plt.show()

Predictions and Visualizations

The notebook visualizes predicted bounding boxes directly without saving images.

Both training and validation datasets can be evaluated to provide precision, recall, and mAP metrics for each class.

Example Images

Example images are included in xray_dataset/valid/images/.

They can be used to test the model or visualize predictions.

Notes

Ensure data.yaml paths match your dataset folders.

Keep images and labels consistent to avoid errors.

This notebook relies on the Ultralytics YOLOv8 library for detection and visualization.
