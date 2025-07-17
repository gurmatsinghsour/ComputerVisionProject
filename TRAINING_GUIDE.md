# YOLOv8 Road Sign Detection - Training Process

This document explains how the YOLOv8 model was fine-tuned for road sign detection.

## Overview

Fine-tuning is the process of taking a pre-trained model and adapting it to a specific task. Instead of training from scratch, we start with YOLOv8's pre-trained weights and modify the final layers to detect our specific classes.

## Dataset Information

- **Source**: Kaggle road sign detection dataset
- **Total Images**: 877 road sign images
- **Classes**: 4 types (speedlimit, stop, trafficlight, crosswalk)
- **Original Format**: Pascal VOC XML annotations
- **Target Format**: YOLO format (normalized coordinates)

## Training Process

### Step 1: Dataset Preparation

```python
import kagglehub
import xml.etree.ElementTree as ET
from collections import Counter
import os

# Download dataset from Kaggle
dataset_path = kagglehub.dataset_download("andrewmvd/road-sign-detection")

# Analyze classes in the dataset
classes = []
annotations_dir = os.path.join(dataset_path, "annotations")

for xml_file in os.listdir(annotations_dir):
    if xml_file.endswith('.xml'):
        tree = ET.parse(os.path.join(annotations_dir, xml_file))
        root = tree.getroot()
        for obj in root.findall('object'):
            class_name = obj.find('name') or obj.find('n')
            if class_name is not None:
                classes.append(class_name.text.strip())

class_counts = Counter(classes)
print(f"Found {len(class_counts)} classes:")
for class_name, count in class_counts.items():
    print(f"  {class_name}: {count} instances")

# Result: Found 4 classes with 1244 total annotations
# speedlimit: 783 instances (62.9%)
# crosswalk: 200 instances (16.1%)  
# trafficlight: 170 instances (13.7%)
# stop: 91 instances (7.3%)
```

### Step 2: Convert Annotations

Convert Pascal VOC XML format to YOLO format:

```python
import os

class_names = ['speedlimit', 'stop', 'trafficlight', 'crosswalk']

# Create labels directory
labels_dir = os.path.join(dataset_path, "labels")
os.makedirs(labels_dir, exist_ok=True)

# Convert each XML file
for xml_file in os.listdir(annotations_dir):
    if xml_file.endswith('.xml'):
        tree = ET.parse(os.path.join(annotations_dir, xml_file))
        root = tree.getroot()
        
        # Get image dimensions
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        
        yolo_annotations = []
        
        for obj in root.findall('object'):
            class_name = obj.find('name') or obj.find('n')
            if class_name is not None:
                class_id = class_names.index(class_name.text.strip())
                
                # Get bounding box
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                
                # Convert to YOLO format (normalized coordinates)
                x_center = (xmin + xmax) / 2.0 / width
                y_center = (ymin + ymax) / 2.0 / height
                bbox_width = (xmax - xmin) / width
                bbox_height = (ymax - ymin) / height
                
                yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"
                yolo_annotations.append(yolo_line)
        
        # Save YOLO format file
        txt_filename = xml_file.replace('.xml', '.txt')
        txt_path = os.path.join(labels_dir, txt_filename)
        with open(txt_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))

print("Conversion completed: 877 XML files converted to YOLO format")
```

### Step 3: Dataset Splitting

Split the dataset into train/validation/test sets using scikit-learn:

```python
from sklearn.model_selection import train_test_split
import shutil

# Get all image files
images_dir = os.path.join(dataset_path, "images")
image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Split dataset: 70% train, 20% val, 10% test
train_files, temp_files = train_test_split(image_files, test_size=0.3, random_state=42)
val_files, test_files = train_test_split(temp_files, test_size=0.33, random_state=42)

print(f"Dataset split:")
print(f"  Training: {len(train_files)} images")
print(f"  Validation: {len(val_files)} images")
print(f"  Test: {len(test_files)} images")

# Create YOLO dataset structure
output_dir = "yolo_dataset"
sets = {'train': train_files, 'val': val_files, 'test': test_files}

for set_name, file_list in sets.items():
    # Create directories
    set_images_dir = os.path.join(output_dir, 'images', set_name)
    set_labels_dir = os.path.join(output_dir, 'labels', set_name)
    os.makedirs(set_images_dir, exist_ok=True)
    os.makedirs(set_labels_dir, exist_ok=True)
    
    # Copy files
    for image_file in file_list:
        # Copy image
        src_image = os.path.join(images_dir, image_file)
        dst_image = os.path.join(set_images_dir, image_file)
        shutil.copy2(src_image, dst_image)
        
        # Copy corresponding label
        label_file = image_file.replace('.png', '.txt').replace('.jpg', '.txt')
        src_label = os.path.join(labels_dir, label_file)
        dst_label = os.path.join(set_labels_dir, label_file)
        if os.path.exists(src_label):
            shutil.copy2(src_label, dst_label)

# Result: 613 train (69.9%), 175 val (20.0%), 89 test (10.1%)
```

### Step 4: Create Configuration File

Create YOLO dataset configuration:

```python
import yaml

config = {
    'path': 'yolo_dataset',
    'train': 'images/train',
    'val': 'images/val', 
    'test': 'images/test',
    'nc': 4,
    'names': ['speedlimit', 'stop', 'trafficlight', 'crosswalk']
}

with open('yolo_dataset/config.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
```

### Step 5: Training

Train the model using YOLOv8's built-in training:

```python
from ultralytics import YOLO

# Load pre-trained YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Train the model
results = model.train(
    data='yolo_dataset/config.yaml',  # Path to dataset config
    epochs=50,                        # Number of training epochs
    imgsz=640,                       # Image size for training
    batch=16,                        # Batch size
    patience=10,                     # Early stopping patience
    lr0=0.01,                       # Initial learning rate
    project='road_sign_detection',   # Project name
    name='yolov8_road_signs'        # Experiment name
)
```

## Training Results

The model achieved excellent performance:

- **Training Duration**: 0.127 hours (7.6 minutes)
- **Overall mAP50**: 94.3%
- **Per-class Performance**:
  - Speed limit signs: 99.0% mAP50
  - Stop signs: 99.5% mAP50
  - Traffic lights: 81.4% mAP50
  - Crosswalk signs: 97.4% mAP50

## Key Concepts Explained

**Transfer Learning**: Instead of training from scratch, we use a model pre-trained on COCO dataset. The lower layers already understand basic features like edges and shapes.

**mAP50**: Mean Average Precision at IoU threshold 0.5. It measures how accurately the model detects and localizes objects.

**Batch Size**: Number of images processed together. Larger batches can speed up training but require more memory.

**Learning Rate**: Controls how fast the model learns. Too high can cause instability, too low makes training slow.

**Early Stopping**: Stops training if validation performance doesn't improve for a set number of epochs, preventing overfitting.

## Usage

After training, the model can be used for inference:

```python
from ultralytics import YOLO

# Load trained model
model = YOLO("road_sign_detector_final.pt")

# Make predictions
results = model("road_image.jpg")
results[0].show()

# Batch processing
import os
image_directory = "test_images"
for image_file in os.listdir(image_directory):
    if image_file.endswith(('.jpg', '.png')):
        results = model(os.path.join(image_directory, image_file))
        results[0].save(f"result_{image_file}")
```

The final model file (road_sign_detector_final.pt) contains all the learned weights and can be deployed for real-time road sign detection.
