# YOLOv8 Road Sign Detection

## Model Performance
- **Overall mAP50**: 94.3%
- **Classes detected**: speedlimit, stop, trafficlight, crosswalk
- **Model size**: ~6.2 MB
- **Training time**: ~8 minutes
- 
## Dataset

- **Source**: Kaggle road sign detection dataset
- **Total images**: 877
- **Train/Val/Test split**: 613/175/89 images
- **Original format**: Pascal VOC XML
- **Converted to**: YOLO format

## Training Summary

The model was fine-tuned using transfer learning from YOLOv8n:

1. Downloaded road sign dataset from Kaggle
2. Converted Pascal VOC annotations to YOLO format
3. Split dataset using scikit-learn
4. Trained for 50 epochs with early stopping
5. Achieved 94.3% mAP50 on test set

For detailed training process, see `TRAINING_GUIDE.md`.

## Usage Examples

### Single Image Prediction
```python
from ultralytics import YOLO

model = YOLO("models/road_sign_detector_final.pt")
results = model("road_image.jpg")
results[0].show()
```

### Batch Prediction
```python
# See model_clean.ipynb for batch prediction function
predict_multiple_images("path/to/images", max_images=5)
```

### Model Evaluation
```python
# Evaluate on test set
results = model.val(data="config.yaml", split='test')
print(f"mAP50: {results.box.map50:.3f}")
```

## Installation

1. Clone this repository
2. Install requirements: `pip install -r requirements.txt`
3. Run the notebook: `jupyter notebook model_clean.ipynb`
