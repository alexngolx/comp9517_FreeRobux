# COMP9517 Pest Detection Project

This repository contains implementations of multiple machine learning approaches for detecting and classifying agricultural pests (insects) in images. The project compares traditional computer vision methods (HOG-SVM, LBP-RF) with deep learning approaches (RetinaNet, YOLOv5) on the AgroPest-12 dataset.

## Dataset

The project uses the **AgroPest-12** dataset, which contains 12 insect classes:
- Ants
- Bees
- Beetles
- Caterpillars
- Earthworms
- Earwigs
- Grasshoppers
- Moths
- Slugs
- Snails
- Wasps
- Weevils

The dataset is organized in YOLO format with:
- Images in `data/{split}/images/`
- Labels in `data/{split}/labels/` (normalized bounding box coordinates)
- Configuration file `data/data.yaml` containing class names and dataset metadata

## Project Structure

```
comp9517_FreeRobux/
├── Exploratory_Data_Analysis/
│   ├── exploratory data analysis.ipynb    # Initial dataset exploration
│   └── data quality check.ipynb           # Data quality validation and quarantine
├── SVM_HOG/
│   └── 9517_SVM_HOG.ipynb                 # HOG feature extraction + SVM classifier
├── LBP_RF/
│   └── lbp_rf_train.ipynb                  # LBP feature extraction + Random Forest
├── RetinaNet/
│   ├── Retinanet_train.ipynb              # RetinaNet model training
│   ├── Retinanet_0_3thres.ipynb          # Evaluation with threshold 0.3
│   ├── Retinanet_0_5thres.ipynb          # Evaluation with threshold 0.5
│   ├── Retinanet_0_7thres.ipynb          # Evaluation with threshold 0.7
│   └── Retinanet50.png                    # Confusion matrix visualization
└── YOLOv5/
    └── yolov5 analysis.ipynb              # YOLOv5 training, validation, and analysis
```

## Methods Implemented

### 1. HOG-SVM (Histogram of Oriented Gradients + Support Vector Machine)

**Location:** `SVM_HOG/9517_SVM_HOG.ipynb`

A traditional computer vision approach using:
- **Feature Extraction:** HOG (Histogram of Oriented Gradients) with parameters:
  - Orientations: 9
  - Pixels per cell: (8, 8)
  - Cells per block: (2, 2)
  - Block normalization: L2-Hys
- **Preprocessing:** Image resizing (128x128), grayscale conversion, Gaussian blur, CLAHE contrast enhancement
- **Classifier:** Support Vector Machine (SVM)
- **Features:**
  - ROI extraction from YOLO bounding boxes
  - Feature scaling with StandardScaler
  - Top-3 class probability predictions

**Key Functions:**
- `extract_roi()`: Converts YOLO coordinates to pixel coordinates and extracts ROI
- `preprocess_image()`: Applies preprocessing pipeline
- `extract_hog_features()`: Batch HOG feature extraction
- `train()`: Trains SVM classifier
- `predict()`: Performs inference with probability scores

### 2. LBP-RF (Local Binary Patterns + Random Forest)

**Location:** `LBP_RF/lbp_rf_train.ipynb`

A texture-based classification approach using:
- **Feature Extraction:** Local Binary Patterns (LBP) with parameters:
  - Radius: 1
  - Number of points: 8
  - Method: uniform
- **Classifier:** Random Forest with:
  - 200 estimators
  - Max depth: 30
  - Balanced class weights
- **Features:**
  - Automatic ROI extraction from YOLO labels
  - LBP histogram feature vectors
  - Model persistence with pickle

**Key Functions:**
- `extract_lbp_features()`: Extracts LBP histogram from single image
- `extract_lbp_features_batch()`: Batch LBP feature extraction
- `train_lbp_rf()`: Complete training pipeline
- `save_model_pickle()`: Saves trained model and parameters

### 3. RetinaNet

**Location:** `RetinaNet/Retinanet_train.ipynb` (training), `Retinanet_*_thres.ipynb` (evaluation)

A deep learning object detection model:
- **Architecture:** RetinaNet with ResNet-50 backbone
- **Training:** Custom PyTorch dataset with YOLO-to-RetinaNet label conversion
- **Evaluation:** Comprehensive metrics with multiple threshold combinations:
  - Score thresholds: 0.3, 0.5, 0.7
  - IoU thresholds: 0.3, 0.5, 0.7
- **Metrics:** mAP, Precision, Recall, Accuracy, F1-score, Confusion Matrix

**Key Components:**
- `yolo_converter()`: Converts YOLO format to RetinaNet format (x_min, y_min, x_max, y_max)
- `ImgDataset`: PyTorch Dataset class for loading images and annotations
- Evaluation notebooks compute metrics using `torchmetrics.detection.mean_ap.MeanAveragePrecision`

### 4. YOLOv5

**Location:** `YOLOv5/yolov5 analysis.ipynb`

State-of-the-art object detection using YOLOv5:
- **Model:** YOLOv5s (small variant)
- **Training Configuration:**
  - Image size: 640x640
  - Batch size: 8
  - Epochs: 20
  - Optimizer: Adam
  - Label smoothing: 0.1
  - Early stopping patience: 5
- **Analysis Features:**
  - Latency measurement (inference time per image)
  - FPS calculation
  - Grad-CAM visualization for model interpretability
  - Validation and test set evaluation

**Training Command:**
```bash
python train.py --img 640 --batch 8 --epochs 20 \
  --data "path/to/data.yaml" --weights yolov5s.pt \
  --device 0 --optimizer Adam --label-smoothing 0.1 \
  --patience 5 --project runs/train --name yolov5s_safe_2
```

## Data Preprocessing

### Exploratory Data Analysis

**Location:** `Exploratory_Data_Analysis/exploratory data analysis.ipynb`

Performs initial dataset exploration to understand:
- Class distribution
- Image statistics
- Bounding box size distributions
- Dataset characteristics

### Data Quality Check

**Location:** `Exploratory_Data_Analysis/data quality check.ipynb`

Identifies and quarantines problematic samples:
- **Missing labels:** Images without corresponding label files
- **Bad label format:** Invalid YOLO coordinate formats
- **Tiny bounding boxes:** Boxes smaller than minimum threshold (default: 0.01 relative size)
- **Class mismatches:** Labels that don't match expected class from filename

**Quarantine Process:**
- Problematic files are moved to `../quarantine/` directory
- Both image and label files are quarantined together
- Preserves dataset integrity for training

**Key Functions:**
- `get_expected_class_from_filename()`: Extracts expected class from filename prefix
- `load_label_classes()`: Loads class IDs from YOLO label file
- `check_label_file()`: Validates label file format and coordinates
- `validate_expected_class()`: Ensures label matches filename-derived class

## Requirements

### Python Packages

```python
# Core libraries
numpy
opencv-python (cv2)
scikit-image
scikit-learn
matplotlib
seaborn
pyyaml

# Deep learning
torch
torchvision
torchmetrics

# YOLOv5 (requires separate repository)
# Clone from: https://github.com/ultralytics/yolov5
```

### System Requirements

- Python 3.7+
- CUDA-capable GPU (recommended for RetinaNet and YOLOv5)
- Sufficient RAM for batch processing (8GB+ recommended)

## How to Run

### 1. Setup Environment

```bash
# Install required packages
pip install numpy opencv-python scikit-image scikit-learn matplotlib seaborn pyyaml torch torchvision torchmetrics

# For YOLOv5, clone the repository
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

### 2. Prepare Dataset

Ensure your dataset is organized as:
```
data/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

### 3. Run Data Quality Check (Recommended)

```bash
# Open and run: Exploratory_Data_Analysis/data quality check.ipynb
# This will quarantine problematic samples before training
```

### 4. Train Models

#### HOG-SVM
```bash
# Open and run: SVM_HOG/9517_SVM_HOG.ipynb
# Execute all cells to train and evaluate the model
```

#### LBP-RF
```bash
# Open and run: LBP_RF/lbp_rf_train.ipynb
# Execute all cells to train the Random Forest classifier
```

#### RetinaNet
```bash
# Open and run: RetinaNet/Retinanet_train.ipynb
# After training, evaluate with different thresholds:
# - Retinanet_0_3thres.ipynb
# - Retinanet_0_5thres.ipynb
# - Retinanet_0_7thres.ipynb
```

#### YOLOv5
```bash
# Navigate to YOLOv5 directory
cd yolov5

# Train model
python train.py --img 640 --batch 8 --epochs 20 \
  --data "path/to/data.yaml" --weights yolov5s.pt \
  --device 0 --optimizer Adam --label-smoothing 0.1 \
  --patience 5 --project runs/train --name yolov5s_safe_2

# Validate model
python val.py --weights runs/train/yolov5s_safe_2/weights/best.pt \
  --data "path/to/data.yaml" --img 640 --batch 1 --save-json

# For analysis and Grad-CAM: Open YOLOv5/yolov5 analysis.ipynb
```

## Model Evaluation

### Metrics Used

- **mAP (Mean Average Precision):** Primary metric for object detection
- **Precision:** Ratio of true positives to all predicted positives
- **Recall:** Ratio of true positives to all actual positives
- **Accuracy:** Overall classification accuracy
- **F1-Score:** Harmonic mean of precision and recall
- **Confusion Matrix:** Per-class classification performance

### Threshold Tuning

For object detection models (RetinaNet, YOLOv5), performance varies with:
- **Score threshold:** Minimum confidence for detections (0.3, 0.5, 0.7)
- **IoU threshold:** Intersection over Union for matching predictions to ground truth (0.3, 0.5, 0.7)

The RetinaNet evaluation notebooks systematically test different threshold combinations.

## Model Outputs

### Saved Models

- **HOG-SVM:** Trained model and scaler saved as pickle files
- **LBP-RF:** Model saved to `LBP_RF/models/classifier.pkl`
- **RetinaNet:** Model weights saved as `.pth` files
- **YOLOv5:** Model weights saved in `runs/train/{name}/weights/best.pt`

### Visualizations

- Confusion matrices (e.g., `RetinaNet/Retinanet50.png`)
- Grad-CAM heatmaps (YOLOv5 analysis notebook)
- Training curves and metrics

## Notes

- All notebooks assume the dataset is located in a `data/` directory relative to the project root
- Path configurations may need adjustment based on your local setup
- GPU acceleration is highly recommended for deep learning models (RetinaNet, YOLOv5)
- The data quality check notebook should be run before training to ensure clean data

## License

This project is part of COMP9517 coursework at UNSW.

