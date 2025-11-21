# COMP9517 Pest Detection Project

This repository presents multiple approaches for detecting and classifying agricultural pests (insects) in images. We implemented and compared traditional computer vision methods, specifically Histogram of Oriented Gradients with Support Vector Machine (HOG-SVM) and Local Binary Patterns with Random Forest (LBP-RF), alongside deep learning approaches including RetinaNet and YOLOv5 on the AgroPest-12 dataset.

## Dataset

The project utilizes the **AgroPest-12** dataset, which contains 12 different insect classes:
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

The dataset is organized in YOLO format with the following structure:
- Images located in `data/{split}/images/`
- Labels in `data/{split}/labels/` (normalized bounding box coordinates)
- Configuration file `data/data.yaml` containing class names and dataset metadata

## Project Structure

The repository is organized as follows:

```
comp9517_FreeRobux/
├── Exploratory_Data_Analysis/
│   ├── exploratory data analysis.ipynb    # Initial dataset exploration
│   └── data quality check.ipynb           # Data quality validation and quarantine
├── SVM_HOG/
│   └── 9517_SVM_HOG.ipynb                 # HOG feature extraction + SVM classifier
├── LBP_RF/
│   ├── lbp_rf_train.ipynb                  # LBP feature extraction + Random Forest training
│   └── lbp_rf_test.ipynb                   # LBP-RF model evaluation and metrics
├── RetinaNet/
│   ├── Retinanet_train.ipynb              # RetinaNet model training
│   ├── Retinanet_0_3thres.ipynb          # Evaluation with threshold 0.3
│   ├── Retinanet_0_5thres.ipynb          # Evaluation with threshold 0.5
│   ├── Retinanet_0_7thres.ipynb          # Evaluation with threshold 0.7
│   └── Retinanet50.png                    # Confusion matrix visualization
└── YOLOv5/
    └── yolov5 analysis.ipynb              # YOLOv5 training, validation, and analysis
```

## Collaboration Workflow

This project was developed collaboratively using a feature branch workflow with pull requests. The development process followed these steps:

### Branching Strategy

Each major feature or method implementation was developed in its own feature branch:
- `feature/Retinanet` - RetinaNet model implementation
- `feature/YOLOv5` - YOLOv5 model implementation  
- `feature/lbp_rf` - Local Binary Patterns with Random Forest implementation
- `svm-branch` - Histogram of Oriented Gradients with Support Vector Machine implementation

### Development Process

1. **Create Feature Branch:** Each team member created a new branch from `main` for their assigned method:
   ```bash
   git checkout -b feature/method-name
   ```

2. **Development:** Work was conducted independently on each feature branch, allowing parallel development without conflicts.

3. **Pull Request:** Once a feature was complete and tested, a pull request (PR) was created to merge the feature branch into `main`.

4. **Code Review:** Pull requests were reviewed by team members before merging to ensure code quality and consistency.

5. **Merge to Main:** After approval, feature branches were merged into `main` via pull requests, maintaining a clean and organized commit history.

### Merged Features

The following features were successfully merged into the main branch:
- **PR #1:** RetinaNet implementation (`feature/Retinanet` → `main`)
- **PR #2:** YOLOv5 implementation (`feature/YOLOv5` → `main`)
- **PR #4:** HOG-SVM implementation (`svm-branch` → `main`)
- **PR #5:** LBP-RF implementation (`feature/lbp_rf` → `main`)

This workflow enabled parallel development of different methods while maintaining code quality and project organization throughout the development lifecycle.

## Methods Implemented

### 1. Histogram of Oriented Gradients with Support Vector Machine (HOG-SVM)

**Location:** `SVM_HOG/9517_SVM_HOG.ipynb`

This method employs a traditional computer vision approach using HOG features combined with an SVM classifier. The implementation uses:
- **HOG (Histogram of Oriented Gradients) features** with the following parameters:
  - 9 orientations
  - 8×8 pixels per cell
  - 2×2 cells per block
  - L2-Hys block normalization
- **Preprocessing pipeline:** Images are resized to 128×128, converted to grayscale, Gaussian blur is applied, and Contrast Limited Adaptive Histogram Equalization (CLAHE) is used for contrast enhancement
- **SVM classifier** for classification

The notebook extracts regions of interest (ROIs) from YOLO bounding boxes, computes HOG features, and provides top-3 class predictions with probability scores.

Key functions include:
- `extract_roi()` - converts YOLO coordinates to pixel coordinates and crops the region
- `preprocess_image()` - applies the complete preprocessing pipeline
- `extract_hog_features()` - batch processes images to extract HOG features
- `train()` - trains the SVM classifier
- `predict()` - performs inference and returns class probabilities

### 2. Local Binary Patterns with Random Forest (LBP-RF)

**Location:** `LBP_RF/lbp_rf_train.ipynb`

This approach utilizes texture-based features through Local Binary Patterns (LBP) combined with a Random Forest classifier. The method is effective for capturing local texture patterns:
- **LBP parameters:** radius=1, 8 sampling points, uniform method
- **Random Forest configuration:** 200 estimators, maximum depth of 30, balanced class weights to handle imbalanced data

The training notebook automatically extracts ROIs from YOLO labels, computes LBP histogram features, and saves the trained model as a pickle file.

**Evaluation:** `LBP_RF/lbp_rf_test.ipynb`

The evaluation notebook provides comprehensive performance analysis:
- **Detection metrics:** Mean Average Precision (mAP), mAP50, mAP75, mAP50-95 using Intersection over Union (IoU) matching
- **Classification metrics:** Precision, Recall, F1-score, Accuracy, and Area Under the Curve (AUC), computed both per-class and overall
- **Visualizations:** Confusion matrices, Receiver Operating Characteristic (ROC) curves, and per-class metric bar charts
- **Detection approach:** Uses a sliding window method for full image inference

The test notebook provides the most comprehensive evaluation framework, generating extensive metrics and visualizations for model performance analysis.

### 3. RetinaNet

**Location:** `RetinaNet/Retinanet_train.ipynb` (training), `Retinanet_*_thres.ipynb` (evaluation)

RetinaNet is a deep learning object detection model that employs a ResNet-50 backbone. The implementation required converting YOLO format labels (normalized center coordinates) to RetinaNet's expected format (absolute x_min, y_min, x_max, y_max coordinates).

We evaluated the model with different threshold configurations, as detection performance can vary significantly based on these parameters:
- **Score thresholds:** 0.3, 0.5, 0.7 (minimum confidence for detections)
- **IoU thresholds:** 0.3, 0.5, 0.7 (for matching predictions to ground truth)

Each evaluation notebook tests a different score threshold combination. Metrics computed include mAP, precision, recall, accuracy, F1-score, and confusion matrices. The implementation uses `torchmetrics.detection.mean_ap.MeanAveragePrecision` for standardized mAP calculation.

Key components:
- `yolo_converter()` - converts YOLO format annotations to RetinaNet format
- `ImgDataset` - custom PyTorch Dataset class for loading images and annotations

### 4. YOLOv5

**Location:** `YOLOv5/yolov5 analysis.ipynb`

We implemented YOLOv5s (small variant) for object detection, selected for its balance between speed and accuracy given computational constraints. Training configuration:
- Image size: 640×640
- Batch size: 8 (limited by GPU memory)
- Training epochs: 20 with early stopping (patience=5)
- Optimizer: Adam
- Label smoothing: 0.1

The analysis notebook includes:
- **Latency measurements:** Inference time per image
- **FPS calculations:** Frames per second performance
- **Grad-CAM visualizations:** Model interpretability analysis to identify important regions
- **Validation and test evaluations:** Comprehensive model assessment

Training command:
```bash
python train.py --img 640 --batch 8 --epochs 20 \
  --data "path/to/data.yaml" --weights yolov5s.pt \
  --device 0 --optimizer Adam --label-smoothing 0.1 \
  --patience 5 --project runs/train --name yolov5s_safe_2
```

## Data Preprocessing

### Exploratory Data Analysis

**Location:** `Exploratory_Data_Analysis/exploratory data analysis.ipynb`

Initial exploratory data analysis was conducted to understand dataset characteristics, including class distributions, image statistics, bounding box size distributions, and overall dataset properties. This analysis informed subsequent modeling decisions.

### Data Quality Check

**Location:** `Exploratory_Data_Analysis/data quality check.ipynb`

A comprehensive data quality check was performed to identify and address problematic samples. The following issues were identified:
- **Missing labels:** Images without corresponding label files
- **Invalid label formats:** Malformed YOLO coordinate formats
- **Tiny bounding boxes:** Boxes smaller than 1% of image size (threshold: 0.01 relative size), which are problematic for training
- **Class mismatches:** Discrepancies between filename-derived class expectations and actual label classes

The notebook quarantines problematic files by moving them to `../quarantine/` directory. Both image and label files are moved together to maintain data integrity. This preprocessing step is essential before training - approximately 40 problematic samples were identified and quarantined from the training set.

Key functions:
- `get_expected_class_from_filename()` - extracts expected class from filename prefix
- `load_label_classes()` - loads class IDs from YOLO label file
- `check_label_file()` - validates label file format and coordinate validity
- `validate_expected_class()` - ensures label matches filename-derived class expectation

## How to Run

### 1. Environment Setup

```bash
# Install required packages from requirements.txt
pip install -r requirements.txt

# For YOLOv5, clone and setup the repository
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
```

### 2. Dataset Preparation

Ensure your dataset follows this structure:
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

### 3. Data Quality Check (Recommended First Step)

```bash
# Run: Exploratory_Data_Analysis/data quality check.ipynb
# This will identify and quarantine problematic samples before training
```

### 4. Model Training

#### HOG-SVM
```bash
# Open and execute: SVM_HOG/9517_SVM_HOG.ipynb
# Run all cells to complete training and evaluation
```

#### LBP-RF
```bash
# Training: LBP_RF/lbp_rf_train.ipynb
# Execute all cells to train the Random Forest classifier

# Evaluation: LBP_RF/lbp_rf_test.ipynb
# This notebook performs comprehensive evaluation with metrics and visualizations
```

#### RetinaNet
```bash
# Training: RetinaNet/Retinanet_train.ipynb
# Evaluation with different thresholds:
# - Retinanet_0_3thres.ipynb
# - Retinanet_0_5thres.ipynb
# - Retinanet_0_7thres.ipynb
```

#### YOLOv5
```bash
# Navigate to the yolov5 directory
cd yolov5

# Train the model
python train.py --img 640 --batch 8 --epochs 20 \
  --data "path/to/data.yaml" --weights yolov5s.pt \
  --device 0 --optimizer Adam --label-smoothing 0.1 \
  --patience 5 --project runs/train --name yolov5s_safe_2

# Validate the model
python val.py --weights runs/train/yolov5s_safe_2/weights/best.pt \
  --data "path/to/data.yaml" --img 640 --batch 1 --save-json

# For analysis and Grad-CAM: YOLOv5/yolov5 analysis.ipynb
```

## Evaluation Metrics

Multiple evaluation metrics were employed depending on the task:

- **Mean Average Precision (mAP):** Primary metric for object detection performance
- **Precision and Recall:** Standard classification metrics
- **Accuracy:** Overall classification correctness
- **F1-Score:** Harmonic mean of precision and recall
- **Confusion Matrix:** Per-class classification performance analysis

For object detection models (RetinaNet, YOLOv5), threshold tuning significantly impacts performance. We systematically tested:
- **Score thresholds** (0.3, 0.5, 0.7): Minimum confidence required for detections
- **IoU thresholds** (0.3, 0.5, 0.7): Intersection over Union threshold for matching predictions to ground truth

The RetinaNet evaluation notebooks systematically test different threshold combinations to assess performance variations.

## Model Outputs

### Saved Models

Models are saved in the following locations:
- **HOG-SVM:** Pickle files containing the trained model and feature scaler
- **LBP-RF:** `LBP_RF/models/classifier.pkl`
- **RetinaNet:** PyTorch weight files (`.pth` format)
- **YOLOv5:** `runs/train/{name}/weights/best.pt`

### Visualizations

Generated visualizations include:
- **RetinaNet:** Confusion matrix visualization (`RetinaNet/Retinanet50.png`)
- **YOLOv5:** Grad-CAM heatmaps for model interpretability (in analysis notebook)
- **LBP-RF:** Confusion matrices, ROC curves, and per-class metric charts
- Training curves and performance metrics plots throughout the notebooks

## Notes

- All notebooks assume the dataset is located in a `data/` directory relative to the project root. Path configurations may require adjustment based on local setup.
- GPU acceleration is essential for deep learning models (RetinaNet, YOLOv5); CPU training would require days of computation time.
- The data quality check should be executed before training to ensure dataset integrity.
- Some path configurations in notebooks are hardcoded and may need updating for different environments.

## License

This project is part of COMP9517 coursework at UNSW.
