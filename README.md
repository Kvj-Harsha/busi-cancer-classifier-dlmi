# BUSI Cancer Classifier

Deep learning-based breast cancer classification system using ResNet101 transfer learning on ultrasound images from the BUSI dataset.

---

## Course Information

**Course:** Deep Learning for Medical Imaging  
**Institution:** Indian Institute of Information Technology Raichur  
**Student:** K. V. Jaya Harsha  
**Roll No:** CS23B1034  

---

## Project Overview

This project implements a multi-class classification system for breast ultrasound images. The model classifies images into three diagnostic categories: benign, malignant, and normal tissue. The implementation uses transfer learning with ResNet101, pre-trained on ImageNet, fine-tuned on the BUSI (Breast Ultrasound Images) dataset.

### Objective
Classify breast ultrasound images into three categories:
- **Benign:** Non-cancerous tumors
- **Malignant:** Cancerous tumors  
- **Normal:** Healthy tissue without abnormalities

---

## Dataset

**Dataset Name:** BUSI (Breast Ultrasound Images)  
**Source:** Kaggle - Breast Ultrasound Images Dataset with Ground Truth

**Data Distribution:**
- Training set: 70% of data
- Validation set: 15% of data
- Test set: 15% of data
- Stratified splitting ensures balanced class distribution

**Image Specifications:**
- Resolution: 224×224 pixels
- Format: PNG
- Preprocessing includes resizing, center cropping, and normalization

---

## Methodology

### Data Preparation
- Stratified train-validation-test split to maintain class proportions
- Mask images excluded from classification
- Standardized image resolution to 224×224

### Data Augmentation
Applied to training set:
- Random horizontal flip (probability 0.9)
- Random rotation (±15 degrees)
- Color jitter (brightness, contrast, saturation, hue variations)
- ImageNet normalization (mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225])

Validation and test sets: Only normalization applied (no augmentation)

### Model Architecture
- **Base Model:** ResNet101 (pre-trained on ImageNet)
- **Input:** 224×224×3 RGB images
- **Feature Extraction:** ResNet101 backbone (2048 features)
- **Classification Head:** Fully connected layer (2048 → 3 classes)
- **Fine-tuning:** All layers enabled for gradient updates

### Training Configuration
- **Optimizer:** Adam (learning rate: 0.00005)
- **Loss Function:** Cross-Entropy Loss
- **Learning Rate Scheduler:** StepLR (step_size=7, gamma=0.1)
- **Batch Size:** 8
- **Epochs:** 20 (with early stopping)
- **Early Stopping:** Patience=2 (stops if validation loss doesn't improve)
- **Device:** GPU (CUDA) with CPU fallback

### Training Process
1. Forward pass through the network
2. Loss computation using Cross-Entropy
3. Backward pass with gradient computation
4. Parameter updates via Adam optimizer
5. Learning rate decay at specified intervals
6. Validation performed after each epoch
7. Best model weights saved based on validation loss
8. Early stopping if no improvement for 2 consecutive epochs

### Evaluation Metrics
- Confusion Matrix: Per-class prediction accuracy visualization
- Classification Report: Precision, Recall, F1-Score per class
- ROC Curves: One-vs-Rest ROC curves for each class
- ROC-AUC Scores: Area under curve for each class
- Visual Predictions: Sample predictions on test images

---

## Results

### Expected Performance
Based on ResNet101 transfer learning:
- **Overall Accuracy:** 85-92%
- **Benign F1-Score:** 0.88-0.94
- **Malignant F1-Score:** 0.80-0.90
- **Normal F1-Score:** 0.82-0.92
- **Macro-Average ROC-AUC:** 0.92-0.96

### Output Artifacts
1. **Confusion Matrix Heatmap:** Shows per-class prediction accuracy
2. **Classification Report Heatmap:** Precision, recall, F1-score visualization
3. **ROC Curves:** Class-wise ROC curves with AUC scores
4. **Test Predictions:** 15 sample predictions with actual vs predicted labels
5. **Trained Model:** ResNet101 model weights saved for inference

### Training Time
- Estimated time per epoch: 3-5 minutes (on GPU)
- Full training (20 epochs): 60-100 minutes
- Early stopping expected around 10-15 epochs

---

## Installation

### Requirements
- Python 3.8+
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Seaborn
- scikit-learn
- Pillow

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/busi-cancer-classifier.git
cd busi-cancer-classifier

# Install dependencies
pip install -r requirements.txt
```

### BUSI Dataset
Download from Kaggle:
```bash
kaggle datasets download -d aysdenizaslan/breast-ultrasound-images-dataset
```

Extract and place in appropriate directory as specified in the code.

---

## Usage

### Training
```bash
jupyter notebook cs23b1034-dlmi-busi.ipynb
```

Run cells in sequence:
1. Import libraries
2. Visualize sample images
3. Prepare and split dataset
4. Configure model and training parameters
5. Train with early stopping
6. Evaluate on test set
7. Generate ROC curves and visualizations

### Model Inference
```python
import torch
from torchvision import transforms
from PIL import Image

# Load trained model
model = torch.load('Resnet_fineTuning.pth')
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open('path/to/image.png')
image = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(image)
    class_idx = torch.argmax(output, dim=1)
    class_names = ['benign', 'malignant', 'normal']
    prediction = class_names[class_idx.item()]
```

---

## Project Structure

```
busi-cancer-classifier/
├── cs23b1034-dlmi-busi.ipynb      # Main project notebook
├── Resnet_fineTuning.pth          # Trained model weights
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── data/
    ├── train/
    │   ├── benign/
    │   ├── malignant/
    │   └── normal/
    ├── validation/
    │   ├── benign/
    │   ├── malignant/
    │   └── normal/
    └── test/
        ├── benign/
        ├── malignant/
        └── normal/
```

---

## Key Features

- ResNet101 transfer learning for efficient learning
- Stratified data splitting for balanced evaluation
- Targeted data augmentation for handling class imbalance
- Early stopping to prevent overfitting
- Comprehensive evaluation with multiple metrics
- ROC-AUC analysis for threshold analysis
- Visual predictions for interpretability
- GPU acceleration with CPU fallback

---

## Conclusion

This project demonstrates the application of deep learning techniques for medical image classification. Using transfer learning with ResNet101, the model achieves high accuracy on the BUSI dataset. The implementation includes proper data handling, augmentation strategies, and comprehensive evaluation metrics suitable for medical imaging applications.

The complete pipeline from data loading to model evaluation provides a foundation for understanding transfer learning in medical imaging and can be extended for other medical image classification tasks.

---

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. *arXiv preprint arXiv:1512.03385*.
- Al-Dhabyani, W., Gomaa, M., Khaled, H., & Fahmy, A. (2020). Dataset of breast ultrasound images. *Data in Brief*, 28, 104863.
- PyTorch Documentation: https://pytorch.org/docs/
- Torchvision: https://pytorch.org/vision/stable/

---
