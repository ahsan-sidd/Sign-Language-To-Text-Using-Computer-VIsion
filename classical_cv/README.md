# Classical Computer Vision Approach - Documentation

## Overview
This directory contains a classical machine learning approach to ASL (American Sign Language) alphabet recognition using traditional computer vision techniques and scikit-learn classifiers.

## What Has Been Done

### 1. Feature Extraction Pipeline
The implementation uses a multi-modal feature extraction approach combining:

- **HOG (Histogram of Oriented Gradients)**: Captures edge and gradient information from hand gestures
  - Parameters: 9 orientations, 8×8 pixel cells, 2×2 cell blocks
  - Provides shape and structure information

- **Hu Moments**: Global shape descriptors that are translation, scale, and rotation invariant
  - Log-transformed for numerical stability
  - Captures overall hand shape characteristics

- **Color Histograms (HSV)**: Captures color distribution patterns
  - 8×4×4 bins for H-S-V channels
  - Provides coarse color information

### 2. Preprocessing Steps
- Gaussian blur for noise reduction
- Skin tone detection using HSV color space masking
- Histogram equalization for brightness normalization
- Image resizing to 64×64 pixels (configurable)
- Fallback mechanism when skin masking fails

### 3. Classifiers Tested
Three classical ML models were implemented:
- **Random Forest** (200 trees, max depth unrestricted)
- **SVM** (RBF kernel, C=10.0)
- **K-Nearest Neighbors** (k=5, distance-weighted)

### 4. Dataset Configuration
- Dataset: ASL Alphabet from Kaggle
- 29 classes (A-Z + del, nothing, space)
- Limited to 300 samples per class for faster experimentation (~8,700 images)
- 80-20 train-validation split with stratification

### 5. Evaluation & Visualization
Implemented comprehensive evaluation tools:
- Classification reports with per-class metrics
- Confusion matrix heatmaps
- Learning curves (to assess data scaling effects)
- Per-class F1-score bar charts
- Real-world testing on custom images

---

## Issues Identified

### 🔴 **Critical Issue 1: Background Sensitivity**
**Problem**: The model is highly sensitive to backgrounds in test images.

**Evidence**:
- Test images taken in real-world settings (e.g., with doors/walls in background) fail dramatically
- The HOG feature extractor captures background gradients, not just the hand
- Skin masking alone is insufficient to isolate the hand region

**Why It Happens**:
- Training data from Kaggle is pre-cropped with clean backgrounds
- Test images contain significant non-hand regions
- HOG descriptor operates on the entire 64×64 frame, treating background and hand equally

**Attempted Fix**: 
Smart cropping algorithm was implemented to:
1. Detect skin regions
2. Find largest contour (assumed to be the hand)
3. Compute bounding box around the hand
4. Crop with padding

**Results**: Partial improvement, but still unreliable when:
- Multiple skin-colored objects appear
- Lighting conditions vary significantly
- Backgrounds have wood/brown tones similar to skin

---

### 🟡 **Issue 2: Limited Training Data**
**Problem**: Only 300 samples per class used (due to computational constraints).

**Impact**:
- Full dataset contains ~87,000 images (3,000 per class)
- Using only 10% of available data
- Learning curves show validation accuracy still improving with more data

**Constraint**: Full dataset training with HOG features takes several hours, impractical for iterative development.

---

### 🟡 **Issue 3: Skin Tone Masking Fragility**
**Problem**: HSV skin detection is brittle across lighting conditions.

**Specific Issues**:
- Fails in dim lighting
- Captures non-skin objects with similar HSV values (wood doors, furniture)
- Fixed threshold values don't adapt to environment

**Current Workaround**: Fallback to use entire image when mask returns empty, but this defeats the purpose of masking.

---

### 🟡 **Issue 4: Inter-Class Confusion**
**Problem**: Visually similar gestures are frequently misclassified.

**Common Confusions** (from confusion matrix):
- Letters with similar hand shapes (M/N/S/T)
- Gestures differing only in thumb position
- Orientation-dependent signs

**Root Cause**: HOG + Hu moments don't capture fine-grained finger positioning well at 64×64 resolution.

---

### 🟡 **Issue 5: Real-World Generalization Gap**
**Problem**: Significant accuracy drop on real-world images vs validation set.

**Validation Accuracy**: ~85-90% (depending on model and samples)
**Real-World Test Accuracy**: ~40-60% (with smart cropping)

**Reasons**:
- Domain shift: Kaggle dataset has controlled backgrounds, lighting, and hand positioning
- Test images have natural variability
- No data augmentation used during training

---

## Key Takeaways

### What Works:
✅ Feature extraction pipeline is well-structured and modular  
✅ Training infrastructure is solid (sklearn pipelines, proper train/val splits)  
✅ Visualization tools provide excellent debugging insight  
✅ Model trains quickly on subsets (~5-10 minutes for 300 samples/class)  

### What Doesn't Work:
❌ Poor generalization to real-world images  
❌ Background interference dominates predictions  
❌ Skin masking is unreliable across environments  
❌ Limited discriminative power for fine-grained hand poses  

---

## Recommendations for Improvement

### Short-term Fixes:
1. **Better Hand Detection**: Use MediaPipe or YOLO for robust hand localization before feature extraction
2. **Increase Resolution**: Use 128×128 or 224×224 images to capture more detail
3. **Data Augmentation**: Add background variations, lighting changes, rotations during training
4. **Ensemble Methods**: Combine multiple models or feature sets

### Long-term Solution:
⚠️ **Classical CV may not be sufficient for this task.**

Consider transitioning to:
- **CNNs** (Convolutional Neural Networks) for automatic feature learning
- **Transfer Learning** (ResNet, EfficientNet) pre-trained on ImageNet
- **Pose Estimation** (MediaPipe Holistic) with LSTM/GCN for temporal modeling

These approaches are less sensitive to backgrounds and better at capturing spatial hierarchies in hand shapes.

---

## File Structure
```
classical_cv/
├── RandomForest.ipynb    # Main implementation notebook
└── README.md            # This documentation
```

## Dependencies
- OpenCV (`cv2`)
- scikit-learn (`sklearn`)
- scikit-image (`skimage`)
- NumPy, Matplotlib, Seaborn
- joblib (for model persistence)

## Usage Notes
- Change `DATA_DIR` path based on environment (Kaggle/Colab/Local)
- Adjust `LIMIT_SAMPLES` for training time vs accuracy tradeoff
- Model saved to `models/classical_asl_model.joblib`

---

**Last Updated**: November 2025  
**Status**: Experimental - Not production-ready
