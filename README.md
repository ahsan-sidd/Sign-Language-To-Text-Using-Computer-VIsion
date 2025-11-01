# ✋ Sign Language to Text using Computer Vision

### 📚 Applied Digital Image Processing — Course Project  
**Team Members:**  
- Ahsan Siddiqui  
- Hassaan Tariq  
- Hanzala Khan  

---

## 🧠 Overview

Communication barriers between sign language users and non-signers create major challenges in education, workplaces, and everyday interactions.  
This project aims to **bridge that communication gap** by building a **computer vision system** that recognizes sign language gestures and converts them into text in real-time.

The system will explore and compare **classical image processing** methods with **deep learning**-based models for sign recognition.

---

## 🎯 Objectives

- Recognize hand gestures from sign language using computer vision.  
- Translate recognized gestures into readable text in real-time.  
- Compare the performance of traditional ML and deep learning approaches.  

---

## 🧩 Methodology

### 🏗 1. Traditional Image Processing Pipeline
- **Preprocessing:** Background subtraction, skin-color segmentation, contour extraction.  
- **Feature Extraction:** Histogram of Oriented Gradients (HOG), Hu Moments.  
- **Classification:** Support Vector Machine (SVM) or Random Forest.  

### 🤖 2. Deep Learning Pipeline
- **Static Signs (Alphabets/Numbers):** Convolutional Neural Networks (CNNs).  
- **Dynamic Gestures:** CNN + LSTM for sequence learning.  
- **Transfer Learning:** Using pretrained lightweight models like MobileNet or EfficientNet for improved accuracy.  

---

## 🗂 Dataset

| Dataset | Description |
|----------|--------------|
| **Kaggle ASL Alphabet Dataset** | 87,000+ images across 29 static sign classes (A–Z, space, delete, nothing). |
| **Custom Dataset (optional)** | Collected via webcam for small-scale live demos. |
| **RWTH-PHOENIX-Weather 2014T (optional)** | For advanced scope: continuous/dynamic sign sequences. |

---

## ⚙️ Evaluation Metrics

| Metric | Purpose |
|---------|----------|
| **Accuracy** | Overall model performance |
| **Precision, Recall, F1-Score** | Class-level performance |
| **Confusion Matrix** | Error analysis and misclassification insights |

---

## 🚀 Expected Outcomes

- A working **real-time demo**:  
  **Camera Input → Gesture Recognition → Text Output**  

- Comparative results between:
  - HOG + SVM  
  - CNN (for static gestures)  
  - CNN + LSTM (for dynamic gestures)

- A clear understanding of **trade-offs between traditional and deep learning methods** in sign language recognition.

---

## 🧮 Deliverables

- 📝 Project Report  
- 💻 Source Code + Jupyter Notebooks  
- 🎥 Demo Video of working prototype  
- 📊 Comparative performance analysis  

---

## 📚 Future Extensions

- Add **speech synthesis** for voice output.  
- Expand to **continuous sign language translation**.  
- Integrate **MediaPipe** or **ST-GCN** for skeleton-based pose estimation.  
- Deploy as a **web or mobile app** for accessibility.  

---

## 🛠 Tools & Libraries

| Category | Tools |
|-----------|--------|
| **Languages** | Python |
| **Libraries** | OpenCV, scikit-learn, NumPy, TensorFlow / PyTorch, Matplotlib |
| **Frameworks** | MediaPipe (for keypoint detection) |
| **Dataset Sources** | Kaggle, RWTH-PHOENIX, custom webcam input |

---

## 📁 Repository Structure (Planned)

