# 🌍 Satellite Image Classification using GLCM & SDH

This project uses **texture-based features** extracted from satellite images to classify land types like crops, roads, water bodies, and more. We utilize **GLCM (Gray-Level Co-Occurrence Matrix)** and **SDH (Sum-Difference Histogram)** features to train machine learning models for effective classification.

---

## 🚀 Project Overview

🛰️ Satellite images contain rich data across multiple spectral bands. Instead of using deep learning or raw pixel data, this project focuses on **handcrafted statistical texture features** that capture patterns in grayscale variations.

---

## 🛠️ Workflow

### 1. 📥 Data Collection
- Collected multiple satellite images of different terrain types:
  - 🌾 Crops
  - 🛣️ Roads
  - 🌊 Water
  - 🏞️ Forest
  - 🏙️ Urban
  - 🏜️ Barren land
- Each image was preprocessed and converted to grayscale for texture analysis.

### 2. 📐 Feature Extraction

**a. GLCM (Gray-Level Co-Occurrence Matrix)**  
- Captures spatial relationships between pixel intensities.
- Extracted features: Contrast, Dissimilarity, Homogeneity, Energy, Correlation, ASM

**b. SDH (Sum-Difference Histogram)**  
- Captures differences and sums of pixel pairs in the image.
- Extracted statistical descriptors from SDH histograms.

> 🔍 These features help understand the texture and structural layout of the terrain.

### 3. 🤖 Model Training
- Combined GLCM and SDH features into a feature vector.
- Trained several ML models like:
  - 🎯 **Random Forest**
  - 💠 **SVM**
  - 🔺 **KNN**
  - 🧠 **Simple Neural Network (optional)**

### 4. 🧪 Evaluation
- Dataset was split into train and test sets.
- Models were evaluated using:
  - Accuracy
  - Confusion Matrix
  - Classification Report

---

## 📊 Results

| Model          | Accuracy |
|----------------|----------|
| Random Forest  | ✅ ~90%+ |
| SVM            | ✅ Good |
| KNN            | ✅ Decent |
| Neural Net     | ✅ Optional |

> **Random Forest** worked best due to its robustness with non-linear features.

---

## 📂 Folder Structure

