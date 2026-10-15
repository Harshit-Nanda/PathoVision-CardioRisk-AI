# PathoVision & CardioRisk AI — Multi-Modal Medical AI Suite

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-Academic%20MIT-brightgreen.svg)](LICENSE)

An enterprise multi-modal medical artificial intelligence suite engineered by **Harshit Nanda (23BAI10489)** for the **MPOnline Internship & Certification Programme (Batch 13A)**.

---

## Subsystems Overview

### 1. PathoVision CNN — Deep Histopathological Lung Cancer Classification
- **Clinical Motivation:** Automated histological classification of lung biopsy tissue specimens.
- **Target Classes:** 
  1. `lung_n`: Normal lung tissue
  2. `lung_aca`: Lung Adenocarcinoma
  3. `lung_scc`: Lung Squamous Cell Carcinoma
- **Architecture:** 3-block Convolutional Neural Network with 32, 64, and 128 filter kernels, `MaxPooling2D`, `BatchNormalization`, `Dropout(0.3)`, and Softmax output.
- **Optimization:** Adam optimizer with Categorical Cross-Entropy loss, `EarlyStopping` (patience=3), and `ReduceLROnPlateau`.

### 2. CardioRisk AI — Advanced Machine Learning Model Comparison for Heart Disease
- **Clinical Motivation:** Predict cardiovascular patient risk based on clinical and demographic attributes.
- **Comparative Framework:**
  - Decision Tree Classifier
  - Random Forest Classifier (GridSearchCV Tuned)
  - Support Vector Machine (SVM RBF & Linear)
- **Validation:** 5-Fold Stratified Cross-Validation, Recursive Feature Elimination (RFE), ROC-AUC and Confusion Matrix analysis.
- **Empirical Recommendation:** Tuned Random Forest achieved optimal clinical balance (ROC-AUC 0.88, 85.4% accuracy, 0.84 recall).

### 3. NLP Review Sentiment & Healthcare Conversational Assistant
- **Amazon Product Reviews Sentiment:** TF-IDF n-gram vectorization with Logistic Regression.
- **Clinical FAQ Chatbot:** Rule-based pattern matching and intent classification for medical questions.

---

## Project Structure

```
PathoVision-CardioRisk-AI/
├── models/
│   ├── cnn_classifier.py            # Deep CNN architecture for histopathology
│   └── heart_disease_comparison.py  # Benchmark: Decision Tree, RF, SVM, GridSearchCV
├── nlp/
│   ├── sentiment_analyzer.py        # Amazon reviews sentiment classifier
│   └── healthcare_chatbot.py        # Clinical conversational assistant
├── data/
│   └── generate_demo_data.py        # Synthetic dataset generator for instant execution
├── train_cnn.py                     # Training script for CNN classifier
├── requirements.txt                 # Dependencies
└── README.md                        # Documentation
```

---

## Getting Started

### 1. Setup Environment
```bash
git clone https://github.com/Harshit-Nanda/PathoVision-CardioRisk-AI.git
cd PathoVision-CardioRisk-AI

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Heart Disease Machine Learning Benchmark
```bash
python models/heart_disease_comparison.py
```

### 3. Run Histopathology CNN Training
```bash
python train_cnn.py
```

---

## Author
**Harshit Nanda**  
Registration No.: `23BAI10489`  
B.Tech CSE (Artificial Intelligence & Machine Learning)  
VIT Bhopal University
