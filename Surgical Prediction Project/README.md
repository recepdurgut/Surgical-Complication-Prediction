# Surgical Complication Prediction using Random Forest & PCA

This repository contains a machine learning pipeline designed to predict post-operative complications based on surgical data. The project utilizes **Principal Component Analysis (PCA)** for dimensionality reduction and a **Random Forest Classifier** for robust prediction.

## 📌 Project Overview
The goal of this project is to analyze clinical data to identify patients at higher risk of surgical complications. By implementing automated classification, healthcare providers can potentially enhance patient monitoring and outcome management.

## 📊 Workflow
1. **Data Preprocessing**: Handling categorical variables via Label Encoding and normalizing features using StandardScaler.
2. **Dimensionality Reduction**: Applying PCA to reduce the feature space while retaining the most significant variance, preventing overfitting.
3. **Model Selection**: Using a Random Forest ensemble to handle non-linear relationships and provide stable predictions.
4. **Evaluation**: Measuring performance through Accuracy, Precision, Recall, and a visualized Confusion Matrix.

## 🛠️ Tech Stack
- **Python 3.x**
- **Pandas** (Data Manipulation)
- **Scikit-Learn** (Machine Learning & PCA)
- **Matplotlib & Seaborn** (Data Visualization)

## 📁 Project Structure
```text
├── Surgical-deepnet.csv     # Clinical dataset
├── surgical_prediction.py   # Main ML script
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
