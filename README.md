# 🔐 Anomaly Detection for Data Leakage Prevention

A hybrid machine learning framework for detecting network intrusions and preventing data leakage using a weighted ensemble approach.

## 🚀 Overview

This project combines:

* Random Forest
* XGBoost
* Multi-Layer Perceptron (MLP)
* Autoencoder (Unsupervised)

The model is trained on **KDDTrain+** and tested on **KDDTest+** to evaluate performance on novel attacks.

## 🧠 Key Features

* Custom feature engineering (5 new behavioral features)
* PCA for dimensionality reduction (10 components)
* Strict data leakage prevention during preprocessing
* Weighted voting ensemble for final prediction

## 📊 Performance (KDDTest+)

* Accuracy: **69.66%**
* Precision: **0.83**
* Recall: **0.59**
* F1-Score: **0.70**
* AUC: **0.7599**

High precision ensures low false alarm rate in real-world deployment.

## 🛠 Tech Stack

* Python
* Scikit-learn
* TensorFlow / Keras
* XGBoost
* Pandas, NumPy

