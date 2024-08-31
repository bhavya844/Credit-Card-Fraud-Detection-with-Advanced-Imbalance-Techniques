# Credit Card Fraud Detection with Advanced Imbalance Techniques

**Author:** Bhavya Dave  
**Date:** August 31, 2024

## Overview

This project addresses the challenges of classifying imbalanced datasets using various machine learning techniques. The project includes:

- **Data preparation**
- **Initial model evaluation**
- **Imbalance mitigation techniques**

The models and techniques applied are evaluated using Precision-Recall and ROC curves, focusing on handling severe class imbalance.

## Data Preparation

### 1. Feature Scaling with MinMaxScaler

- **Objective:** Standardize the features by scaling them within a specific range (0 to 1) using MinMaxScaler.
- **Implementation:** 
  - Loaded the dataset.
  - Scaled the features using MinMaxScaler.
- **Class Distribution Analysis:** 
  - Analyzed the class distribution of the target variable.
  - Highlighted significant imbalance in the dataset.

## Initial Model Evaluation

### 2. Model Implementation and Evaluation

- **Objective:** Evaluate the performance of different machine learning models on the imbalanced dataset.
- **Models Implemented:**
  - RandomForestClassifier
  - LinearSVC
  - NaiveBayes
- **Evaluation Metrics:**
  - Precision-Recall Curves
  - ROC Curves

## Imbalance Mitigation Techniques

### 3. Addressing Imbalance with SMOTE and NearMiss

- **Objective:** Balance the class distribution by generating synthetic samples and under-sampling the majority class.
- **Techniques Applied:**
  - **SMOTE (Synthetic Minority Over-sampling Technique):** Created synthetic samples of the minority class.
  - **NearMiss (Version 3):** Under-sampled the majority class to reduce redundancy.
  
- **Easy Ensemble Method:**
  - **Objective:** Evaluate the effect of ensemble methods on imbalanced datasets.
  - **Implementation:** Applied the Easy Ensemble method to create balanced subsets and trained a model on each subset.

## Results

- **Model Performance:** Evaluated models before and after applying imbalance mitigation techniques.
- **Impact of Imbalance Mitigation:** 
  - Improved recall of the minority class using SMOTE and NearMiss.
  - Enhanced model robustness with Easy Ensemble.

## Conclusion

This project demonstrated the importance of addressing class imbalance to improve model performance, particularly for the minority class. Significant improvements were observed in model accuracy after applying SMOTE, NearMiss, and Easy Ensemble techniques.

## Requirements

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (imblearn)
- Matplotlib
- Seaborn

## References

- [SMOTE Documentation](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- [NearMiss Documentation](https://imbalanced-learn.org/stable/references/generated/imblearn.under_sampling.NearMiss.html)
- [Easy Ensemble Documentation](https://imbalanced-learn.org/stable/references/generated/imblearn.ensemble.EasyEnsembleClassifier.html)
