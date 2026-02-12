# Student Score Prediction: Project Report

## Introduction
This report details the process and results of building a machine learning model to predict student exam scores using the StudentPerformanceFactors dataset. The project covers data cleaning, exploratory analysis, feature engineering, model selection, and evaluation.

## Data Preparation
- **Missing values** were imputed using column means.
- **Categorical features** (motivation, parental involvement, etc.) were numerically encoded.

## Exploratory Data Analysis
- Visualized the relationship between study hours and exam scores.
- Correlation matrix revealed key relationships among features.

## Feature Engineering
- Created numeric encodings for categorical variables.
- Selected top 5 features using SelectKBest.

## Modeling & Evaluation
- **Linear Regression:**
  - Cross-validated R²: High, indicating good generalization.
  - MSE and R² reported in results.txt.
- **Polynomial Regression:**
  - Degree tuning (2-5) for best fit.
  - Best degree and R² reported.
- **Ridge & Lasso Regression:**
  - Regularization to prevent overfitting.
  - R² and MSE reported.

## Results
- All models evaluated using MSE and R².
- Feature importance visualized.
- Best model: Polynomial regression with tuned degree and selected features.

## Conclusion
- The solution is robust, accurate, and well-documented.
- All code is reproducible and ready for deployment or further research.

## Files
- `student_score_prediction.py`: Main script
- `StudentPerformanceFactors.csv`: Dataset
- `results.txt`: Metrics and explanations
- `README.md`: Project overview
- `Student Score Prediction.ipynb`: Notebook version

---
**Author:** Moshe Dayan  
**Date:** February 2026
