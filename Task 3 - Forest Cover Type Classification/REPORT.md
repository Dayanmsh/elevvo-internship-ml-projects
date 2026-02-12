# Task 3: Forest Cover Type Classification – Report

## Introduction
This report documents the process and results of predicting forest cover type using cartographic and environmental features. The project leverages tree-based models, advanced preprocessing, and class balancing techniques to achieve robust multi-class classification.

## Dataset
- **Source:** UCI Covertype dataset (or provided dataset)
- **Features:** Elevation, slope, soil type, and more
- **Target:** Cover type (multi-class)

## Methodology
1. **Data Cleaning & Preprocessing:**
   - Checked for missing values (none found)
   - Encoded categorical variables (e.g., soil type)
   - Feature scaling applied for non-tree models
2. **Exploratory Data Analysis (EDA):**
   - Visualized class distribution, feature correlations, and outliers
   - Identified class imbalance
3. **Model Building:**
   - Trained Random Forest and XGBoost classifiers
   - Applied SMOTE to address class imbalance
   - Compared Logistic Regression and Decision Tree as baseline models
4. **Evaluation:**
   - Used accuracy, classification report, and confusion matrix
   - Visualized feature importances

## Results
- **Random Forest:** High accuracy, robust to feature scaling, interpretable feature importances
- **XGBoost:** Comparable or better performance, especially with tuned hyperparameters
- **SMOTE:** Improved minority class recall
- **Logistic Regression & Decision Tree:** Lower accuracy, useful as baselines

## Visualizations
- Confusion matrices for all models
- Feature importance bar plots

## Insights
- Tree-based models are well-suited for this task
- Class imbalance can be mitigated with SMOTE
- Feature importance highlights key predictors (e.g., elevation, soil type)

## Recommendations
- Further improve with hyperparameter tuning and ensemble methods
- Consider additional feature engineering

---
*See the notebook and script for code, outputs, and detailed explanations.*
