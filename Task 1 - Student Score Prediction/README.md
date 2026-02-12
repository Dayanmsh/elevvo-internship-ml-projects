# Task 1: Student Score Prediction

## Overview
This project builds a machine learning model to predict students' exam scores based on their study hours and other factors, using the provided StudentPerformanceFactors dataset. The solution includes data cleaning, exploratory data analysis, feature engineering, model selection, and performance evaluation.


## Dataset
- **File:** StudentPerformanceFactors.csv (excluded from repo)
- **Features:** Study hours, attendance, sleep hours, previous scores, motivation, parental involvement, resources, family income, teacher quality, peer influence, physical activity, and more.
- **Target:** Exam_Score
- **Download:** [Google Drive Link or Dropbox Link] (add your dataset link here)

*Note: The dataset is excluded from the repository due to size. Please download it using the link above and place it in the Task 1 folder before running the code.*

## Approach
1. **Data Cleaning:** Handle missing values and encode categorical features.
2. **Exploratory Data Analysis:** Visualize relationships and correlations.
3. **Feature Engineering:** Create and select the most relevant features.
4. **Modeling:**
   - Linear Regression (with cross-validation)
   - Polynomial Regression (degree tuning)
   - Ridge and Lasso Regression (regularization)
5. **Evaluation:**
   - Mean Squared Error (MSE)
   - R² Score
   - Feature importance and selection
6. **Results:**
   - All results and explanations are saved in `results.txt`.

## Bonus
- Polynomial regression with degree tuning
- Feature selection and regularization
- All code is robust to missing values and optimized for best performance

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the script:
   ```bash
   python student_score_prediction.py
   ```
3. See `results.txt` for metrics and explanations.

## Notebook Version
A Jupyter notebook version is provided for step-by-step exploration and visualization.

## Author
- Your Name
- February 2026
