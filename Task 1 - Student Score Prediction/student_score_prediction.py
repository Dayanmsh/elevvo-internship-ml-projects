
"""
Task 1: Student Score Prediction

Goal: Build a regression model to predict students' exam scores based on their study hours and other factors.
Enhancements: Includes advanced feature engineering, polynomial regression tuning, feature selection, cross-validation, and detailed explanations.
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression


# Load real dataset
import os
data_path = os.path.join(os.path.dirname(__file__), 'StudentPerformanceFactors.csv')
data = pd.read_csv(data_path)

# Data cleaning: drop rows with missing Exam_Score or Hours_Studied
data = data.dropna(subset=['Exam_Score', 'Hours_Studied'])



# Data exploration
print("Data Head:\n", data.head())
print("Data Description:\n", data.describe())
print("\nColumns:", data.columns.tolist())

# Visualize relationship between Hours_Studied and Exam_Score
plt.figure(figsize=(6,4))
plt.scatter(data['Hours_Studied'], data['Exam_Score'], alpha=0.5)
plt.xlabel('Hours Studied')
plt.ylabel('Exam Score')
plt.title('Hours Studied vs Exam Score')
plt.show()

# Visualize correlation matrix for numeric features
plt.figure(figsize=(10,6))
numeric_cols = data.select_dtypes(include=[np.number]).columns
correlation = data[numeric_cols].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()



# Feature engineering: encode categoricals, select best features
data['Motivation_Level_Num'] = data['Motivation_Level'].map({'Low': 0, 'Medium': 1, 'High': 2})
data['Parental_Involvement_Num'] = data['Parental_Involvement'].map({'Low': 0, 'Medium': 1, 'High': 2})
data['Access_to_Resources_Num'] = data['Access_to_Resources'].map({'Low': 0, 'Medium': 1, 'High': 2})
data['Family_Income_Num'] = data['Family_Income'].map({'Low': 0, 'Medium': 1, 'High': 2, 'High ': 2})
data['Teacher_Quality_Num'] = data['Teacher_Quality'].map({'Low': 0, 'Medium': 1, 'High': 2})
data['Peer_Influence_Num'] = data['Peer_Influence'].map({'Negative': -1, 'Neutral': 0, 'Positive': 1})


# Select features for best performance
feature_cols = [
    'Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores',
    'Motivation_Level_Num', 'Parental_Involvement_Num', 'Access_to_Resources_Num',
    'Family_Income_Num', 'Teacher_Quality_Num', 'Peer_Influence_Num',
    'Physical_Activity'
]
X = data[feature_cols]
y = data['Exam_Score']

# Impute missing values in features with column mean
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)


# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Linear Regression with cross-validation
lin_reg = LinearRegression()
cv_scores = cross_val_score(lin_reg, X_scaled, y, cv=5, scoring='r2')
print(f"Linear Regression CV R2: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)
print("Linear Regression Results:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Feature importance (coefficients)
plt.figure(figsize=(8,4))
plt.bar(feature_cols, np.abs(lin_reg.coef_))
plt.xticks(rotation=45, ha='right')
plt.title('Linear Regression Feature Importance (abs coef)')
plt.tight_layout()
plt.show()


# Bonus: Polynomial Regression (degree tuning)
best_poly_r2 = -np.inf
best_deg = 1
for deg in range(2, 6):
    poly = PolynomialFeatures(degree=deg)
    X_poly = poly.fit_transform(X_scaled)
    X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(X_poly, y, test_size=0.2, random_state=42)
    poly_reg = LinearRegression()
    poly_reg.fit(X_train_poly, y_train_poly)
    y_pred_poly = poly_reg.predict(X_test_poly)
    r2 = r2_score(y_test_poly, y_pred_poly)
    print(f"Polynomial Regression (deg={deg}) R2: {r2:.3f}")
    if r2 > best_poly_r2:
        best_poly_r2 = r2
        best_deg = deg
        best_poly_model = poly_reg
        best_poly_X_test = X_test_poly
        best_poly_y_test = y_test_poly
        best_poly_y_pred = y_pred_poly

print(f"Best Polynomial Degree: {best_deg} (R2={best_poly_r2:.3f})")

plt.scatter(best_poly_y_test, best_poly_y_pred, alpha=0.5)
plt.xlabel('Actual Exam Score')
plt.ylabel('Predicted Exam Score')
plt.title(f'Best Polynomial Regression (deg={best_deg})')
plt.plot([min(best_poly_y_test), max(best_poly_y_test)], [min(best_poly_y_test), max(best_poly_y_test)], 'r--')
plt.show()



# Feature selection: SelectKBest
selector = SelectKBest(score_func=f_regression, k=5)
X_selected = selector.fit_transform(X_scaled, y)
selected_features = np.array(feature_cols)[selector.get_support()]
print("Top 5 Features:", selected_features)

# Ridge and Lasso regression for regularization
ridge = RidgeCV(alphas=np.logspace(-3, 3, 7), cv=5)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
print("Ridge Regression Results:")
print("MSE:", mean_squared_error(y_test, ridge_pred))
print("R2 Score:", r2_score(y_test, ridge_pred))

lasso = LassoCV(alphas=np.logspace(-3, 3, 7), cv=5, max_iter=10000)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
print("Lasso Regression Results:")
print("MSE:", mean_squared_error(y_test, lasso_pred))
print("R2 Score:", r2_score(y_test, lasso_pred))


# Save results and model explanations
with open('results.txt', 'w') as f:
    f.write(f"Linear Regression MSE: {mean_squared_error(y_test, y_pred):.4f}\n")
    f.write(f"Linear Regression R2: {r2_score(y_test, y_pred):.4f}\n")
    f.write(f"Linear Regression CV R2: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")
    f.write(f"Best Polynomial Degree: {best_deg}\n")
    f.write(f"Polynomial Regression R2: {best_poly_r2:.4f}\n")
    f.write(f"Ridge Regression R2: {r2_score(y_test, ridge_pred):.4f}\n")
    f.write(f"Lasso Regression R2: {r2_score(y_test, lasso_pred):.4f}\n")
    f.write(f"Top 5 Features: {selected_features.tolist()}\n")
    f.write("\nExplanations:\n")
    f.write("- Linear regression models the relationship between features and exam score as a straight line.\n")
    f.write("- Polynomial regression allows for a curved relationship, which may fit the data better.\n")
    f.write("- Feature engineering and selection (e.g., motivation, attendance, sleep) can improve model accuracy.\n")
    f.write("- Ridge and Lasso add regularization to prevent overfitting and select important features.\n")
    f.write("- Cross-validation provides a robust estimate of model performance.\n")
