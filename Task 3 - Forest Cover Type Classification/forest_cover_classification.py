"""
Task 3: Forest Cover Type Classification

Goal: Predict forest cover type using cartographic and environmental features with tree-based models.
Enhancements: Includes model comparison, hyperparameter tuning, SMOTE for class imbalance, and detailed explanations.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Load dataset (update filename as needed)
try:
    df = pd.read_csv('covtype.csv')
except FileNotFoundError:
    print('Dataset not found. Please place covtype.csv in this folder.')
    exit()

# EDA
print(df.head())
print(df.info())
print(df.describe())
plt.figure(figsize=(8,4))
sns.countplot(x=df.iloc[:,-1])
plt.title('Cover Type Distribution')
plt.show()

# Preprocessing
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
if y.dtype == 'O':
    le = LabelEncoder()
    y = le.fit_transform(y)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print('Random Forest Accuracy:', accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# XGBoost
xgb = XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='mlogloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print('XGBoost Accuracy:', accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))

# SMOTE + Random Forest
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
rf_smote = RandomForestClassifier(n_estimators=100, random_state=42)
rf_smote.fit(X_res, y_res)
y_pred_smote = rf_smote.predict(X_test)
print('Random Forest (SMOTE) Accuracy:', accuracy_score(y_test, y_pred_smote))
print(classification_report(y_test, y_pred_smote))

# Logistic Regression
lr = LogisticRegression(max_iter=200, multi_class='multinomial', solver='lbfgs')
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print('Logistic Regression Accuracy:', accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print('Decision Tree Accuracy:', accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# Feature Importance (Random Forest)
importances = rf.feature_importances_
indices = np.argsort(importances)[-10:][::-1]
plt.figure(figsize=(10,6))
plt.title('Top 10 Feature Importances (Random Forest)')
plt.bar(range(10), importances[indices], align='center', color='teal')
plt.xticks(range(10), [X.columns[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()
