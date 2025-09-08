# Earthquake_Tsunami_ML.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer


# ================== 1. Load Dataset ==================
file_path = "D:/ML_Project\project yash\Earth2-23_modified.csv"
df = pd.read_csv(file_path)

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)


# ================== 2. Preprocessing ==================
# Separate numerical and categorical columns
numerical_cols = df.select_dtypes(include=['number']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Handle missing values
numeric_imputer = SimpleImputer(strategy='mean')
# Fit the imputer ONLY on the feature columns
numeric_imputer.fit(df[numerical_cols])
df[numerical_cols] = numeric_imputer.transform(df[numerical_cols])


categorical_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

# Label Encode categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ================== 3. Define Features and Target ==================
target = 'tsunami'   # New Target Column
features = df.columns[df.columns != target]
X = df[features]
y = df[target]

# Scale numerical features
scaler = MinMaxScaler()
# Fit the scaler ONLY on the feature columns
scaler.fit(df[numerical_cols])
df[numerical_cols] = scaler.transform(df[numerical_cols])


X = df[features]
y = df[target]

# Define columns ONLY from X (features, not including target)
numerical_cols = X.select_dtypes(include=['number']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Split train & test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)


# ================== 4. Machine Learning Models ==================
# ---- KNN ----
for k in range(1, 3):
    classifier = KNeighborsClassifier(n_neighbors=k)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    print(f"\n🔹 KNN (k={k})")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Accuracy:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

# ---- Decision Tree ----
dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train, y_train)
y_pred_dt = dtc.predict(X_test)
print("\n🔹 Decision Tree")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_dt))
print("Accuracy:", round(accuracy_score(y_test, y_pred_dt) * 100, 2), "%")
print("Classification Report:\n", classification_report(y_test, y_pred_dt))

# ---- Random Forest ----
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred_rf = rf_classifier.predict(X_test)
print("\n🔹 Random Forest")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Accuracy:", round(accuracy_score(y_test, y_pred_rf) * 100, 2), "%")
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

# ---- SVM -----------------------------
svm_linear = LinearSVC(max_iter=2000, class_weight="balanced", random_state=42)
svm_linear.fit(X_train, y_train)
y_pred_svm_linear = svm_linear.predict(X_test)
print("\n🔹 Linear SVM (fast)")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm_linear))
print("Accuracy:", round(accuracy_score(y_test, y_pred_svm_linear) * 100, 2), "%")
print("Classification Report:\n", classification_report(y_test, y_pred_svm_linear, zero_division=1))


# ---- Logistic Regression ----
log_reg_model = LogisticRegression(max_iter=1000)
log_reg_model.fit(X_train, y_train)
y_pred_lr = log_reg_model.predict(X_test)
print("\n🔹 Logistic Regression")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("Accuracy:", round(accuracy_score(y_test, y_pred_lr) * 100, 2), "%")
print("Classification Report:\n", classification_report(y_test, y_pred_lr))

# ---- Naive Bayes ----
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
print("\n🔹 Naive Bayes")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))
print("Accuracy:", round(accuracy_score(y_test, y_pred_nb) * 100, 2), "%")
print("Classification Report:\n", classification_report(y_test, y_pred_nb))

# Store the trained model and scaler
final_model = log_reg_model # Or any other model you want to use for prediction
global_scaler = scaler # Store the scaler in a global variable
# Store the numeric imputer as well for consistent preprocessing of new data
global_numeric_imputer = numeric_imputer


# ================== 5. Visualizations ==================
# Heatmap
# plt.figure(figsize=(12, 8))
# sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
# plt.title("Feature Correlation Heatmap")
# plt.show(block=False)

# # Scatter Plot example (choose 2 numeric cols if available)
# if 'Magnitude' in df.columns and 'Depth' in df.columns:
#     plt.figure(figsize=(8, 6))
#     sns.scatterplot(data=df, x='Magnitude', y='Depth', hue='tsunami', palette='Set1', alpha=0.6)
#     plt.title("Scatter Plot: Magnitude vs Depth (Tsunami)")
#     plt.show()

# # Histograms
# #fig, axes = plt.subplots(len(numerical_cols), 1, figsize=(10, 5 * len(numerical_cols)))
# fig, axes = plt.subplots(len(numerical_cols), 1, figsize=(10, 5 * len(numerical_cols)))
# if len(numerical_cols) == 1:
#     axes = [axes]  # make it iterable
# for i, col in enumerate(numerical_cols):
#     sns.histplot(df[col], bins=30, kde=True, ax=axes[i])
#     axes[i].set_title(f"Histogram of {col}")
# plt.tight_layout()
# plt.show(block=False)

# Boxplots
# fig, axes = plt.subplots(1, len(numerical_cols), figsize=(15, 5))
# for i, col in enumerate(numerical_cols):
#     sns.boxplot(x=y, y=df[col], ax=axes[i])
#     axes[i].set_title(f"{col} vs {target}")
# plt.tight_layout()
# plt.show()
'''
plt.figure(figsize=(6, 4))
sns.boxplot(y=df['tsunami'], color='skyblue')
plt.title("Box Plot of Tsunami")
plt.ylabel("Tsunami")
plt.tight_layout()
plt.show()

plt.ioff()
'''

# ---------------- Final Prediction Example ----------------
import numpy as np

# Let's use the trained Random Forest model (you can change this to any)
final_model = RandomForestClassifier()
final_model.fit(X_train, y_train)

# Example new patient data (must match the features in X)
# Format: [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, ...]
# ⚠️ Use actual columns from your dataset (order must match X.columns)
sample_data = pd.DataFrame([{
    'Pregnancies': 2,
    'Glucose': 150,
    'BloodPressure': 80,
    'SkinThickness': 25,
    'Insulin': 120,
    'BMI': 32.5,
    'DiabetesPedigreeFunction': 0.5,
    'Age': 45
}])

# 1. Apply the same preprocessing (scaling, encoding)
# If you had categorical columns, encode them
for col, le in label_encoders.items():
    if col in sample_data:
        sample_data[col] = le.transform(sample_data[col])

# Scale numerical columns
sample_data[numerical_cols] = scaler.transform(sample_data[numerical_cols])

# 2. Make prediction
prediction = final_model.predict(sample_data)[0]
prediction_proba = final_model.predict_proba(sample_data)[0][1]

# 3. Show result
print("Final Prediction (0 = No Diabetes, 1 = Diabetes):", prediction)
print(f"Probability of Diabetes: {prediction_proba:.2f}")
