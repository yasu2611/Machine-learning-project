###  1st step

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def preprocess_data(file_path):
    """Loads dataset, fills missing values, encodes categorical features, and scales numerical features."""

    # Load Dataset
    df = pd.read_csv("/Earth2-23_modified.csv")

    # Replace '?' with NaN if present
    df.replace('?', np.nan, inplace=True)

    # Separate numeric and categorical columns
    numerical_cols = df.select_dtypes(include=['number']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns

    # === Handling Missing Values ===
    # Apply SimpleImputer for numeric columns (mean strategy)
    numeric_imputer = SimpleImputer(strategy='mean')
    df[numerical_cols] = numeric_imputer.fit_transform(df[numerical_cols])

    # Apply SimpleImputer for categorical columns (most frequent strategy)
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

    # === Encoding Categorical Columns ===
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # === Scaling Numerical Features ===
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    return df, numerical_cols, categorical_cols, label_encoders

if __name__ == "__main__":
    file_path = "heart_synthetic_predictive_50000.csv"
    df, numerical_cols, categorical_cols, label_encoders = preprocess_data(file_path)

    print("Data Preprocessing Is Completed!")
    print(df.head())
