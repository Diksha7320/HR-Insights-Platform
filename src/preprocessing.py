import pandas as pd
import numpy as np

def clean_data(df, target_col):
    df = df.copy()

    # 1️⃣ Clean target column (Attrition)
    if target_col == "Attrition":
        df[target_col] = df[target_col].replace({'Yes': 1, 'No': 0})

    # 2️⃣ Categorical encodings from notebook

    if "EducationField" in df.columns:
        df["EducationField"] = df["EducationField"].replace({
            'Life Sciences': 1,
            'Medical': 2,
            'Marketing': 3,
            'Other': 4,
            'Technical Degree': 5,
            'Human Resources': 6
        })

    if "Department" in df.columns:
        df["Department"] = df["Department"].replace({
            'Research & Development': 1,
            'Sales': 2,
            'Human Resources': 3
        })

    if "BusinessTravel" in df.columns:
        df["BusinessTravel"] = df["BusinessTravel"].replace({
            "Travel_Rarely": 1,
            "Travel_Frequently": 2,
            "Non-Travel": 3
        })

    if "OverTime" in df.columns:
        df["OverTime"] = df["OverTime"].replace({"Yes": 1, "No": 2})

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].replace({"Male": 1, "Female": 2})

    if "JobRole" in df.columns:
        df["JobRole"] = df["JobRole"].replace({
            "Laboratory Technician": 8,
            "Sales Executive": 7,
            "Research Scientist": 6,
            "Sales Representive": 5,
            "Human Resources": 4,
            "Manufacturing Director": 3,
            "Healthcare Representative": 2,
            "Manager": 1,
            "Research Director": 0
        })

    # 3️⃣ Drop irrelevant columns
    cols_to_drop = ["EmployeeCount","EmployeeNumber","Over18","StandardHours"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    # 4️⃣ Drop missing values
    df = df.dropna()

    # 5️⃣ Convert remaining categorical features into numeric
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype("category").cat.codes

    return df
