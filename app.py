import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import imblearn

def make_column_names_unique(columns):
    """
    Makes column names unique by adding .1, .2, .3 etc. to duplicates.
    """
    seen = {}
    new_cols = []
    for col in columns:
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}.{seen[col]}")
    return new_cols

st.set_page_config(page_title="HR Analytics App", layout="wide")

# -----------------------------
# SIDEBAR NAVIGATION
# -----------------------------
st.sidebar.title("HR Analytics App")

page = st.sidebar.radio(
    "Go to:",
    ["Home", "Data", "SQL Runner", "EDA", "Modeling", "Dashboard", "About"]
)

# -----------------------------
# HOME PAGE
# -----------------------------
if page == "Home":
    st.title("👩‍💼 HR Analytics — End-to-End Project")

    st.markdown("""
    Welcome to the **HR Analytics App** — a complete end-to-end data analytics project 
    designed to help organizations understand **employee attrition, workforce distribution, 
    demographic insights, and HR KPIs** using data-driven techniques.

    ### 📌 What this project is about
    This application brings together:
    - **Data Engineering**
    - **Exploratory Data Analysis (EDA)**
    - **SQL Transformations & KPI Calculations**
    - **Machine Learning (Attrition Prediction)**
    - **Dashboarding & Storytelling**

    All integrated into a single interactive Streamlit application.

    
    """)

# -----------------------------
# DATA PAGE
# -----------------------------
elif page == "Data":
    st.title("📁 Data Explorer")

    st.markdown("""
    Upload your HR dataset or load the default dataset provided with the app.  
    This data will be used for SQL, EDA, Dashboard, and ML model building.
    """)

    # --- File Upload ---
    file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

    # CASE 1: user uploaded
    if file is not None:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        st.success(f"Uploaded file: **{file.name}**")

        # FIX DUPLICATE COLUMNS
        df.columns = make_column_names_unique(df.columns)

    # CASE 2: try default data folder
    else:
        st.info("No file uploaded. Trying to load default dataset from **/data** folder...")

        import glob
        csv_files = glob.glob("data/*.csv")

        if len(csv_files) > 0:
            default_path = csv_files[0]
            df = pd.read_csv(default_path)

            # fix duplicates
            df.columns = make_column_names_unique(df.columns)

            st.success(f"Loaded default dataset: **{default_path}**")
        else:
            st.error("❌ No CSV found in the /data folder. Please upload a dataset.")
            st.stop()

    # store
    st.session_state["df"] = df

    # preview
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    # shape
    st.subheader("📐 Dataset Shape")
    st.write(f"Rows: **{df.shape[0]}**, Columns: **{df.shape[1]}**")

    # column info
    st.subheader("🧾 Column Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Column Names**")
        st.write(list(df.columns))
    with col2:
        st.write("**Data Types**")
        st.write(df.dtypes)

    # missing
    st.subheader("⚠️ Missing Values")
    st.write(df.isna().sum())

    # download cleaned
    st.subheader("⬇️ Download Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download as CSV", csv, "cleaned_hr_data.csv")


# ------------------------------
# SQL RUNNER PAGE
# ------------------------------
elif page == "SQL Runner":
    st.title("🧮 SQL Query Runner")

    st.markdown("""
    Run SQL queries directly on your dataset.  
    Useful for KPI calculations, validation, and transformations.

    ⚠️ Note: Only **SELECT** queries are allowed for safety.
    """)

    if "df" not in st.session_state:
        st.warning("Please upload a dataset first in the **Data** page.")
        st.stop()

    df = st.session_state["df"]

    # in-memory sqlite
    import sqlite3
    conn = sqlite3.connect(":memory:")
    df.to_sql("hr_table", conn, index=False, if_exists="replace")
    st.success("Dataset loaded into SQL as table **hr_table**")

    # load docx queries if exists (optional)
    import os
    preloaded_queries = ""
    if os.path.exists("sql/queries.docx"):
        from docx import Document
        doc = Document("sql/queries.docx")
        preloaded_queries = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

    query = st.text_area(
        "Enter SQL Query",
        value=preloaded_queries if preloaded_queries else "SELECT * FROM hr_table LIMIT 10;",
        height=250
    )

    if st.button("Execute SQL"):
        try:
            result = pd.read_sql_query(query, conn)
            st.dataframe(result, use_container_width=True)
            csv_data = result.to_csv(index=False).encode('utf-8')
            st.download_button("Download Result as CSV", csv_data, "sql_result.csv")
        except Exception as e:
            st.error(f"SQL Error: {e}")

# -----------------------------
# EDA PAGE
# -----------------------------
elif page == "EDA":
    st.title("📊 Exploratory Data Analysis (EDA)")

    if "df" not in st.session_state:
        st.warning("Please upload a dataset first from the **Data** page.")
        st.stop()

    df = st.session_state["df"]

    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]

    st.markdown("""
    Explore your dataset visually using distributions, value counts, correlations,
    and summary statistics.
    """)

    # --- Summary Stats ---
    st.subheader("📈 Summary Statistics")
    st.dataframe(df.describe(include='all').T, use_container_width=True)

    # --- Missing values ---
    st.subheader("⚠️ Missing Values")
    st.write(df.isna().sum())

    # --- Column selector ---
    st.subheader("🎯 Select Column for Visualization")
    all_cols = list(df.columns)
    selected_col = st.selectbox("Choose a column", all_cols)

    # --- Univariate analysis ---
    if pd.api.types.is_numeric_dtype(df[selected_col]):
        st.markdown(f"### 🔢 Distribution of **{selected_col}**")
        fig = px.histogram(df, x=selected_col, nbins=30)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown(f"### 🔠 Value Counts of **{selected_col}**")
        counts = df[selected_col].value_counts().reset_index()
        counts.columns = [selected_col, "count"]
        fig = px.bar(counts, x=selected_col, y="count")
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.pie(counts, names=selected_col, values="count")
        st.plotly_chart(fig2, use_container_width=True)

    # -------------------------
    # Bivariate analysis
    # -------------------------
    st.subheader("🔄 Bivariate Analysis")

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    if len(numeric_cols) > 1:
        x_col = st.selectbox("Select X (numeric)", numeric_cols)
        y_col = st.selectbox("Select Y (numeric)", numeric_cols)
        fig3 = px.scatter(df, x=x_col, y=y_col)
        st.plotly_chart(fig3, use_container_width=True)

    # -------------------------
    # Correlation Heatmap
    # -------------------------
    st.subheader("🔥 Correlation Heatmap (Numeric Columns Only)")
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig4 = px.imshow(corr, text_auto=True, aspect="auto")
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Not enough numeric columns for correlation heatmap.")

# -------------------------
# MODELING PAGE
# -------------------------
elif page == "Modeling":
    st.title("🤖 Attrition Prediction Model")

    st.markdown("""
    Train a machine learning model to predict employee attrition.  
    Includes clean metrics, feature importance, and SHAP-style human-readable explanations.
    """)

    # Load dataframe
    if "df" not in st.session_state:
        st.warning("Please upload a dataset first in the **Data** page.")
        st.stop()

    df = st.session_state["df"]

    # Imports from src folder (note: file should be src/processing.py)
    from src.preprocessing import clean_data
    from src.modeling import train_model

    # Select target column
    target = st.selectbox("Select Target Column", df.columns)

    if st.button("Train Model"):
        df_clean = clean_data(df, target)

        # Train model
        model, metrics, feature_importances, explanations = train_model(df_clean, target)

        st.success("Model training completed!")

        # --- METRICS ---
        st.subheader("📊 Model Performance")
        st.write(f"**Accuracy:** {metrics.get('accuracy', 'N/A')}")
        st.write(f"**Precision (Attrition = Yes):** {metrics.get('precision_yes', 'N/A')}")
        st.write(f"**Recall (Attrition = Yes):** {metrics.get('recall_yes', 'N/A')}")
        st.write(f"**F1 Score (Attrition = Yes):** {metrics.get('f1_yes', 'N/A')}")
        st.write("---")
        st.write(f"**Precision (Attrition = No):** {metrics.get('precision_no', 'N/A')}")
        st.write(f"**Recall (Attrition = No):** {metrics.get('recall_no', 'N/A')}")
        st.write(f"**F1 Score (Attrition = No):** {metrics.get('f1_no', 'N/A')}")

        # --- FEATURE IMPORTANCE ---
        st.subheader("🔥 Feature Importance")
        fig_imp = px.bar(feature_importances, x="importance", y="feature", orientation="h")
        st.plotly_chart(fig_imp, use_container_width=True)

        # --- TEXT EXPLANATIONS ---
        st.subheader("📝 Model Explanation")
        st.markdown("These describe how each feature increases or decreases attrition risk:")

        for exp in explanations:
            st.write(exp)

        # --- DOWNLOAD MODEL ---
        import joblib
        joblib.dump(model, "attrition_model.joblib")

        st.download_button(
            "Download Trained Model",
            data=open("attrition_model.joblib", "rb").read(),
            file_name="attrition_model.joblib"
        )

# -----------------------------
# DASHBOARD PAGE
# -----------------------------
elif page == "Dashboard":
    st.title("📊 HR Analytics Dashboard")

    if "df" not in st.session_state:
        st.warning("Please upload a dataset in the Data page.")
        st.stop()

    df = st.session_state["df"]

    st.markdown("""
    This dashboard provides key HR insights such as attrition rate, employee distribution,
    and demographic patterns.
    """)

    # ------------ Gender Filter (SLICER) ------------
    st.subheader("🎛️ Filters")

    # Show gender filter only if Gender column exists
    if "Gender" in st.session_state["df"].columns:
        gender_filter = st.radio(
            "Filter by Gender:",
            ["All", "Male", "Female"],
            horizontal=True
        )
    else:
        gender_filter = "All"

    # Apply filter if column exists
    if gender_filter != "All" and "Gender" in df.columns:
        df = df[df["Gender"] == gender_filter]

    st.markdown("---")

    # small helper for attrition counting (handles 'Yes'/'No' and 1/0)
    def count_yes(series):
        if series.dtype == object:
            return series.isin(["Yes", "YES", "yes"]).sum()
        else:
            return ((series == 1) | (series == True)).sum()

    # ------------ KPIs ------------
    st.subheader("📌 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_emp = len(df)
        st.metric("Total Employees", total_emp)

    with col2:
        if "Attrition" in df.columns:
            yes_count = count_yes(df["Attrition"])
            attrition_rate = (yes_count / total_emp * 100) if total_emp > 0 else 0
            st.metric("Attrition Rate", f"{attrition_rate:.2f}%")

    with col3:
        avg_age = df["Age"].mean() if total_emp > 0 and "Age" in df.columns else 0
        st.metric("Average Age", f"{avg_age:.1f}")

    with col4:
        avg_income = df["MonthlyIncome"].mean() if total_emp > 0 and "MonthlyIncome" in df.columns else 0
        st.metric("Avg Monthly Income", f"{avg_income:.0f}")

    st.markdown("---")

    # ------------ Attrition by Department ------------
    st.subheader("🏢 Department-wise Attrition")

    if "Department" in df.columns and "Attrition" in df.columns:
        dept_attrition = (
            df[df["Attrition"].isin(["Yes", 1])]["Department"]
            .value_counts()
            .reset_index()
        )
        dept_attrition.columns = ["Department", "AttritionCount"]

        fig1 = px.bar(
            dept_attrition,
            x="Department",
            y="AttritionCount",
            color="Department",
            title=f"Attrition Count per Department ({gender_filter})",
        )
        st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    # ------------ Gender Distribution ------------
    st.subheader("🧑‍🤝‍🧑 Gender Distribution (Before Filter)")

    if "Gender" in st.session_state["df"].columns:
        full_gender_counts = st.session_state["df"]["Gender"].value_counts().reset_index()
        full_gender_counts.columns = ["Gender", "Count"]

        fig2 = px.pie(
            full_gender_counts,
            names="Gender",
            values="Count",
            title="Overall Gender Distribution",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")

    # ------------ Age Distribution ------------
    st.subheader(f"🎂 Age Distribution ({gender_filter})")

    if "Age" in df.columns:
        fig3 = px.histogram(
            df,
            x="Age",
            nbins=20,
            title=f"Age Distribution of Employees ({gender_filter})",
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # ------------ Job Role Breakdown ------------
    st.subheader(f"💼 Job Role Breakdown ({gender_filter})")

    if "JobRole" in df.columns:
        job_role_counts = df["JobRole"].value_counts().reset_index()
        job_role_counts.columns = ["JobRole", "Count"]

        fig4 = px.bar(
            job_role_counts,
            x="JobRole",
            y="Count",
            color="JobRole",
            title=f"Employee Count per Job Role ({gender_filter})",
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("---")

    # ------------ Monthly Income Distribution ------------
    st.subheader(f"💰 Monthly Income Distribution ({gender_filter})")

    if "MonthlyIncome" in df.columns:
        fig5 = px.histogram(
            df,
            x="MonthlyIncome",
            nbins=25,
            title=f"Distribution of Monthly Income ({gender_filter})",
        )
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("---")

    # ------------ Overtime vs Attrition ------------
    st.subheader(f"⏱️ Overtime vs Attrition ({gender_filter})")

    if "OverTime" in df.columns and "Attrition" in df.columns:
        overtime_attrition = (
            df.groupby("OverTime")["Attrition"].value_counts(normalize=True)
            .rename("Rate")
            .reset_index()
        )

        fig6 = px.bar(
            overtime_attrition[overtime_attrition["Attrition"].isin(["Yes", 1])],
            x="OverTime",
            y="Rate",
            title=f"Attrition Rate Among Overtime vs Non-Overtime Employees ({gender_filter})",
        )
        st.plotly_chart(fig6, use_container_width=True)

elif page == "About":
    st.title("ℹ️ About This Project")

    st.markdown("""
    ## 👩‍💼 HR Analytics Application

    This application is an **end-to-end HR Analytics project** built to help 
    organizations understand employee behaviour, attrition trends, and workforce patterns  
    using data-driven insights.

    ---

    ## 🎯 What This App Includes
    - 📁 **Data Upload & Preview**  
    - 🧮 **SQL Query Runner**
    - 📊 **EDA (Visual Analysis)**
    - 🤖 **Attrition Prediction Model**  
    - 📈 **Interactive Streamlit Dashboard**
    - 🟦 **Power BI Dashboard** (for business storytelling)

    ---

    ## 🧠 Machine Learning
    - Random Forest classifier  
    - SMOTE for class balance  
    - Clean metrics  
    - Text-based feature explanations  

    ---

    ## 🛠️ Tech Stack
    **Python**, **Pandas**, **NumPy**, **Plotly**, **Scikit-learn**, **SMOTE**, **SQLite**, **Streamlit**, **Power BI**

    ---

    ## 🧩 Why I Built This
    To demonstrate practical, real-world HR analytics by combining  
    **data engineering + BI dashboards + ML modeling**  
    in one interactive application.

    ---

    """)
