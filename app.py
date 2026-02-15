import os
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# ------------------ PATHS ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "breast_cancer.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="ML Assignment 2",
    layout="wide"
)

st.title("Classification Models Evaluation Dashboard")

# ------------------ LOAD BASE DATASET ------------------
df_base = pd.read_csv(DATA_PATH)

# ------------------ SIDEBAR ------------------
st.sidebar.header("Model Settings")

model_name = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Regression", "Decision Tree", "KNN",
     "Naive Bayes", "Random Forest", "XGBoost"]
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV (for prediction)",
    type="csv"
)

# ------------------ SECTION 1: DATASET OVERVIEW ------------------
st.header("Dataset Overview")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Dataset Shape")
    st.write(f"Rows: **{df_base.shape[0]}**")
    st.write(f"Columns: **{df_base.shape[1]}**")

with col2:
    st.subheader("Target Distribution")
    target_counts = df_base["target"].value_counts()
    st.bar_chart(target_counts)

st.subheader("Sample Data (First 5 Rows)")
st.dataframe(df_base.head())

# ------------------ SECTION 2: FEATURE CORRELATION ------------------
st.header("Feature Correlation Heatmap")

# Use only first 10 features to keep it readable
selected_features = df_base.columns[:10]

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(
    df_base[selected_features].corr(),
    cmap="coolwarm",
    annot=False,
    ax=ax
)

st.pyplot(fig)

# ------------------ SECTION 3: MODEL EVALUATION ------------------
st.header("Model Evaluation")

if uploaded_file:
    df_test = pd.read_csv(uploaded_file)

    X = df_test.drop("target", axis=1)
    y = df_test["target"]

    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    X_scaled = scaler.transform(X)

    model = joblib.load(os.path.join(MODEL_DIR, f"{model_name}.pkl"))
    y_pred = model.predict(X_scaled)

    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    st.write(cm)

else:
    st.info(
        "Upload a CSV file from the sidebar to see model predictions and metrics."
    )
