import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
from scipy.stats import ks_2samp
import warnings
import logging

# Suppress warnings and logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# Streamlit App
# -----------------------------
st.title("Blockchain Data Synthesis and Validation App")

# Step 1: Input - Upload Excel file
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

if uploaded_file is not None:
    # Step 2: Read Data
    df = pd.read_excel(uploaded_file)
    st.subheader("Original Data Preview")
    st.write(df.head())

    # Step 3: Categorize Data
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime64"]).columns.tolist()

    st.write("Categorical Columns:", categorical_cols)
    st.write("Numeric Columns:", numeric_cols)
    st.write("Datetime Columns (to drop):", datetime_cols)

    # Step 4: Drop unsuitable columns (date/time)
    df = df.drop(columns=datetime_cols, errors="ignore")

    # Step 5: Drop missing values
    df = df.dropna()
    st.subheader("Cleaned Data")
    st.write(df.head())

    # Step 6: Apply CTGAN for synthesizing
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df)
    synthesizer = CTGANSynthesizer(metadata)
    synthesizer.fit(df)
    synthetic_data = synthesizer.sample(num_rows=len(df))

    st.subheader("Synthetic Data Preview")
    st.write(synthetic_data.head())

    # Step 7: Check column synthesis & connections
    st.write("Synthetic Data Shape:", synthetic_data.shape)
    st.write("Columns match:", df.columns.equals(synthetic_data.columns))

    # Step 8: Combine synthetic + preserved non-synthetic data
    combined_data = pd.concat([df, synthetic_data], ignore_index=True)
    st.subheader("Combined Synthetic Dataset")
    st.write(combined_data.head())

    # Step 9: Perform statistical calculations
    stats = pd.DataFrame({
        "Original Mean": df[numeric_cols].mean(),
        "Synthetic Mean": synthetic_data[numeric_cols].mean(),
        "Original Std": df[numeric_cols].std(),
        "Synthetic Std": synthetic_data[numeric_cols].std(),
    })
    corr_original = df[numeric_cols].corr()
    corr_synthetic = synthetic_data[numeric_cols].corr()

    st.subheader("Statistical Comparison")
    st.write(stats)
    st.write("Original Correlation Matrix")
    st.write(corr_original)
    st.write("Synthetic Correlation Matrix")
    st.write(corr_synthetic)

    # Step 10: Validation and Graphical Representation
    st.subheader("Validation Plots")

    # Categorical Histogram
    for col in categorical_cols:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        df[col].value_counts().plot(kind="bar", ax=ax[0], title=f"Original {col}")
        synthetic_data[col].value_counts().plot(kind="bar", ax=ax[1], title=f"Synthetic {col}")
        st.pyplot(fig)

    # Numerical KS and KDE plots
    for col in numeric_cols:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        
        # KDE Plot
        sns.kdeplot(df[col], label="Original", ax=ax[0])
        sns.kdeplot(synthetic_data[col], label="Synthetic", ax=ax[0])
        ax[0].set_title(f"KDE Plot - {col}")
        ax[0].legend()

        # KS Test
        ks_stat, p_val = ks_2samp(df[col], synthetic_data[col])
        ax[1].hist(df[col], alpha=0.5, label="Original")
        ax[1].hist(synthetic_data[col], alpha=0.5, label="Synthetic")
        ax[1].set_title(f"KS Test - {col} (p={p_val:.3f})")
        ax[1].legend()

        st.pyplot(fig)
