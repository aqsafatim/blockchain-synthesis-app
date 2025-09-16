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
from datetime import timedelta
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

    # Step 3: Drop optional unneeded columns
    possible_defaults = ["ID", "Timestamp"]
    valid_defaults = [col for col in possible_defaults if col in df.columns]
    columns_to_drop = st.multiselect(
        "Select columns to drop (e.g., ID, Timestamp)",
        df.columns.tolist(),
        default=valid_defaults
    )
    dropped_columns = [col for col in columns_to_drop if col in df.columns]
    dropped_data = df[dropped_columns].copy()
    df = df.drop(columns=dropped_columns, errors="ignore")

    # Step 4: Remove columns with >20% missing values
    missing_threshold = st.slider("Missing value threshold for column removal (%)", 0, 100, 20) / 100
    missing_ratio = df.isnull().mean()
    columns_to_keep = missing_ratio[missing_ratio <= missing_threshold].index
    df = df[columns_to_keep]

    # Step 5: Fill missing values
    df = df.fillna(method="ffill").fillna(method="bfill")
    st.subheader("Cleaned Data (Ready for CTGAN)")
    st.write(df.head())

    # Step 6: Categorize Data
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    st.write("Categorical Columns:", categorical_cols)
    st.write("Numeric Columns:", numeric_cols)

    # Step 7: Apply CTGAN for synthesizing
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df)

    if st.button("Train CTGAN and Generate Synthetic Data"):
        synthesizer = CTGANSynthesizer(metadata, epochs=300)
        with st.spinner("Training CTGAN..."):
            synthesizer.fit(df)
        st.success("✅ CTGAN training complete!")

        synthetic_data = synthesizer.sample(num_rows=len(df))
        st.subheader("Synthetic Data Preview")
        st.write(synthetic_data.head())

        # Step 8: Re-add dropped columns (ID, Timestamp etc.)
        combined_data = synthetic_data.copy()
        for col in dropped_columns:
            if col in dropped_data.columns:
                if col.lower() == "id":
                    combined_data["ID"] = range(1, len(synthetic_data) + 1)
                elif col.lower() == "timestamp" and pd.api.types.is_datetime64_any_dtype(dropped_data[col]):
                    min_time = dropped_data[col].min()
                    max_time = dropped_data[col].max()
                    time_range = (max_time - min_time).total_seconds()
                    random_seconds = np.random.uniform(0, time_range, size=len(synthetic_data))
                    synthetic_timestamps = [min_time + timedelta(seconds=sec) for sec in random_seconds]
                    combined_data["Timestamp"] = synthetic_timestamps
                else:
                    if pd.api.types.is_numeric_dtype(dropped_data[col]):
                        combined_data[col] = dropped_data[col].median()
                    else:
                        combined_data[col] = dropped_data[col].mode()[0]

        st.subheader("Combined Synthetic Dataset")
        st.write(combined_data.head())

        # Download button
        st.download_button(
            label="Download Combined Synthetic Data",
            data=combined_data.to_csv(index=False),
            file_name="Combined_Synthetic_Blockchain_Data_CTGAN.csv",
            mime="text/csv"
        )

        # Step 9: Statistical Comparison
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

        # Mean & Std Plots
        fig_mean = stats[["Original Mean", "Synthetic Mean"]].plot(kind="bar", figsize=(10,5)).get_figure()
        plt.title("Mean Comparison (Original vs Synthetic)")
        st.pyplot(fig_mean)

        fig_std = stats[["Original Std", "Synthetic Std"]].plot(kind="bar", figsize=(10,5)).get_figure()
        plt.title("Std Deviation Comparison (Original vs Synthetic)")
        st.pyplot(fig_std)

        # Correlation Heatmaps
        st.subheader("Correlation Heatmaps")
        fig_corr, axes = plt.subplots(1, 2, figsize=(14, 6))
        sns.heatmap(corr_original, annot=False, cmap="coolwarm", vmin=-1, vmax=1, ax=axes[0])
        axes[0].set_title("Original Data Correlation Heatmap")
        sns.heatmap(corr_synthetic, annot=False, cmap="coolwarm", vmin=-1, vmax=1, ax=axes[1])
        axes[1].set_title("Synthetic Data Correlation Heatmap")
        st.pyplot(fig_corr)

        # Correlation Difference
        corr_diff = (corr_original - corr_synthetic).abs()
        st.subheader("Correlation Difference Heatmap")
        fig_diff, ax_diff = plt.subplots(figsize=(8,6))
        sns.heatmap(corr_diff, annot=True, cmap="Reds", vmin=0, vmax=1, ax=ax_diff)
        st.pyplot(fig_diff)

        # Step 10: Validation Plots
        st.subheader("Validation Plots")

        # Categorical Distribution
        for col in categorical_cols:
            fig, ax = plt.subplots(figsize=(6, 4))
            real_counts = df[col].value_counts(normalize=True)
            synth_counts = synthetic_data[col].value_counts(normalize=True)
            pd.DataFrame({"Real": real_counts, "Synthetic": synth_counts}).plot(kind="bar", ax=ax)
            ax.set_title(f"Categorical Distribution: {col}")
            st.pyplot(fig)

        # Numeric Validation
        for col in numeric_cols:
            ks_stat, p_val = ks_2samp(df[col], synthetic_data[col])
            st.write(f"Column: {col} | KS Statistic: {ks_stat:.4f} | P-value: {p_val:.4f}")

            fig, axs = plt.subplots(1, 3, figsize=(15, 4))

            # Histogram
            axs[0].hist(df[col], bins=30, alpha=0.5, label="Original", density=True)
            axs[0].hist(synthetic_data[col], bins=30, alpha=0.5, label="Synthetic", density=True)
            axs[0].set_title(f"Histogram: {col}")
            axs[0].legend()

            # KDE
            sns.kdeplot(df[col], label="Original", fill=True, alpha=0.5, ax=axs[1])
            sns.kdeplot(synthetic_data[col], label="Synthetic", fill=True, alpha=0.5, ax=axs[1])
            axs[1].set_title(f"KDE: {col}")
            axs[1].legend()

            # Boxplot
            sns.boxplot(data=[df[col], synthetic_data[col]], orient="h", ax=axs[2])
            axs[2].set_yticks([0, 1])
            axs[2].set_yticklabels(["Original", "Synthetic"])
            axs[2].set_title(f"Boxplot: {col}")

            st.pyplot(fig)
        # -----------------------------
        # Step 11: Summary Report
        # -----------------------------
        st.subheader("📊 Summary of Real vs Synthetic Data Differences")

        # Numerical summary (Mean, Std, KS test results)
        summary_list = []
        for col in numeric_cols:
            ks_stat, p_val = ks_2samp(df[col], synthetic_data[col])
            summary_list.append({
                "Column": col,
                "Original Mean": df[col].mean(),
                "Synthetic Mean": synthetic_data[col].mean(),
                "Mean Diff": abs(df[col].mean() - synthetic_data[col].mean()),
                "Original Std": df[col].std(),
                "Synthetic Std": synthetic_data[col].std(),
                "Std Diff": abs(df[col].std() - synthetic_data[col].std()),
                "KS Statistic": ks_stat,
                "P-Value": p_val
            })

        summary_df = pd.DataFrame(summary_list)
        st.dataframe(summary_df.style.format({
            "Original Mean": "{:.4f}",
            "Synthetic Mean": "{:.4f}",
            "Mean Diff": "{:.4f}",
            "Original Std": "{:.4f}",
            "Synthetic Std": "{:.4f}",
            "Std Diff": "{:.4f}",
            "KS Statistic": "{:.4f}",
            "P-Value": "{:.4f}"
        }))

        # Textual interpretation
        st.markdown("### 📝 Interpretation")
        st.markdown("""
        - **Mean & Std Differences** show how close synthetic values are to the real distribution.  
        - **KS Statistic & P-Value** test whether the synthetic distribution matches the original:
            - Small KS & high P-value → synthetic is very similar to real.  
            - Large KS & low P-value → synthetic is significantly different.  
        - **Correlation Difference Heatmap** already shows dependency shifts between features.  
        - **Categorical distributions** (above) highlight how well categorical balance is preserved.  
        """)

       