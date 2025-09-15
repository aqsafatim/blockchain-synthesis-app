import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
import streamlit as st

# Streamlit App
st.title("Blockchain Data Synthesis and Evaluation App")

# Step 1: Upload the Excel file
uploaded_file = st.file_uploader("Upload Blockchain Excel File", type=["xlsx"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_excel(uploaded_file)
    st.write("Original Data Sample:", df.head())
    st.write("Columns in dataset:", df.columns.tolist())

    # Step 2: Drop unneeded columns
    # Dynamically set default columns that exist in the DataFrame
    possible_defaults = ["ID", "Timestamp"]
    valid_defaults = [col for col in possible_defaults if col in df.columns]
    columns_to_drop = st.multiselect(
        "Select columns to drop (e.g., ID, Timestamp)",
        df.columns.tolist(),
        default=valid_defaults
    )
    dropped_columns = [col for col in columns_to_drop if col in df.columns]
    df_cleaned = df.drop(columns=dropped_columns, errors="ignore")

    # Save dropped columns for combining later
    dropped_data = df[dropped_columns].copy()

    # Step 3: Remove columns with >20% missing values
    missing_threshold = st.slider("Missing value threshold for column removal (%)", 0, 100, 20) / 100
    missing_ratio = df_cleaned.isnull().mean()
    columns_to_keep = missing_ratio[missing_ratio <= missing_threshold].index
    df_filtered = df_cleaned[columns_to_keep]

    # Step 4: Fill missing values
    df_filtered = df_filtered.fillna(method="ffill").fillna(method="bfill")
    st.write("Columns kept for synthesis (<=20% missing):", df_filtered.columns.tolist())

    # Step 5: Create Metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df_filtered)

    # Step 6: Initialize and Train CTGAN
    if st.button("Train CTGAN and Generate Synthetic Data"):
        ctgan = CTGANSynthesizer(metadata, epochs=300)
        with st.spinner("Training CTGAN..."):
            ctgan.fit(df_filtered)
        st.write("✅ CTGAN training complete!")

        # Step 7: Generate synthetic dataset
        synthetic_data = ctgan.sample(num_rows=len(df_filtered))
        st.write("Synthetic Data Sample:", synthetic_data.head())

        # Step 8: Combine synthetic and non-synthetic data
        st.header("Combining Synthetic and Non-Synthetic Data")
        combined_synthetic_data = synthetic_data.copy()

        # Handle dropped columns
        for col in dropped_columns:
            if col in dropped_data.columns:
                if col.lower() == "id":
                    # Generate sequential IDs
                    combined_synthetic_data["ID"] = range(1, len(synthetic_data) + 1)
                elif col.lower() == "timestamp" and pd.api.types.is_datetime64_any_dtype(dropped_data[col]):
                    # Generate timestamps within the original range
                    min_time = dropped_data[col].min()
                    max_time = dropped_data[col].max()
                    time_range = (max_time - min_time).total_seconds()
                    random_seconds = np.random.uniform(0, time_range, size=len(synthetic_data))
                    synthetic_timestamps = [min_time + timedelta(seconds=sec) for sec in random_seconds]
                    combined_synthetic_data["Timestamp"] = synthetic_timestamps
                else:
                    # Placeholder for other dropped columns (e.g., mode for categorical, median for numeric)
                    if pd.api.types.is_numeric_dtype(dropped_data[col]):
                        combined_synthetic_data[col] = dropped_data[col].median()
                    else:
                        combined_synthetic_data[col] = dropped_data[col].mode()[0]

        st.write("Combined Synthetic Data Sample:", combined_synthetic_data.head())
        
        # Save combined synthetic dataset
        output_file = "Combined_Synthetic_Blockchain_Data_CTGAN.xlsx"
        combined_synthetic_data.to_excel(output_file, index=False)
        st.download_button(
            label="Download Combined Synthetic Data",
            data=combined_synthetic_data.to_csv(index=False),
            file_name="Combined_Synthetic_Blockchain_Data_CTGAN.csv",
            mime="text/csv"
        )
        st.write(f"✅ Combined synthetic data saved as: {output_file}")

        # Step 9: Distribution Comparison
        st.header("Distribution Comparisons")
        for col in df_filtered.columns:
            # Categorical columns
            if not pd.api.types.is_numeric_dtype(df_filtered[col]):
                st.subheader(f"Categorical Distribution: {col}")
                fig_cat, ax_cat = plt.subplots(figsize=(6,4))
                real_counts = df_filtered[col].value_counts(normalize=True)
                synth_counts = synthetic_data[col].value_counts(normalize=True)
                pd.DataFrame({"Real": real_counts, "Synthetic": synth_counts}).plot(kind="bar", ax=ax_cat)
                ax_cat.set_title(f"Categorical Distribution: {col}")
                ax_cat.set_ylabel("Proportion")
                st.pyplot(fig_cat)
            
            # Numeric columns
            if pd.api.types.is_numeric_dtype(df_filtered[col]):
                # KS Test
                ks_stat, p_value = ks_2samp(df_filtered[col], synthetic_data[col])
                st.write(f"Column: {col} | KS Statistic: {ks_stat:.4f} | P-value: {p_value:.4f}")

                # Histogram
                fig_hist, ax_hist = plt.subplots(figsize=(6,4))
                ax_hist.hist(df_filtered[col], bins=30, alpha=0.5, label="Real", density=True)
                ax_hist.hist(synthetic_data[col], bins=30, alpha=0.5, label="Synthetic", density=True)
                ax_hist.set_title(f"Histogram: {col}")
                ax_hist.set_xlabel(col)
                ax_hist.set_ylabel("Density")
                ax_hist.legend()
                st.pyplot(fig_hist)

                # KDE
                fig_kde, ax_kde = plt.subplots(figsize=(6,4))
                sns.kdeplot(df_filtered[col], label="Real", fill=True, alpha=0.5, ax=ax_kde)
                sns.kdeplot(synthetic_data[col], label="Synthetic", fill=True, alpha=0.5, ax=ax_kde)
                ax_kde.set_title(f"KDE Density Plot: {col}")
                ax_kde.set_xlabel(col)
                ax_kde.set_ylabel("Density")
                ax_kde.legend()
                st.pyplot(fig_kde)

                # Boxplot
                fig_box, ax_box = plt.subplots(figsize=(6,4))
                sns.boxplot(data=[df_filtered[col], synthetic_data[col]], orient="h", ax=ax_box)
                ax_box.set_yticks([0,1])
                ax_box.set_yticklabels(["Real", "Synthetic"])
                ax_box.set_title(f"Boxplot: {col}")
                st.pyplot(fig_box)

        # Step 10: Correlation Heatmap Comparison
        st.header("Correlation Heatmaps")
        real_corr = df_filtered.corr(numeric_only=True)
        synthetic_corr = synthetic_data.corr(numeric_only=True)

        fig_corr, axes = plt.subplots(1, 2, figsize=(14, 6))
        sns.heatmap(real_corr, annot=False, cmap="coolwarm", vmin=-1, vmax=1, ax=axes[0])
        axes[0].set_title("Real Data Correlation Heatmap")
        sns.heatmap(synthetic_corr, annot=False, cmap="coolwarm", vmin=-1, vmax=1, ax=axes[1])
        axes[1].set_title("Synthetic Data Correlation Heatmap")
        st.pyplot(fig_corr)

        # Step 11: Statistical Summary
        st.header("Statistical Summary")
        stats_summary = pd.DataFrame({
            "Real Mean": df_filtered.mean(numeric_only=True),
            "Synthetic Mean": synthetic_data.mean(numeric_only=True),
            "Real Std": df_filtered.std(numeric_only=True),
            "Synthetic Std": synthetic_data.std(numeric_only=True)
        })
        st.dataframe(stats_summary)

        # Mean & Std Visualizations
        fig_mean = stats_summary[["Real Mean", "Synthetic Mean"]].plot(kind="bar", figsize=(10,5)).get_figure()
        plt.title("Mean Comparison (Real vs Synthetic)")
        plt.ylabel("Mean Value")
        st.pyplot(fig_mean)

        fig_std = stats_summary[["Real Std", "Synthetic Std"]].plot(kind="bar", figsize=(10,5)).get_figure()
        plt.title("Standard Deviation Comparison (Real vs Synthetic)")
        plt.ylabel("Standard Deviation")
        st.pyplot(fig_std)

        # Correlation Difference
        corr_diff = (real_corr - synthetic_corr).abs()
        st.header("Correlation Difference Heatmap")
        fig_diff, ax_diff = plt.subplots(figsize=(8,6))
        sns.heatmap(corr_diff, annot=True, cmap="Reds", vmin=0, vmax=1, ax=ax_diff)
        ax_diff.set_title("Correlation Difference Heatmap (Real vs Synthetic)")
        st.pyplot(fig_diff)