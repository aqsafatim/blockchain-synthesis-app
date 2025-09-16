import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from io import BytesIO

# New SDV imports
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality

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
    possible_defaults = ["ID", "Timestamp"]
    valid_defaults = [col for col in possible_defaults if col in df.columns]
    columns_to_drop = st.multiselect(
        "Select columns to drop (e.g., ID, Timestamp)",
        df.columns.tolist(),
        default=valid_defaults
    )
    dropped_columns = [col for col in columns_to_drop if col in df.columns]
    df_cleaned = df.drop(columns=dropped_columns, errors="ignore")
    dropped_data = df[dropped_columns].copy()

    # Step 3: Remove columns with >20% missing values
    missing_threshold = st.slider("Missing value threshold for column removal (%)", 0, 100, 20) / 100
    missing_ratio = df_cleaned.isnull().mean()
    columns_to_keep = missing_ratio[missing_ratio <= missing_threshold].index
    df_filtered = df_cleaned[columns_to_keep]

    # Step 4: Fill missing values
    df_filtered = df_filtered.ffill().bfill()
    st.write("Columns kept for synthesis (<=20% missing):", df_filtered.columns.tolist())

    # Step 5: Create Metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df_filtered)

    # Step 6: Train CTGAN
    epochs = st.slider("Number of CTGAN epochs", 50, 500, 300)
    if st.button("Train CTGAN and Generate Synthetic Data"):
        ctgan = CTGANSynthesizer(metadata, epochs=epochs)
        with st.spinner("Training CTGAN..."):
            ctgan.fit(df_filtered)
        st.success("✅ CTGAN training complete!")

        # Step 7: Generate synthetic dataset
        synthetic_data = ctgan.sample(num_rows=len(df_filtered))
        st.write("Synthetic Data Sample:", synthetic_data.head())

        # Step 8: Combine synthetic + dropped columns
        combined_synthetic_data = synthetic_data.copy()
        for col in dropped_columns:
            if col in dropped_data.columns:
                try:
                    if col.lower() == "id":
                        combined_synthetic_data["ID"] = range(1, len(synthetic_data) + 1)
                    elif col.lower() == "timestamp" and pd.api.types.is_datetime64_any_dtype(dropped_data[col]):
                        min_time = dropped_data[col].min()
                        max_time = dropped_data[col].max()
                        time_range = (max_time - min_time).total_seconds()
                        random_seconds = np.random.uniform(0, time_range, size=len(synthetic_data))
                        synthetic_timestamps = [min_time + timedelta(seconds=sec) for sec in random_seconds]
                        combined_synthetic_data["Timestamp"] = synthetic_timestamps
                    else:
                        if pd.api.types.is_numeric_dtype(dropped_data[col]):
                            combined_synthetic_data[col] = dropped_data[col].median()
                        else:
                            combined_synthetic_data[col] = dropped_data[col].mode()[0]
                except Exception as e:
                    st.warning(f"Error processing column {col}: {e}")
                    combined_synthetic_data[col] = dropped_data[col].iloc[0]

        st.write("Combined Synthetic Data Sample:", combined_synthetic_data.head())

        # Save + Download in memory
        output_file = "Combined_Synthetic_Blockchain_Data_CTGAN.xlsx"
        output = BytesIO()
        combined_synthetic_data.to_excel(output, index=False)
        output.seek(0)
        st.download_button(
            label="Download Combined Synthetic Data",
            data=output,
            file_name=output_file,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # Step 9: Distribution Comparisons
        st.header("Distribution Comparisons")
        for col in df_filtered.columns:
            if not pd.api.types.is_numeric_dtype(df_filtered[col]):
                # Categorical
                st.subheader(f"Categorical Distribution: {col}")
                fig_cat, ax_cat = plt.subplots(figsize=(6,4))
                real_counts = df_filtered[col].value_counts(normalize=True)
                synth_counts = synthetic_data[col].value_counts(normalize=True)
                pd.DataFrame({"Real": real_counts, "Synthetic": synth_counts}).plot(kind="bar", ax=ax_cat)
                ax_cat.set_ylabel("Proportion")
                st.pyplot(fig_cat)
            else:
                # Numeric with KS test
                ks_stat, p_value = ks_2samp(df_filtered[col], synthetic_data[col])
                st.write(f"Column: {col} | KS Statistic: {ks_stat:.4f} | P-value: {p_value:.4f}")

                # Histogram
                fig_hist, ax_hist = plt.subplots(figsize=(6,4))
                ax_hist.hist(df_filtered[col], bins=30, alpha=0.5, label="Real", density=True)
                ax_hist.hist(synthetic_data[col], bins=30, alpha=0.5, label="Synthetic", density=True)
                ax_hist.legend()
                st.pyplot(fig_hist)

                # KDE
                fig_kde, ax_kde = plt.subplots(figsize=(6,4))
                sns.kdeplot(df_filtered[col], label="Real", fill=True, alpha=0.5, ax=ax_kde)
                sns.kdeplot(synthetic_data[col], label="Synthetic", fill=True, alpha=0.5, ax=ax_kde)
                ax_kde.legend()
                st.pyplot(fig_kde)

        # Step 10: Correlation Heatmaps
        st.header("Correlation Heatmaps")
        real_corr = df_filtered.corr(numeric_only=True)
        synthetic_corr = synthetic_data.corr(numeric_only=True)

        fig_corr, axes = plt.subplots(1, 2, figsize=(14, 6))
        sns.heatmap(real_corr, cmap="coolwarm", vmin=-1, vmax=1, ax=axes[0])
        axes[0].set_title("Real Data Correlation")
        sns.heatmap(synthetic_corr, cmap="coolwarm", vmin=-1, vmax=1, ax=axes[1])
        axes[1].set_title("Synthetic Data Correlation")
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