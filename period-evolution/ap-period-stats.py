### Import libraries
import os
import pandas as pd
import numpy as np
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# Set working directory
data_dir = os.getcwd()

# Create directory for results
results_dir = data_dir + "/results"
isExist = os.path.exists(results_dir)
if not isExist:
    os.makedirs(results_dir)

# Find all CSV files
csv_files = glob.glob(os.path.join(data_dir, "*.csv"))

# Define time bins
time_bins = list(range(50, 1250, 200))  # 50 to 1250 minutes in 200-minute bins
time_labels = [f"{time_bins[i]}-{time_bins[i+1]}" for i in range(len(time_bins) - 1)]

# List to store data
data = []

# Process all CSV files
for file in csv_files:
    if 'periods' in file:
        df = pd.read_csv(file)

    # Multiply time column by 10
    df["Time"] = df["Time"] * 10 # Multiply time values by 10

    # Determine condition and region from filename
    condition = "DAPT" if "DAPT" in file else "CTRL"
    region = "arrival" if "AR" in file else "origin"

    # Assign time bins
    df["time_bin"] = pd.cut(df["Time"], bins=time_bins, labels=time_labels, include_lowest=True, right=False)

    # Convert to long format
    df_melted = df.melt(id_vars=["Time", "time_bin"], var_name="sample_id", value_name="period_value")
    df_melted["condition"] = condition
    df_melted["region"] = region

    data.append(df_melted)

# Concatenate data
df_all = pd.concat(data, ignore_index=True)

# Compute mean periods per sample, time bin, and condition
df_means = df_all.groupby(["time_bin", "condition", "region", "sample_id"], as_index=False, observed=False)["period_value"].mean()

# Extract CTRL samples and pivot the dataframe for ratio analysis
df_ctrl = df_means[df_means["condition"] == "CTRL"]
df_ctrl_pivot = df_ctrl.pivot(index=["time_bin", "sample_id"], columns="region", values="period_value").reset_index()

# Compute arrival/origin (AP) ratio in CTRL samples
df_ctrl_pivot["ap_ratio_ctrl"] = df_ctrl_pivot["arrival"] / df_ctrl_pivot["origin"]

# Calculate mean and standard deviation of arrival/origin (AP) ratio for each time bin
ratio_summary = df_ctrl_pivot.groupby("time_bin", observed=False)["ap_ratio_ctrl"].agg(["mean", "std"]).reset_index()

# Extract DAPT samples and pivot dataframe for analysis
df_dapt = df_means[df_means["condition"] == "DAPT"]
df_dapt_pivot = df_dapt.pivot(index=["time_bin", "sample_id"], columns="region", values="period_value").reset_index()

# Compute arrival/origin (AP) ratio in DAPT samples
df_dapt_pivot["ap_ratio_dapt"] = df_dapt_pivot["arrival"] / df_dapt_pivot["origin"]

# Merge dataframes for comparisons and add anterior-posterior ratios for both conditions
comparison_df = df_dapt_pivot.merge(ratio_summary, on="time_bin", suffixes=("_dapt", "_ctrl"))
comparison_df = comparison_df.merge(df_ctrl_pivot[["time_bin", "sample_id", "ap_ratio_ctrl"]], on=["time_bin", "sample_id"], how="left")

### Statistical analysis

# IQR code (prepare columns to fill)
comparison_df["iqr"] = np.nan 
comparison_df["iqr_threshold"] = np.nan   # Q3 + 1.5 * IQR
comparison_df["outlier"] = np.nan 
comparison_df["outlier"] = comparison_df["outlier"].astype("object")

# Detect outliers by calculating the IQR for each time-bin
for time_bin in comparison_df["time_bin"].unique():
    mask = comparison_df["time_bin"] == time_bin
    ctrl_values = df_ctrl_pivot[df_ctrl_pivot["time_bin"] == time_bin]["ap_ratio_ctrl"].dropna()

    if len(ctrl_values) > 3:
        q1 = ctrl_values.quantile(0.25)
        q3 = ctrl_values.quantile(0.75)
        iqr = q3 - q1
        threshold = q3 + 1.5 * iqr

        comparison_df.loc[mask, "iqr"] = iqr
        comparison_df.loc[mask, "iqr_threshold"] = threshold

        # Detect IQR outliers (mark as outliers if threshold is exceeded)
        comparison_df.loc[mask, "outlier"] = comparison_df.loc[mask, "ap_ratio_dapt"] > threshold

# Ensure outlier column type is now Boolean
comparison_df["outlier"] = comparison_df["outlier"].astype("boolean")

# Save results
comparison_df.to_csv(results_dir + "/" + "AP_period_comparison.csv", index=False)

# Filter dataframe by significance

filtered_dataframe = comparison_df.loc[(comparison_df["outlier"] == True)]

# Save results
filtered_dataframe.to_csv(results_dir + "/" + "AP_period_comparison_outliers.csv", index=False)
