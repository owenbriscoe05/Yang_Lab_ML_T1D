import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    inspection_path = "./data/processed/inspections/"
    os.makedirs(inspection_path, exist_ok=True)
    
    # 1. Load Data
    master_df = load_and_merge_features()
    
    # 2. Clean impossible vitals
    master_df = clean_impossible_vitals(master_df)
    
    # 3. Double-check and filter time windows
    master_df = filter_time_windows(master_df)
    
    # Export for inspection
    master_df.to_csv(os.path.join(inspection_path, "merged_features_inspection.csv"), index=False)

    # 4. Score
    scored_df = calculate_resilience_score(master_df)

    # 5. Generate Stats
    generate_summary_statistics(scored_df)

    # 6. Save target matrix
    # only need IDs, time windows, scores for the ML model
    target_df = scored_df[["ID", "TIME_WINDOW", "RESILIENCE_SCORE"]]
    target_df.to_parquet("./data/processed/ML_targets.parquet", index=False)

    print("Complete")

def load_and_merge_features(data_dir="./data/processed/grouped_checkpoints/"):
    """
    Loads all Parquet feature files and merges them on ID and TIME_WINDOW.
    """
    print("Loading Parquet feature files...")
    
    expected_files = [
        "agap_grouped.parquet",
        "amputations_grouped.parquet", 
        "dialysis_grouped.parquet",
        "glucose_grouped.parquet",
        "hba1c_grouped.parquet",
        "hypertension_grouped.parquet",
        "neuropathy_grouped.parquet",
        "obesity_grouped.parquet",
        "potassium_grouped.parquet",
        "retinopathy_grouped.parquet",
        "underweight_grouped.parquet"
    ]
    
    merged_df = None
    
    for file in expected_files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            print(f"Reading {file}...")
            df = pd.read_parquet(file_path) 
            
            if merged_df is None:
                merged_df = df
            else:
                merged_df = pd.merge(merged_df, df, on=["ID", "TIME_WINDOW"], how="outer")
        else:
            print(f"Warning: {file} not found. Skipping...")
            
    return merged_df

def clean_impossible_vitals(df):
    """Replaces biologically impossible vital signs with NaN to prevent skewed statistics."""
    print("\nCleaning impossible vital signs...")
    
    # Humanly impossible diastolic BP (usually > 200 is a severe typo or measurement error)
    if "diastolic_mean" in df.columns:
        invalid_dias = df["diastolic_mean"] > 250
        if invalid_dias.sum() > 0:
            print(f"Dropped {invalid_dias.sum()} impossible diastolic readings (> 250).")
            df.loc[invalid_dias, "diastolic_mean"] = np.nan
            
    # Humanly impossible systolic BP 
    if "systolic_mean" in df.columns:
        invalid_sys = df["systolic_mean"] > 350
        if invalid_sys.sum() > 0:
            print(f"Dropped {invalid_sys.sum()} impossible systolic readings (> 350).")
            df.loc[invalid_sys, "systolic_mean"] = np.nan
            
    return df

def filter_time_windows(df):
    """Validates if Years 15-27 contain data before dropping them."""
    print("\nValidating Years 15-27 for data...")
    
    valid_years = [f"Year_{i}" for i in range(1, 15)]
    dropped_df = df[~df["TIME_WINDOW"].isin(valid_years)]
    
    # Safely count how many actual lab values/flags exist in the dropped years
    data_cols = [c for c in dropped_df.columns if c not in ["ID", "TIME_WINDOW"]]
    valid_data_points = dropped_df[data_cols].notna().sum().sum()
    
    print(f"Found {valid_data_points} valid clinical data points in Years 15-27.")
    if valid_data_points > 0:
        print("WARNING: You are dropping valid historical data! Check your output tables.")
        
    print("Filtering out Year 15 through Year 27...")
    starting_length = len(df)
    filtered_df = df[df["TIME_WINDOW"].isin(valid_years)].copy()
    
    print(f"Dropped {starting_length - len(filtered_df)} historical rows.")
    return filtered_df

def calculate_resilience_score(df):
    """
    Applies the 100-point hierarchical penalty logic to the merged dataframe.
    """
    print("\nCalculating 100-point continuous resilience score...")
    
    df["RESILIENCE_SCORE"] = 99.97
    
    amp_flag = df.get("amputation_flag", pd.Series(0, index=df.index)).fillna(0)
    deb_flag = df.get("neuropathy_flag", pd.Series(0, index=df.index)).fillna(0)
    
    amputation_penalty = np.where(amp_flag == 1, 22.2, 0)
    debridement_penalty = np.where((deb_flag == 1) & (amp_flag != 1), 14.8, 0)
    
    df["RESILIENCE_SCORE"] -= (amputation_penalty + debridement_penalty)
    
    under_flag = df.get("underweight_flag", pd.Series(0, index=df.index)).fillna(0)
    obese_flag = df.get("obesity_flag", pd.Series(0, index=df.index)).fillna(0)
    
    underweight_penalty = np.where(under_flag == 1, 7.4, 0)
    obese_penalty = np.where((obese_flag == 1) & (under_flag != 1), 2.96, 0)
    
    df["RESILIENCE_SCORE"] -= (underweight_penalty + obese_penalty)

    hba1c_val = df.get("hba1c_routine_mean", df.get("hba1c_poc_mean", pd.Series(0, index=df.index))).fillna(0)
    
    hba1c_conditions = [
        (hba1c_val >= 9.7),
        (hba1c_val >= 8.8) & (hba1c_val < 9.7),
        (hba1c_val >= 7.9) & (hba1c_val < 8.8),
        (hba1c_val >= 7.0) & (hba1c_val < 7.9)
    ]
    hba1c_choices = [19.31, 4.07, 2.37, 0.07]
    
    hba1c_penalty = np.select(hba1c_conditions, hba1c_choices, default=0)
    df["RESILIENCE_SCORE"] -= hba1c_penalty

    agap_max = df.get("agap_poc_max", df.get("agap_routine_mean", pd.Series(0, index=df.index))).fillna(0)
    potassium_max = df.get("potassium_poc_max", df.get("potassium_routine_mean", pd.Series(0, index=df.index))).fillna(0)
    glucose_min = df.get("glucose_poc_min", df.get("glucose_routine_mean", pd.Series(100, index=df.index))).fillna(100)
    
    acute_crisis_mask = (agap_max > 16) | (potassium_max > 6) | (glucose_min < 50)
    acute_crisis_penalty = np.where(acute_crisis_mask, 7.4, 0)
    df["RESILIENCE_SCORE"] -= acute_crisis_penalty

    standard_penalties = {
        "dialysis_flag": 22.2,
        "hypertension_flag": 6.66, 
        "retinopathy_flag": 14.8,
    }
    
    for column, penalty_weight in standard_penalties.items():
        if column in df.columns:
            event_occurred = df[column].fillna(0)
            df["RESILIENCE_SCORE"] -= (event_occurred * penalty_weight)
            
    df["RESILIENCE_SCORE"] = df["RESILIENCE_SCORE"].clip(lower=0)
    
    return df

def generate_summary_statistics(df, output_dir="./data/processed/plots/"):
    """Generates analytical plots and tables for the clinical cohort."""
    print("\nGenerating summary statistics and plots...")
    os.makedirs(output_dir, exist_ok=True)

    df_copy = df[df["RESILIENCE_SCORE"] < 99.97]
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df_copy["RESILIENCE_SCORE"], bins=40, kde=True, color='royalblue')
    plt.title("Distribution of T1D Continuous Resilience Scores", fontsize=14)
    plt.xlabel("Resilience Score (0 = Severe Failure)", fontsize=12)
    plt.ylabel("Frequency (Patient-Years)", fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(os.path.join(output_dir, "resilience_histogram.png"), dpi=300)
    plt.close()

    lab_cols = [
        "hba1c_routine_mean", "hba1c_poc_mean", 
        "glucose_routine_mean", "glucose_poc_mean", 
        "systolic_mean", "diastolic_mean"
    ]
    existing_labs = [c for c in lab_cols if c in df.columns]
    
    lab_stats = df[existing_labs].describe().T
    lab_stats.to_csv(os.path.join(output_dir, "lab_summary_statistics.csv"))
    
    htn_col = "hypertension_flag" if "hypertension_flag" in df.columns else "hypertension_pen"
    if htn_col in df.columns:
        htn_counts = df[htn_col].value_counts(dropna=False).reset_index()
        htn_counts.columns = ["Hypertension_Status", "Patient_Years"]
        htn_counts.to_csv(os.path.join(output_dir, "hypertension_counts.csv"), index=False)

    if "hba1c_routine_mean" in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="hba1c_routine_mean", y="RESILIENCE_SCORE", data=df, alpha=0.3, color='crimson')
        plt.title("Mean HbA1c vs. Resilience Score Penalty Drop-off", fontsize=14)
        plt.xlabel("Mean Routine HbA1c (%)", fontsize=12)
        plt.ylabel("Resilience Score", fontsize=12)
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(output_dir, "hba1c_vs_resilience.png"), dpi=300)
        plt.close()

    print("\nGenerating N-counts table for resilience metrics...")
    
    resilience_metrics = [
        "amputation_flag", "neuropathy_flag", "dialysis_flag", 
        "hypertension_flag", "obesity_flag", "underweight_flag", 
        "retinopathy_flag", "bp_crisis_count",
        "hba1c_routine_mean", "hba1c_poc_mean",
        "agap_poc_max", "agap_routine_mean",
        "potassium_poc_max", "potassium_routine_mean",
        "glucose_poc_min", "glucose_routine_mean"
    ]
    
    existing_metrics = [col for col in resilience_metrics if col in df.columns]
    
    n_counts = df[existing_metrics].count().reset_index()
    n_counts.columns = ["Resilience_Metric", "Valid_N_Count"]
    n_counts = n_counts.sort_values(by="Valid_N_Count", ascending=False)
    
    total_rows = len(df)
    n_counts["Percent_Populated"] = (n_counts["Valid_N_Count"] / total_rows * 100).round(2)
    
    n_counts_path = os.path.join(output_dir, "metric_n_counts.csv")
    n_counts.to_csv(n_counts_path, index=False)
    
    print(f"\n--- Data Density Report (Total Patient-Years: {total_rows}) ---")
    print(n_counts.to_string(index=False))
    print("-" * 60)

if __name__ == "__main__":
    main()