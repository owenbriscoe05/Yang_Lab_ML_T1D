# Quick script to locate the absolute last day of data in the EMR
# This is crucial for definine drug durations in resilience_features, for drugs without an end date

import pandas as pd

def find_ehr_extraction_date():
    # Point this to your cleanest, most reliable file (Encounters is best)
    file_path = "./data/processed/T1D_encounters_clean.csv"
    date_cols_to_check = ["ADMIT_DATE_OFFSET", "DISCHARGE_DATE_OFFSET"]
    
    print(f"Scanning {file_path} to find the global maximum date...")
    
    global_max_date = pd.NaT
    
    # usecols is the secret here: it only loads the 2 date columns, saving massive RAM
    chunk_iterator = pd.read_csv(file_path, usecols=date_cols_to_check, chunksize=500000)
    
    for i, chunk in enumerate(chunk_iterator):
        for col in date_cols_to_check:
            # Convert to datetime. 'coerce' turns messy strings into NaT safely
            dates = pd.to_datetime(chunk[col], errors="coerce")
            
            # Find the max date in this specific chunk
            chunk_max = dates.max()
            
            # Update the global maximum if this chunk has a newer date
            if pd.isna(global_max_date) or (not pd.isna(chunk_max) and chunk_max > global_max_date):
                
                # SANITY CHECK: EMRs are notorious for typos (e.g., a doctor typing the year 2099)
                # We know the pull was in 2024, so we ignore any impossible future dates
                if chunk_max.year <= 2024:
                    global_max_date = chunk_max
                    
        if i % 10 == 0:
            print(f"Processed chunk {i}...")

    print("-" * 30)
    print(f"SUCCESS: The latest valid date in the dataset is: {global_max_date.strftime('%Y-%m-%d')}")
    print("Use this date as your study right-censor cutoff.")
    print("-" * 30)
    
    return global_max_date

if __name__ == "__main__":
    find_ehr_extraction_date()