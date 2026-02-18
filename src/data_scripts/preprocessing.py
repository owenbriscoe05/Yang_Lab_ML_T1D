import pandas as pd
import os
import gc  # Garbage Collector interface

raw_data_path = "./data/raw/" 

def main():

    # concat_datasets()

    # Process files one by one to prevent memory faults
    print("--- Processing Demographics ---")
    usable_ids = process_demographics()
    
    print("\n--- Processing Encounters ---")
    # process_encounters(usable_ids)
    
    print("\n--- Processing Labs ---")
    # hba1c = process_labs(usable_ids)
    
    print("\n--- Processing Meds ---")
    process_meds(usable_ids)

    # print(hba1c)

    # if ((hba1c["raw_unit"] != "%") or (2 >= hba1c["raw_unit"] >= 20)):
    #     hba1c["raw_"]

def concat_datasets():
    prescriptions_raw = pd.concat([pd.read_csv("./data/raw/T1D_prescriptions_1.csv", header=0),
                                      pd.read_csv("./data/raw/T1D_prescriptions_2.csv", header=0), pd.read_csv("./data/raw/T1D_prescriptions_3.csv", header=0),
                                      pd.read_csv("./data/raw/T1D_prescriptions_4.csv", header=0)], ignore_index=True)
    
    # confounding druglist for inclusion in the model (tend to be beneficial for T1D)
    lisinopril = prescriptions_raw[prescriptions_raw["RAW_RX_MED_NAME"].str.contains("lisinopril", case=False, na=False)].copy()
    losartan = prescriptions_raw[prescriptions_raw['RAW_RX_MED_NAME'].str.contains("losartan", case=False, na=False)].copy()
    metformin = prescriptions_raw[prescriptions_raw["RAW_RX_MED_NAME"].str.contains("metformin", case=False, na=False)].copy()
    ozempic = prescriptions_raw[prescriptions_raw["RAW_RX_MED_NAME"].str.contains("ozempic", case=False, na=False)].copy()
    cgm = prescriptions_raw[prescriptions_raw["RAW_RX_MED_NAME"].str.contains("CGM|DEXCOM|GLUCOSE MONITOR", case=False, na=False)].copy()
    insulin_pump = prescriptions_raw[prescriptions_raw["RAW_RX_MED_NAME"].str.contains("PUMP|OMNIPOD|TANDEM", case=False, na=False)].copy()

    # saboteur meds (tend to be detrimental for T1D)
    steroids = prescriptions_raw[prescriptions_raw["RAW_RX_MED_NAME"].str.contains("prednisone|dexamethasone|hydrocortisone", case=False, na=False)].copy()
    immunosuppressants = prescriptions_raw[prescriptions_raw["RAW_RX_MED_NAME"].str.contains("tacrolimus|cyclosporine|methotrexate", case=False, na=False)].copy()

    print(f"made it prior to confound_drug concatenation")

    confound_drugs = pd.concat([lisinopril, losartan, metformin, ozempic, cgm, insulin_pump, steroids, immunosuppressants])

    print(f"made it past concatenation, head(5): {confound_drugs.head(5)}")
    T1D_procedures_raw = pd.concat([pd.read_csv("./data/raw/T1D_procedures_1.csv", header=0),
                                      pd.read_csv("./data/raw/T1D_procedures_2.csv", header=0), pd.read_csv("./data/raw/T1D_procedures_3.csv", header=0),
                                      pd.read_csv("./data/raw/T1D_procedures_4.csv", header=0)], ignore_index=True)
    T1D_vitals_raw = pd.concat([pd.read_csv("./data/raw/T1D_vitals_1.csv", header=0), pd.read_csv("./data/raw/T1D_vitals_2.csv", header=0)], ignore_index=True)

    confound_drugs.to_csv("./data/processed/confounding_drugs_external.csv", index=False)
    prescriptions_raw.to_csv("./data/processed/prescribed_drugs.csv", index=False)
    T1D_procedures_raw.to_csv("./data/processed/procedures.csv", index=False)
    T1D_vitals_raw.to_csv("./data/processed/vitals.csv", index=False)

def process_demographics():
    df = pd.read_csv(f"{raw_data_path}T1D_demographics_raw.csv", header=0)
    
    df = df.drop(columns=["BIOBANK_FLAG"])
    df["BIRTH_YEAR_OFFSET"] = pd.to_numeric(df["BIRTH_YEAR_OFFSET"], errors='coerce')
    df = df[df["BIRTH_YEAR_OFFSET"] <= 2007]
    df = df.dropna(subset=["ID", "SEX", "BIRTH_DATE_OFFSET"])
    print(df.head())
    # save back to disk if want to keep these changes
    df.to_csv(f"./data/processed/T1D_demographics_clean.csv", index=False)
    usable_ids = set(df["ID"].unique())
    
    # Free memory, avoid running into segmentation faults
    del df
    gc.collect()

    return usable_ids

def process_encounters(ids):
    # Specify dtypes to fix the Warning and save memory
    # Columns 4, 11, 12, 13 were causing issues (admit_time, payer_types, facility_type)
    dtype_map = {
        "4": str, "11": str, "12": str, "13": str
    }
    
    # Load (using header=0 since your raw files likely have headers from the previous step)
    df = pd.read_csv(f"{raw_data_path}T1D_encounters_raw.csv", header=0, dtype=str) 
    
    
    df = df.drop(columns=["FACILITY_LOCATION", "PAYER_TYPE_PRIMARY", "PAYER_TYPE_SECONDARY"])
    df = df[df["ID"].isin(ids)]

    er_visit = df[df["ENC_TYPE"].str.contains("ER", case=False, na=False)].copy()
    inpatient_visit = df[df["ENC_TYPE"].str.contains("EI|OC|EI", case=False, na=False)].copy()

    special_encs = pd.concat([er_visit, inpatient_visit])

    
    print(df.head())
    df.to_csv(f"./data/processed/T1D_encounters_clean.csv", index=False)
    special_encs.to_csv("./data/processed/special_encounters.csv")
    del df
    gc.collect()

def process_labs(ids):
    print("  Loading labs in chunks...")
    
    # 1. Define the columns upfront (saves memory vs inferring them)
    # cols = ["lab_result_cm_id", "id", "encounterid", "specimen_source", "lab_loinc",
    #         "lab_result_source", "lab_loinc_source", "lab_px_type", "lab_order_date_offset",
    #         "lab_order_date_offset_relative", "specimen_date_offset", "specimen_date_offset_relative",
    #         "result_date_offset", "result_date_offset_relative", "result_qual", "result_num",
    #         "result_modifier", "result_unit", "norm_range_low", "norm_modifier_low", "norm_range_high",
    #         "norm_modifier_high", "abn_ind", "raw_lab_name", "raw_result", "raw_unit", "masked_source"]

    filtered_chunks = []
    
    # 2. Iterate through the file in blocks of 500,000 rows
    #    chunksize return an iterator, not the whole dataframe
    reader = pd.read_csv(
        f"{raw_data_path}T1D_labs_raw.csv", 
        dtype=str, 
        header=0,        # Assumes you have a header from the get_raw step
        chunksize=500000 
    )

    for i, chunk in enumerate(reader):
        # A. Filter immediately
        kept_rows = chunk[chunk["ID"].isin(ids)]
        
        # B. If we found relevant rows, save them
        if not kept_rows.empty:
            filtered_chunks.append(kept_rows)
            
        if i % 10 == 0:
            print(f"    Processed chunk {i}...", end="\r")

    print(f"    Chunks processed. Merging {len(filtered_chunks)} relevant pieces...")
    
    # 3. Concatenate only the rows that mattered
    if not filtered_chunks:
        print("WARNING: No labs matched your patient IDs!")
        return pd.DataFrame(cols=0)
        
    df = pd.concat(filtered_chunks, ignore_index=True)
    
    # Filter for HbA1c, blood glucose, etc
    # Using str.contains is safer than exact match for "HBA1C"
    hba1c = df[df["RAW_LAB_NAME"].str.contains("A1C|HBA1C", case=False, na=False)].copy()
    blood_glucose = df[df["RAW_LAB_NAME"].str.contains("glucose", case=False, na=False)].copy()
    # OTHER USEFUL MEASURES (kidney health, more DKA events, etc)
    # already manually checked this one for confounding names, none present
    anion_gap = df[df["RAW_LAB_NAME"].str.contains("AGAP", case=False, na=False)].copy()
    creatinine = df[df["RAW_LAB_NAME"].str.contains("creatinine", case=False, na=False)].copy()
    has_potassium = df["RAW_LAB_NAME"].str.contains("potassium", case=False, na=False)
    pot_confound = df["RAW_LAB_NAME"].str.contains("gas|urine", case=False, na=False)
    potassium = df[has_potassium & ~pot_confound].copy()

    resilience_metrics = pd.concat([hba1c, blood_glucose, anion_gap, creatinine, potassium])
    
    print(f"Found {len(hba1c)} valid HbA1c records.")
    
    df.to_csv(f"./data/processed/T1D_labs_clean.csv", index=False)
    resilience_metrics.to_csv(f"./data/processed/resilience_metrics.csv", index=False)
    
    del df
    gc.collect()
    
    return hba1c

def process_meds(ids):
    
    # cols = ["medadminid", "id", "encounterid", "prescribingid", "medadmin_start_date_offset",
    #               "medadmin_start_date_offset_relative", "medadmin_stop_date_offset",
    #               "medadmin_stop_date_offset_relative", "medadmin_type", "medadmin_code",
    #               "medadmin_dose_admin", "medadmin_dose_admin_unit", "medadmin_route", "medadmin_source",
    #               "raw_medadmin_med_name", "masked_source"]
    
    filtered_chunks = []
    
    # 2. Iterate through the file in blocks of 500,000 rows
    #    chunksize return an iterator, not the whole dataframe
    reader = pd.read_csv(
        f"{raw_data_path}T1D_meds_raw.csv", 
        dtype=str, 
        header=0,        # Assumes you have a header from the get_raw step
        chunksize=500000 
    )

    for i, chunk in enumerate(reader):
        # A. Filter immediately
        kept_rows = chunk[chunk["ID"].isin(ids)]
        
        # B. If we found relevant rows, save them
        if not kept_rows.empty:
            filtered_chunks.append(kept_rows)
            
        if i % 10 == 0:
            print(f"    Processed chunk {i}...", end="\r")

    print(f"    Chunks processed. Merging {len(filtered_chunks)} relevant pieces...")
    
    # 3. Concatenate only the rows that mattered
    if not filtered_chunks:
        print("WARNING: No med files matched your patient IDs!")
        return pd.DataFrame(columns=cols)
        
    df = pd.concat(filtered_chunks, ignore_index=True)

    df = df.dropna(subset=["ENCOUNTERID", "MEDADMIN_START_DATE_OFFSET", "MEDADMIN_DOSE_ADMIN", "MEDADMIN_CODE"])

    # searching for drugs indicative of a serious T1D condition (DKA, hypoglycemia, kidney failure, etc)
    # dextrose 50 for "dextrose 50%, or dextrose 500mg/ml (a high amount of dextrose indicative of hypoglycemia)"
    dextrose = df[df["RAW_MEDADMIN_MED_NAME"].str.contains("dextrose 50", case=False, na=False)].copy()
    # glucagon is also potentially given to hypoglycemic patients
    glucagon = df[df["RAW_MEDADMIN_MED_NAME"].str.contains("glucagon|baqsimi")].copy()
    # insulin and/or potassium delivered intraveneously is a good indication of a DKA event
    insulin_iv = df[df["RAW_MEDADMIN_MED_NAME"].str.contains("insulin&infusion", case=False, na=False)].copy()
    potassium_iv = df[df["RAW_MEDADMIN_MED_NAME"].str.contains("potassium&iv", case=False, na=False)].copy()

    # saboteur meds (tend to be detrimental for T1D)
    steroids = df[df["RAW_MEDADMIN_MED_NAME"].str.contains("prednisone|dexamethasone|hydrocortisone", case=False, na=False)].copy()
    immunosuppressants = df[df["RAW_MEDADMIN_MED_NAME"].str.contains("tacrolimus|cyclosporine|methotrexate", case=False, na=False)].copy()

    indicative_drugs = pd.concat([dextrose, glucagon, insulin_iv, potassium_iv, steroids, immunosuppressants])
    
    
    print(df.head())
    df.to_csv(f"./data/processed/T1D_meds_clean.csv", index=False)
    indicative_drugs.to_csv(f"./data/processed/indicative_drugs.csv", index=False)
    del df
    gc.collect()

if __name__ == "__main__":
    main()