import pandas as pd
import pyreadr
import os

GET_PATH_1 = "/u/project/xyang123/rainyliu/EHR_OneFlorida_V1/intermediate_data/"
GET_PATH_2 = "/u/project/xyang123/shared/datasets/EHR_data/OneFlorida/V1/"
WRITE_PATH = "./data/raw/"

print("Loading patient list...")
try:
    T1D_patients_r = pyreadr.read_r(f"{GET_PATH_1}/T1D_raw_data/T1D_patients.RDS")
    patient_df = T1D_patients_r[None]
    
    target_ids = set(patient_df.iloc[:, 0].astype(str).unique())
    
except Exception as e:
    print(f"Error loading RDS file: {e}")
    exit()

def process_and_save(filename, num, output_name, id_col_index=1):
    if (num == 1):
        input_path = f"{GET_PATH_1}/T1D_raw_data/{filename}"
    else:
        input_path = f"{GET_PATH_2}/{filename}"
    output_path = f"{WRITE_PATH}{output_name}"
    
    print(f"Processing {filename}...")
    
    if os.path.exists(output_path):
        os.remove(output_path)
    
    file_sep = '\t' if filename.endswith('.txt') else ','

    # dtype=str' forces pandas to read all columns as text
    # chunking to mitigate memory errors
    chunk_reader = pd.read_csv(
        input_path, 
        header=0, 
        sep=file_sep, 
        chunksize=100000, 
        # older EHR, uses older encoding system
        encoding='cp1252',
        dtype=str,        
    )

    total_rows = 0
    
    for i, chunk in enumerate(chunk_reader):
        actual_id_col_name = chunk.columns[id_col_index]

        # Rename the target column index to "ID" to standardize
        chunk.rename(columns={id_col_index: "ID"}, inplace=True)
        
        # Filter: strictly string vs string
        filtered_chunk = chunk[chunk["ID"].astype(str).isin(target_ids)]
        
        if not filtered_chunk.empty:
            write_header = (total_rows == 0)
            filtered_chunk.to_csv(output_path, mode='a', index=False, header=write_header)
            total_rows += len(filtered_chunk)
            
        if i % 10 == 0:
            print(f"  Processed chunk {i}...")

    print(f"Finished {filename}. Saved {total_rows} rows.")

if len(target_ids) > 1:
    #
    # Encounters documents hospital visits, check-ups, etc
    # Prescribing documents presciptions ordered (but not necessarily realized), while med.txt documents in-house meds administered
    # procedures contains important medical device information (need to know if on pump/cgm)
    #
    process_and_save("T1D_lab.txt", 1, "T1D_labs_raw.csv", id_col_index=1)
    process_and_save("T1D_demographic.txt", 1, "T1D_demographics_raw.csv", id_col_index=0)
    process_and_save("T1D_encounter.txt", 1, "T1D_encounters_raw.csv", id_col_index=1)
    process_and_save("T1D_med.txt", 1, "T1D_meds_raw.csv", id_col_index=1)
    process_and_save("VITAL_YANG1_V1.csv", 2, "T1D_vitals_1.csv", id_col_index=1)
    process_and_save("VITAL_YANG2_V1.csv", 2, "T1D_vitals_2.csv", id_col_index=1)
    process_and_save("PROCEDURES_YANG1_V1.csv", 2, "T1D_procedures_1.csv", id_col_index=1)
    process_and_save("PROCEDURES_YANG2_V1.csv", 2, "T1D_procedures_2.csv", id_col_index=1)
    process_and_save("PROCEDURES_YANG3_V1.csv", 2, "T1D_procedures_3.csv", id_col_index=1)
    process_and_save("PROCEDURES_YANG4_V1.csv", 2, "T1D_procedures_4.csv", id_col_index=1)
    process_and_save("PRESCRIBING_YANG1_V1.csv", 2, "T1D_prescriptions_1.csv", id_col_index=1)
    process_and_save("PRESCRIBING_YANG2_V1.csv", 2, "T1D_prescriptions_2.csv", id_col_index=1)
    process_and_save("PRESCRIBING_YANG3_V1.csv", 2, "T1D_prescriptions_3.csv", id_col_index=1)
    process_and_save("PRESCRIBING_YANG4_V1.csv", 2, "T1D_prescriptions_4.csv", id_col_index=1)
else:
    print("Skipping processing because patient list seems empty.")
