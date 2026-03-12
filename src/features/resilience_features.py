import os
import pandas as pd

data_path = "./data/raw/"
export_path = "./data/processed/grouped_checkpoints/"

def main():
    os.makedirs(export_path, exist_ok=True)

    filtered_labs = filter_lab_data()
    filtered_meds = filter_druglist()
    filtered_vitals = filter_vitals()
    filtered_procedures = filter_procedures()
    filtered_encounters = filter_encounters()

    anchors = create_patient_anchors()
    # using specimen date of collection for labs, could change
    windowed_labs = {}
    for name, df in filtered_labs:
        print(f"Defining universal window for {name}")
        windowed_labs[name] = create_universal_window(df, anchors, "SPECIMEN_DATE_OFFSET")
    windowed_meds = {}
    for name, df in filtered_meds:
        print(f"Defining universal window for {name}")
        windowed_meds[name] = create_universal_window(df, anchors, "RX_START_DATE_OFFSET")
    windowed_vitals = {}
    for name, df in filtered_vitals:
        print(f"Defining universal window for {name}")
        windowed_vitals[name] = create_universal_window(df, anchors, "MEASURE_DATE_OFFSET")
    windowed_procedures = {}
    for name, df in filtered_procedures:
        print(f"Defining universal window for {name}")
        windowed_procedures[name] = create_universal_window(df, anchors, "PX_DATE_OFFSET")
    windowed_encounters = {}
    for name, df in filtered_encounters:
        print(f"Defining universal window for {name}")
        windowed_encounters[name] = create_universal_window(df, anchors, "ADMIT_DATE_OFFSET")

    final_labs = group_labs(list(windowed_labs.items()))
    final_meds = group_drugs(list(windowed_meds.items()))
    final_vitals = group_vitals(list(windowed_vitals.items()))
    final_procedures = group_procedures(list(windowed_procedures.items()))
    final_encounters = group_encounters(list(windowed_encounters.items()))

    print("\nInitiating Parquet Export...")
    all_final_data = final_labs + final_meds + final_vitals + final_procedures + final_encounters
    
    for name, df in all_final_data:
        # Save as Parquet to preserve the ["ID", "TIME_WINDOW"] index types perfectly
        file_name = f"{export_path}{name}_grouped.parquet"
        df.to_parquet(file_name, index=False)
        print(f"Successfully exported: {file_name}")
    
    print("Done")

def filter_lab_data():
    """Could return a list of tuples: {name, df} to help keep track of all the metrics"""

    labs = pd.read_csv("./data/processed/resilience_metrics.csv", header=0)
    labs["RESULT_NUM_CLEAN"] = labs["RESULT_NUM"].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)
    na_mask = (labs["RESULT_NUM_CLEAN"].isna() | labs["RESULT_NUM_CLEAN"] == "NI")
    labs = labs[~na_mask]
    # labs taken at POC or "bedside" indicate some type of hospitalization event occurring
    labs["POC_LAB"] = (labs["RAW_LAB_NAME"].str.contains("POC|bedside|point of care", case=False, na=False)).astype(int)

    hba1c = labs[labs["RAW_LAB_NAME"].str.contains("hba1c|a1c", case=False, na=False)].copy()
    glucose = labs[labs["RAW_LAB_NAME"].str.contains("glucose", case=False, na=False)].copy()
    agap = labs[labs["RAW_LAB_NAME"].str.contains("AGAP", case=False, na=False)].copy()
    creat = labs[labs["RAW_LAB_NAME"].str.contains("creatinine", case=False, na=False)].copy()
    pot = labs[labs["RAW_LAB_NAME"].str.contains("potassium", case=False, na=False)].copy()

    # bedside_glucose = blood_glucose[blood_glucose["raw_lab_name"].str.contains("bedside|POC", case=False, na=False)]
    
    print("Applying lab filters \n")
    #HBA1C filters
    hba1c = hba1c[(hba1c["RESULT_NUM_CLEAN"] >= 2) & (hba1c["RESULT_NUM_CLEAN"] <= 20)]

    #Glucose filters
    glucose = glucose[glucose["RAW_UNIT"].astype(str).str.contains("mg/dl", case=False, na=False)]
    # very loose glucose filtering, mainly to ensure no extraneous accidental values
    glucose = glucose[(glucose["RESULT_NUM_CLEAN"] >= 30) & (glucose["RESULT_NUM_CLEAN"] <= 600)]

    #AGAP filters
    agap = agap[(agap["RESULT_NUM_CLEAN"] >= 2) & (agap["RESULT_NUM_CLEAN"] <= 25)]

    #creatinine filters
    # useful metric for early-stage kidney failure
    uacr = creat[creat["RAW_LAB_NAME"].str.contains("microalbumin", case=False, na=False)].copy()
    creat_ratio_mask = creat["RAW_LAB_NAME"].str.contains("ratio", case=False, na=False)
    creat = creat[~creat_ratio_mask]

    #potassium filters
    pot = pot[pot["RAW_UNIT"].astype(str).str.contains("mmol/L", case=False, na=False)]

    return [("labs", labs), ("hba1c", hba1c), ("glucose", glucose), 
            ("agap", agap), ("creatinine", creat), ("potassium", pot), ("uacr", uacr)]

def group_labs(windowed_labs_list):
    """group labs by ID and date to find long-term averages/st. deviations/sums/etc"""
    grouped_results = []
    group_cols = ["ID", "TIME_WINDOW"]

    for name, df in windowed_labs_list:
        print(f"Aggregating {name}\n")
        routine = df[df["POC_LAB"] == 0].groupby(group_cols).agg(
            **{
                f"{name}_routine_mean": ("RESULT_NUM_CLEAN", "mean"),
                f"{name}_routine_median": ("RESULT_NUM_CLEAN", "median"),
                f"{name}_routine_std": ("RESULT_NUM_CLEAN", "std"),
                f"{name}_routine_count": ("RESULT_NUM_CLEAN", "count")
            }
        ).reset_index()
        poc = df[df["POC_LAB"] == 1].groupby(group_cols).agg(
            **{
                f"{name}_poc_max": ("RESULT_NUM_CLEAN", "max"),
                f"{name}_poc_min": ("RESULT_NUM_CLEAN", "min"),
                f"{name}_poc_mean": ("RESULT_NUM_CLEAN", "mean"),
                f"{name}_poc_count": ("RESULT_NUM_CLEAN", "count")
            }
        ).reset_index()

        agg_df = pd.merge(routine, poc, on=group_cols, how="outer")
        grouped_results.append((name, agg_df))
    
    return grouped_results



def filter_druglist():
    """same type of tuple returned (name, df)"""

    drugs = pd.read_csv("./data/processed/prescribed_drugs.csv", header=0)
    na_mask = (drugs["RX_START_DATE_OFFSET"].isna()) | (drugs["RX_ORDER_DATE_OFFSET"].isna()) | (drugs["RX_START_DATE_OFFSET"] == "NI") | (drugs["RX_ORDER_DATE_OFFSET"] == "NI")
    drugs = drugs[~na_mask]
    # if end date offset is NA, use date of EHR download in place
    drug_starts = pd.to_datetime(drugs["RX_START_DATE_OFFSET"], errors="coerce")
    drug_ends = pd.to_datetime(drugs["RX_END_DATE_OFFSET"], errors="coerce")

    study_cutoff_date = pd.to_datetime("2023-09-28")
    drug_ends = drug_ends.fillna(study_cutoff_date)

    drugs["DRUG_DURATION"] = (drug_ends - drug_starts).dt.days

    lisinopril = drugs[drugs["RAW_RX_MED_NAME"].str.contains("lisinopril", case=False, na=False)].copy()
    losartan = drugs[drugs['RAW_RX_MED_NAME'].str.contains("losartan", case=False, na=False)].copy()
    metformin = drugs[drugs["RAW_RX_MED_NAME"].str.contains("metformin", case=False, na=False)].copy()
    ozempic = drugs[drugs["RAW_RX_MED_NAME"].str.contains("ozempic", case=False, na=False)].copy()
    cgm = drugs[drugs["RAW_RX_MED_NAME"].str.contains("CGM|DEXCOM|GLUCOSE MONITOR", case=False, na=False)].copy()
    insulin_pump = drugs[drugs["RAW_RX_MED_NAME"].str.contains("PUMP|OMNIPOD|TANDEM", case=False, na=False)].copy()
    steroids = drugs[drugs["RAW_RX_MED_NAME"].str.contains("prednisone|dexamethasone|hydrocortisone", case=False, na=False)].copy()
    immunosuppressants = drugs[drugs["RAW_RX_MED_NAME"].str.contains("tacrolimus|cyclosporine|methotrexate", case=False, na=False)].copy()

    print("Applying med/drug filters \n")
    lisinopril = lisinopril[lisinopril["DRUG_DURATION"] > 7]
    losartan = losartan[losartan["DRUG_DURATION"] > 7]
    metformin = metformin[metformin["DRUG_DURATION"] > 30]
    ozempic = ozempic[ozempic["DRUG_DURATION"] > 30]
    # to really gain an advantage from cgm or pump use, needs to be a longer period
    cgm = cgm[cgm["DRUG_DURATION"] > 90]
    insulin_pump = insulin_pump[insulin_pump["DRUG_DURATION"] > 90]
    immunosuppressants = immunosuppressants[immunosuppressants["DRUG_DURATION"] > 7]

    return [("drugs", drugs), ("lisinopril", lisinopril), ("losartan", losartan), ("metformin", metformin), 
            ("ozempic", ozempic), ("cgm", cgm), ("pump", insulin_pump), ("steroids", steroids), ("immuno", immunosuppressants)]

def group_drugs(windowed_meds_list):
    grouped_results = []
    group_cols=["ID", "TIME_WINDOW"]
    # change if different threshold desired
    prevalence_threshold = 0.05

    for name, df in windowed_meds_list:
        if (name == "drugs"):
            # using prevalence threshold 5%
            print("Identifying and grouping top drugs among prescribed meds")
            total_patients = df["ID"].nunique()
            patients_per_drug = df.groupby("RAW_RXNORM_CUI")["ID"].nunique()
            min_required_patients = total_patients*prevalence_threshold
            prevalent_drugs = patients_per_drug[patients_per_drug >= min_required_patients].index.tolist()
            if not prevalent_drugs:
                print("Warning: No drugs met the threshold. Returning empty dataframe.")
                grouped_results.append((name, pd.DataFrame(columns=["ID", "TIME_WINDOW"])))
                continue
            filtered_df = df[df["RAW_RXNORM_CUI"].isin(prevalent_drugs)].copy()

            filtered_df["RAW_RXNORM_CUI"] = filtered_df["RAW_RXNORM_CUI"].astype(str)

            pivot_df = pd.pivot_table(
                filtered_df,
                index=["ID", "TIME_WINDOW"],
                columns="RAW_RXNORM_CUI",
                values="DRUG_DURATION",
                aggfunc=["sum", "count"],
                fill_value=0
            )

            pivot_df.columns = [f"{col[1].replace(' ', '_')}_duration_{col[0]}" for col in pivot_df.columns]

            grouped_results.append((name, pivot_df.reset_index()))
        
        else:
            print(f"Aggregating {name}\n")

            agg_df = df.groupby(group_cols).agg(
                **{
                    f"{name}_total_days": ("DRUG_DURATION", "sum"),
                    f"{name}_total_prescribed": ("DRUG_DURATION", "count")
                }
            ).reset_index()

            grouped_results.append((name, agg_df))
    
    return grouped_results

def filter_vitals():
    reader = pd.read_csv("./data/processed/vitals.csv", header=0, chunksize=500000)
    chunks = []
    for i, chunk in enumerate(reader):
        chunks.append(chunk)
        if (i%30 == 0):
            print(f"Filtered chunk {i} of vitals.csv \n")
    vitals = pd.concat(chunks, ignore_index=True)

    print("Applying vitals filters \n")

    vitals = vitals.dropna(subset=["MEASURE_DATE_OFFSET", "SYSTOLIC", "DIASTOLIC", "ORIGINAL_BMI"])
    vitals = vitals[(vitals["SYSTOLIC"] != "NI") & (vitals["DIASTOLIC"] != "NI") & (vitals["ORIGINAL_BMI"] != "NI")]

    vitals["SYSTOLIC"] = vitals["SYSTOLIC"].astype(float)
    vitals["DIASTOLIC"] = vitals["DIASTOLIC"].astype(float)
    vitals["ORIGINAL_BMI"] = vitals["ORIGINAL_BMI"].astype(float)

    hypertension = vitals[(vitals["SYSTOLIC"] > 140) | (vitals["DIASTOLIC"] > 90)].copy()
    emergency_vitals = vitals[(vitals["SYSTOLIC"] > 180) | (vitals["DIASTOLIC"] > 120)].copy()
    underweight = vitals[vitals["ORIGINAL_BMI"] < 18.5].copy()
    obesity = vitals[vitals["ORIGINAL_BMI"] > 30].copy()

    smoker = vitals[(vitals["SMOKING"].astype(str).str.startswith("1")) | 
                    (vitals["TOBACCO"].astype(str).str.startswith("1"))].copy()

    return [("vitals", vitals), ("hypertension", hypertension), ("er_vitals", emergency_vitals), ("obesity", obesity), ("underweight", underweight), ("smoker", smoker)]

def group_vitals(windowed_vitals_list):
    """group vitals by ID and date to find long-term averages/st. deviations/sums/etc"""
    grouped_results = []
    group_cols = ["ID", "TIME_WINDOW"]

    for name, df in windowed_vitals_list:
        if name == "er_vitals":
            agg_df = df.groupby(group_cols).agg(
                # count returns the number of extreme hypertensives this patient had in the time window,
                # while lambda just returns if they had any extreme hypertensives in this time window
                bp_crisis_count=("ID", "count"),
                bp_crisis_flag=("ID", lambda x: 1)
            ).reset_index()
        
        elif name == "smoker":
            agg_df = df.groupby(group_cols).agg(
                smoker_flag=("ID", lambda x: 1)
            ).reset_index()
        
        elif name == "obesity":
            agg_df = df.groupby(group_cols).agg(
                obesity_flag=("ID", lambda x: 1)
            )
        
        elif name == "underweight":
            agg_df = df.groupby(group_cols).agg(
                underweight_flag=("ID", lambda x: 1)
            )
        
        elif name == "else":
            df["SYSTOLIC"] = df["SYSTOLIC"].astype(float)
            df["DIASTOLIC"] = df["DIASTOLIC"].astype(float)
            df["BMI"] = df["BMI"].astype(float)

            agg_df = df.groupby(group_cols).agg(
                bmi_max=("BMI", "max"),
                bmi_min=("BMI", "min"),
                systolic_mean=("SYSTOLIC", "mean"),
                systolic_max=("SYSTOLIC", "max"),
                systolic_std=("SYSTOLIC", "std"),
                diastolic_mean=("DIASTOLIC", "mean"),
                diastolic_max=("DIASTOLIC", "max"),
                diastolic_std=("DIASTOLIC", "std")
            ).reset_index()
        grouped_results.append((name, agg_df))
    
    return grouped_results

def filter_procedures():
    reader = pd.read_csv("./data/processed/procedures.csv", header=0, chunksize=500000)
    chunks = []
    for i, chunk in enumerate(reader):
        procedures_mask = (chunk["PX"].isna()) | (chunk["PX"] == "NI") | (chunk["PX_DATE_OFFSET"].isna())
        chunk_clean = chunk[~procedures_mask]

        chunks.append(chunk_clean)
        if (i%30 == 0):
            print(f"Filtered chunk {i} of procedures.csv \n")
    procedures = pd.concat(chunks, ignore_index=True)

    print("Applying procedures filters \n")

    cgm_codes = ['A9276', 'A9277', 'A9278', 'K0553', 'K0554']
    pump_codes = ['E0784', 'A9274', 'S1034', 'A4224']

    on_cgm = procedures[procedures["PX"].isin(cgm_codes)].copy()
    on_pump = procedures[procedures["PX"].isin(pump_codes)].copy()  

    # procedures has some useful overlap with encounters that provides more context for emergency visits
    icu_codes = ['99291', '99292', '31500', '94002', '94003']
    dialysis_codes = ['90935', '90937', '90945', '90947'] + [str(x) for x in range(90951, 90971)]
    amputation_codes = ['28820', '28825', '28810', '28805', '28800', '27880', '27881', '27882', '27590', '27591', '27592']
    retinopathy_codes = ['67228', '67028']
    # 11042-44 represent surgical debridement, 97597-8 represent wound care for debridement
    neuropathy_codes = ['11042', '11043', '11044', '97597', '97598']

    icu_visits = procedures[procedures["PX"].isin(icu_codes)].copy()
    dialysis = procedures[procedures["PX"].isin(dialysis_codes)].copy()
    amputations = procedures[procedures["PX"].isin(amputation_codes)].copy()
    retinopathy_surgery = procedures[procedures["PX"].isin(retinopathy_codes)].copy()
    neuropathy_surgery = procedures[procedures["PX"].isin(neuropathy_codes)].copy()
    
    return [("procedures", procedures), ("cgm", on_cgm), ("pump", on_pump), ("icu", icu_visits),
            ("dialysis", dialysis), ("amputations", amputations), ("retinopathy", retinopathy_surgery),
            ("neuropathy", neuropathy_surgery)]

def group_procedures(windowed_procedures_list):
    """group vitals by ID and date to find long-term averages/st. deviations/sums/etc"""
    grouped_results = []
    group_cols = ["ID", "TIME_WINDOW"]

    for name, df in windowed_procedures_list:
        if name == "icu":
            agg_df = df.groupby(group_cols).agg(
                # count returns the number of extreme hypertensives this patient had in the time window,
                # while lambda just returns if they had any extreme hypertensives in this time window
                icu_count=("ID", "count"),
                icu_flag=("ID", lambda x: 1)
            ).reset_index()
        
        elif name == "amputations":
            agg_df = df.groupby(group_cols).agg(
                amputation_flag=("ID", lambda x: 1)
            ).reset_index()
        
        elif name == "neuropathy":
            agg_df = df.groupby(group_cols).agg(
                neuropathy_flag=("ID", lambda x: 1)
            ).reset_index()
        
        elif name == "dialysis":
            agg_df = df.groupby(group_cols).agg(
                dialysis_flag=("ID", lambda x: 1)
            ).reset_index()
        
        elif name == "retinopathy":
            agg_df = df.groupby(group_cols).agg(
                retinopathy_flag=("ID", lambda x: 1)
            ).reset_index()
        
        elif name == "cgm":
            agg_df = df.groupby(group_cols).agg(
                cgm_count=("ID", "count"),
                cgm_flag=("ID", lambda x: 1)
            ).reset_index()
        
        elif name == "pump":
            agg_df = df.groupby(group_cols).agg(
                pump_count=("ID", "count"),
                pump_flag=("ID", lambda x: 1)
            ).reset_index()
        
        else:
            agg_df = df.groupby(group_cols).agg(
                all_px_count=("PX", "count"),
                unique_px_days=("PX_DATE_OFFSET", "nunique"),
                unique_px_types=("PX", "nunique")
            ).reset_index()
        grouped_results.append((name, agg_df))
    
    return grouped_results

def filter_encounters():
    reader = pd.read_csv("./data/processed/T1D_encounters_clean.csv", header=0, chunksize=500000)
    chunks = []
    for i, chunk in enumerate(reader):
        # remove unusable data and "E" for expired (dead) patients
        enc_mask = ((chunk["ADMIT_DATE_OFFSET"].isna()) & (chunk["DISCHARGE_DATE_OFFSET"].isna())) | (chunk["DISCHARGE_DISPOSITION"] == "E")
        clean_chunk = chunk[~enc_mask]
        chunks.append(clean_chunk)
        if (i%30 == 0):
            print(f"Filtered chunk {i} of T1D_encounters_clean.csv \n")
    encounters = pd.concat(chunks, ignore_index=True)

    print("Applying encounters filters \n")

    er_visit = encounters[encounters["ENC_TYPE"].str.contains("ED", case=False, na=False)].copy()
    inpatient_visit = encounters[encounters["ENC_TYPE"].str.contains("EI|IP", case=False, na=False)].copy()
    ambulatory_visit = encounters[encounters["ENC_TYPE"].str.contains("AV|OA", case=False, na=False)].copy()
    hospice = encounters[encounters["DISCHARGE_STATUS"] == "HS"]

    return [("encounters", encounters), ("ER", er_visit), ("IP", inpatient_visit), ("AV", ambulatory_visit), ("hospice", hospice)]

def group_encounters(windowed_encounters_list):
    grouped_results = []
    group_cols = ["ID", "TIME_WINDOW"]

    for name, df in windowed_encounters_list:
        if (name == "encounters"):
            continue
        else:
            print(f"Aggregating {name} from encounters \n")
            agg_df = df.groupby(group_cols).agg(
                **{
                    f"{name}_total_count": ("ID", "count")
                }
            ).reset_index()
        grouped_results.append((name, agg_df))

    return grouped_results

def create_patient_anchors():
    reader = pd.read_csv("./data/processed/T1D_encounters_clean.csv", header=0, chunksize=500000)
    chunks = []
    for i, chunk in enumerate(reader):
        chunks.append(chunk)
        if (i%30 == 0):
            print(f"Filtered chunk {i} of T1D_encounters_clean.csv \n")
    encounters = pd.concat(chunks, ignore_index=True)

    enc_mask = ((encounters["ADMIT_DATE_OFFSET"].isna()) & (encounters["DISCHARGE_DATE_OFFSET"].isna())) | (encounters["DISCHARGE_DISPOSITION"] == "E")
    encounters = encounters[~enc_mask]

    encounters["ADMIT_DATE_OFFSET"] = pd.to_datetime(encounters["ADMIT_DATE_OFFSET"], errors="coerce")
    anchor_dates = encounters.groupby("ID")["ADMIT_DATE_OFFSET"].min().reset_index()
    anchor_dates.rename(columns={"ADMIT_DATE_OFFSET": "DAY_0"}, inplace=True)

    return anchor_dates

def create_universal_window(df, anchor_dates, date_column, window_size_days=365):
    """Assigns time windows to specified df"""

    df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
    df = df.dropna(subset=[date_column]) # might need to change
    df = df.merge(anchor_dates, on="ID", how="left")
    df["DAYS_SINCE_START"] = (df[date_column] - df["DAY_0"]).dt.days
    df = df[df["DAYS_SINCE_START"] >= 0]

    bins = range(0, 10000, window_size_days)
    labels = [f"Year_{i+1}" for i in range(len(bins) - 1)]

    df["TIME_WINDOW"] = pd.cut(df["DAYS_SINCE_START"], bins=bins, labels=labels, right=False)

    return df


def create_features():
    """What the ML model actually sees needs to be tailored very carefully. We want it to discover resiliency features, not predict resilience USING resilience"""


if __name__ == "__main__":
    main()