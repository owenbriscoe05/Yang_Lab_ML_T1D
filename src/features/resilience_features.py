import pandas as pd

data_path = "./data/raw/"

def main():
    ...


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
    
    #HBA1C filters
    # Apply Range Filter (2-20%)
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

def filter_druglist():
    """same type of tuple returned (name, df)"""

    drugs = pd.read_csv("./data/processed/confounding_drugs_external.csv", header=0)
    na_mask = (drugs["RX_START_DATE_OFFSET"].isna()) | (drugs["RX_ORDER_DATE_OFFSET"].isna()) | (drugs["RX_START_DATE_OFFSET"] == "NI") | (drugs["RX_ORDER_DATE_OFFSET"] == "NI")
    drugs = drugs[~na_mask]
    # if end date offset is NA, use date of EHR download in place
    drug_starts = pd.to_datetime(drugs["RX_START_DATE_OFFSET"], errors="coerce")
    drug_ends = pd.to_datetime(drugs["RX_END_DATE_OFFSET"], errors="coerce")
    drugs["DRUG_DURATION"] = (drug_ends - drug_starts).dt.days

    lisinopril = drugs[drugs["RAW_RX_MED_NAME"].str.contains("lisinopril", case=False, na=False)].copy()
    losartan = drugs[drugs['RAW_RX_MED_NAME'].str.contains("losartan", case=False, na=False)].copy()
    metformin = drugs[drugs["RAW_RX_MED_NAME"].str.contains("metformin", case=False, na=False)].copy()
    ozempic = drugs[drugs["RAW_RX_MED_NAME"].str.contains("ozempic", case=False, na=False)].copy()
    cgm = drugs[drugs["RAW_RX_MED_NAME"].str.contains("CGM|DEXCOM|GLUCOSE MONITOR", case=False, na=False)].copy()
    insulin_pump = drugs[drugs["RAW_RX_MED_NAME"].str.contains("PUMP|OMNIPOD|TANDEM", case=False, na=False)].copy()
    steroids = drugs[drugs["RAW_RX_MED_NAME"].str.contains("prednisone|dexamethasone|hydrocortisone", case=False, na=False)].copy()
    immunosuppressants = drugs[drugs["RAW_RX_MED_NAME"].str.contains("tacrolimus|cyclosporine|methotrexate", case=False, na=False)].copy()

    lisinopril = lisinopril[lisinopril["DRUG_DURATION"] > 7]
    losartan = losartan[losartan["DRUG_DURATION"] > 7]
    metformin = metformin[metformin["DRUG_DURATION"] > 30]
    ozempic = ozempic[ozempic["DRUG_DURATION"] > 30]
    # to really gain an advantage from cgm or pump use, needs to be a longer period
    cgm = cgm[cgm["DRUG_DURATION"] > 90]
    insulin_pump = insulin_pump[insulin_pump["DRUG_DURATION"] > 90]
    immunosuppressants = immunosuppressants[immunosuppressants["DRUG_DURATION"] > 7]

    return [("drugs", drugs), ("lisinopril", lisinopril), ("losartan", losartan), ("metformin", metformin), 
            ("ozempic", ozempic), ("cgm", cgm), ("pump", insulin_pump), ("immuno", immunosuppressants)]

def filter_vitals():
    reader = pd.read_csv("./data/processed/vitals.csv", header=0, chunksize=500000)
    chunks = []
    for i, chunk in enumerate(reader):
        chunks.append(chunk)
        if (i%30 == 0):
            print(f"Filtered chunk {i} of vitals.csv \n")
    vitals = pd.concat(chunks, ignore_index=True)

    vitals_mask = (((vitals["DIASTOLIC"].isna() | vitals["DIASTOLIC"] == "NI") |
                   (vitals["SYSTOLIC"].isna() | vitals["SYSTOLIC"] == "NI")) & 
                   (vitals["ORIGINAL_BMI"].isna() | vitals["ORIGINAL_BMI"] == "NI") &
                   (vitals["SMOKING"].isna() | vitals["SMOKING"] == "NI") &
                   (vitals["TOBACCO"].isna() | vitals["TOBACCO"] == "NI"))
    vitals = vitals[~vitals_mask]

    hypertension = vitals[(vitals["SYSTOLIC"].astype(float) > 140) | (vitals["DIASTOLIC"].astype(float) > 90)].copy()
    emergency_vitals = vitals[(vitals["SYSTOLIC"].astype(float) > 180) | (vitals["DIASTOLIC"].astype(float) > 120)].copy()
    obesity = vitals[vitals["ORIGINAL_BMI"].astype(float) > 30].copy()
    smoker = vitals[(vitals["SMOKING"].astype(int) == 1) | (vitals["TOBACCO"].astype(int) == 1)].copy()

    return [("hypertension", hypertension), ("er_vitals", emergency_vitals), ("obesity", obesity), ("smoker", smoker)]

def filter_procedures():
    reader = pd.read_csv("./data/processed/procedures.csv", header=0, chunksize=500000)
    chunks = []
    for i, chunk in enumerate(reader):
        chunks.append(chunk)
        if (i%30 == 0):
            print(f"Filtered chunk {i} of procedures.csv \n")
    procedures = pd.concat(chunks, ignore_index=True)

    procedures_mask = procedures[(procedures["PX"].isna()) | (procedures["PX"] == "NI") | (procedures["PX_DATE_OFFSET"].isna())]
    procedures = [~procedures_mask]

    cgm_codes = ['A9276', 'A9277', 'A9278', 'K0553', 'K0554']
    pump_codes = ['E0784', 'A9274', 'S1034']

    on_cgm = procedures[procedures["PX"].isin(cgm_codes)].copy()
    on_pump = procedures[procedures["PX"].isin(pump_codes)].copy()  

    # procedures has some useful overlap with encounters that provides more context for emergency visits
    icu_codes = ['99291', '99292', '31500', '94002', '94003']
    dialysis_codes = ['90935', '90937', '90945', '90947'] + [str(x) for x in range(90951, 90971)]
    amputation_codes = ['28820', '28825', '28810', '28805', '28800', '27880', '27881', '27882', '27590', '27591', '27592']
    retinopathy_codes = ['67228', '67028']

    icu_visits = procedures[procedures["PX"].isin(icu_codes)].copy()
    dialysis = procedures[procedures["PX"].isin(dialysis_codes)].copy()
    amputations = procedures[procedures["PX"].isin(amputation_codes)].copy()
    retinopathy_surgery = procedures[procedures["PX"].isin(retinopathy_codes)].copy()
    
    return [("procedures", procedures), ("cgm", on_cgm), ("pump", on_pump), ("icu", icu_visits),
            ("dialysis", dialysis), ("amputations", amputations), ("retinopathy", retinopathy_surgery)]

def filter_encounters():
    reader = pd.read_csv("./data/processed/T1D_encounters_clean.csv", header=0, chunksize=500000)
    chunks = []
    for i, chunk in enumerate(reader):
        chunks.append(chunk)
        if (i%30 == 0):
            print(f"Filtered chunk {i} of T1D_encounters_clean.csv \n")
    encounters = pd.concat(chunks, ignore_index=True)

    # remove unusable data and "E" for expired (dead) patients
    enc_mask = ((encounters["ADMIT_DATE_OFFSET"].isna()) & (encounters["DISCHARGE_DATE_OFFSET"].isna())) | encounters["DISCHARGE_DISPOSITION" == "E"]
    encounters = encounters[~enc_mask]

    er_visit = encounters[encounters["ENC_TYPE"].str.contains("ED", case=False, na=False)].copy()
    inpatient_visit = encounters[encounters["ENC_TYPE"].str.contains("EI|IP", case=False, na=False)].copy()
    ambulatory_visit = encounters[encounters["ENC_TYPE"].str.contains("AV|OA", case=False, na=False)].copy()
    hospice = encounters[encounters["DISCHARGE_STATUS"] == "HS"]

    return [("encounters", encounters), ("ER", er_visit), ("IP", inpatient_visit), ("AV", ambulatory_visit), ("hospice", hospice)]


def create_features():
    """What the ML model actually sees needs to be tailored very carefully. We want it to discover resiliency features, not predict resilience USING resilience"""

def combine():
    ...

def pivot():
    ...


if __name__ == "__main__":
    main()