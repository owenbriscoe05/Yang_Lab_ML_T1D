import pandas as pd

data_path = "./data/raw/"

def main():
    ...


def filter_data():
    """Could return a list of tuples: {name, df} to help keep track of all the metrics"""

    labs = pd.read_csv("./data/processed/resilience_metrics.csv", header=0)
    labs["RESULT_NUM_CLEAN"] = labs["RESULT_NUM"].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)
    na_mask = labs["RESULT_NUM_CLEAN"].isna()
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

    return [("hba1c", hba1c), ("glucose", glucose), ("agap", agap), ("creatinine", creat), ("potassium", pot), ("uacr", uacr)]

def create_features():
    """What the ML model actually sees needs to be tailored very carefully. We want it to discover resiliency features, not predict resilience USING resilience"""

def combine():
    ...

def pivot():
    ...


if __name__ == "__main__":
    main()