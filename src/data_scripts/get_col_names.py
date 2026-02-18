import pandas as pd
"""Run to get a printed output of most common labs run, meds taken, and vitals recorded among T1D patients"""

labs = "./data/raw/T1D_labs_clean.csv"
meds = "./data/raw/T1D_meds_clean.csv"
encs = "./data/raw/T1D_encounters_clean.csv"

def main():
    get_lab_names()
    get_imp_meds_names()
    get_encounter_types()

def get_lab_names():
    df = pd.read_csv(labs, usecols=["raw_lab_name", "lab_loinc"], chunksize=1_000_000)

    test_counts = pd.Series(dtype=int)
    loinc_test_counts = pd.Series(dtype=int)

    for i, chunk in enumerate(df):
        chunk_counts = chunk["raw_lab_name"].value_counts()
        loinc_counts = chunk["lab_loinc"].value_counts()
        test_counts = test_counts.add(chunk_counts, fill_value=0)
        loinc_test_counts = loinc_test_counts.add(loinc_counts, fill_value=0)
    
    print(test_counts.sort_values(ascending=False).head(50))
    test_counts.sort_values(ascending=False).to_csv("./data/processed/all_raw_lab_names.csv")
    loinc_test_counts.sort_values(ascending=False).to_csv("./data/processed/all_raw_loinc.csv")

def get_imp_meds_names():
    df = pd.read_csv(meds, usecols=["raw_medadmin_med_name"], chunksize=1_000_000)

    test_counts = pd.Series(dtype=int)

    for i, chunk in enumerate(df):
        chunk_counts = chunk["raw_medadmin_med_name"].value_counts()
        test_counts = test_counts.add(chunk_counts, fill_value=0)
    
    print(chunk_counts.sort_values(ascending=False).head(50))
    test_counts.sort_values(ascending=False).to_csv("./data/processed/all_raw_med_names.csv")

def get_encounter_types():
    df = pd.read_csv(encs, usecols=["enc_type"], chunksize=1_000_000)

    test_counts = pd.Series(dtype=int)

    for i, chunk in enumerate(df):
        chunk_counts = chunk["enc_type"].value_counts()
        test_counts = test_counts.add(chunk_counts, fill_value=0)
    
    print(chunk_counts.sort_values(ascending=False).head(50))
    test_counts.sort_values(ascending=False).to_csv("./data/processed/all_enc_types.csv")
def get_vital_names():
    ...


if __name__ == "__main__":
    main()