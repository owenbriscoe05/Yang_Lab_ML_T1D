import pandas as pd

data_path = "./data/raw/"

def main():
    ...


def filter_data():

    labs = pd.read_csv("./data/processed/resilience_metrics.csv", header=0)
    labs["RESULT_NUM"] = pd.to_numeric(labs["RESULT_NUM"], errors='coerce')
    

    # bedside_glucose = blood_glucose[blood_glucose["raw_lab_name"].str.contains("bedside|POC", case=False, na=False)]
    
    # Convert to numeric
    hba1c["result_num"] = pd.to_numeric(hba1c["result_num"], errors='coerce')
    
    # Apply Range Filter (2-20%)
    hba1c = hba1c[(hba1c["result_num"] >= 2) & (hba1c["result_num"] <= 20)]

def create_features():
    """What the ML model actually sees needs to be tailored very carefully. We want it to discover resiliency features, not predict resilience USING resilience"""

def combine():
    ...

def pivot():
    ...


if __name__ == "__main__":
    main()