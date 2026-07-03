"""
-----------------------------------------
Features Engineer
-----------------------------------------

This script transforms the cleaned dataset into model-ready features by:
    1. Loading the cleaned CSV and removing any remaining exact duplicates
    2. Enriching each listing with Statbel commune-level median prices
       via a postal_code to NIS code lookup
    3. Applying feature-specific transformations (availability, category,
       EPC, flooding area, parkings, building state)
    4. Engineering derived features (property_age)
    5. Separating the price target from input features and splitting
       into train/test sets

Functions:
    ### Orchestrators ###
    - prepare_training_data()        : full pipeline returning X_train, X_test,
                                       y_train, y_test for model training
    - prepare_cv_data()              : full pipeline returning X, y for cross validation (CV)
    - engineer_features()            : calls all feature transformation steps in order

    ### Data loading ###
    - load_data()                    : loads the cleaned CSV into a dataframe
    - load_statbel_commune()         : loads Statbel commune-level median prices
                                       from the xlsx file (2025 data only)
    - load_postal_nis_mapping()      : builds a postal_code to NIS code dictionary
                                       from the INS/NIS reference CSV

    ### Feature transformations ###
    - remove_exact_duplicates()      : removes remaining exact row duplicates
    - enrich_with_statbel()          : joins commune median price and transaction
                                       count per listing (house vs. apartment split)
    - process_availability()         : converts availability to binary available_immediately flag
    - process_category()             : removes out-of-scope categories and maps student_house to appartment
    - process_epc()                  : maps regional EPC labels (Flanders, Brussels, Wallonia) to a unified quality scale
    - process_flooding_area()        : normalises flooding_area_type and nulls unavailable values
    - process_parkings()             : converts indoor/outdoor parking counts to binary presence flags
    - process_building_state()       : merges sparse building_state labels
    - engineer_derived_features()    : computes property_age from post_year minus build_year (clipped to 0)

    ### Splitting ###
    - split_target_features()        : separates target (price) from feature columns
    - split_train_test()             : splits X and y into train/test sets

    ### Standalone entry point ###
    - main()                         : runs the full data preparation pipeline
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

# Orchestrator : Prepare data for final model training
def prepare_training_data():
    df = load_data()
    df = remove_exact_duplicates(df)
    df = engineer_features(df)
    X, y = split_target_features(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    return X_train, X_test, y_train, y_test

# Orchestrator : Prepare data for cross validation
def prepare_cv_data():
    df = load_data()
    df = remove_exact_duplicates(df)
    df = engineer_features(df)
    X, y = split_target_features(df)
    return X, y

# Load the cleaned CSV selected in config.DATA_CLN
def load_data():
    df = pd.read_csv(config.DATA_CLN)
    print("Loaded CSV")
    return df

# Load 2025 Statbel commune-level transaction counts and median prices
def load_statbel_commune():
    raw = pd.read_excel(config.DATA_STATBEL_COMMUNE, sheet_name="Par commune", header=None)
    data = raw.iloc[3:].copy()  # Rows 0-2 are headers; data starts at row 3
    data.columns = range(data.shape[1])
    df = pd.DataFrame({
        "nis_code": pd.to_numeric(data[0], errors="coerce"),
        "year": pd.to_numeric(data[2], errors="coerce"),
        "house_count": pd.to_numeric(data[5], errors="coerce"),
        "house_median": pd.to_numeric(data[6], errors="coerce"),
        "apt_count": pd.to_numeric(data[20], errors="coerce"),
        "apt_median": pd.to_numeric(data[21], errors="coerce"),
    })
    return df[df["year"] == 2025].dropna(subset=["nis_code"]).reset_index(drop=True)

# Load postal-code to NIS-code mapping used to join Statbel data
def load_postal_nis_mapping():
    df = pd.read_csv(config.DATA_POSTAL_NIS, sep=";", usecols=["Code INS Commune", "Code postal"])
    df.columns = ["nis_code", "postal_code"]
    df["postal_code"] = df["postal_code"].astype(str).str.strip()
    return df.drop_duplicates(subset="postal_code").set_index("postal_code")["nis_code"].to_dict()

# Add Statbel commune median price and transaction count to each listing
def enrich_with_statbel(df):
    df = df.copy()

    statbel = load_statbel_commune()
    postal_to_nis = load_postal_nis_mapping()

    statbel_lookup = statbel.set_index("nis_code")

    postal_code_str = df["postal_code"].astype(str).str.strip()
    nis_codes = postal_code_str.map(postal_to_nis)

    is_apt = df["category"] == "appartment"

    commune_median = pd.Series(index=df.index, dtype="float64")
    transaction_count = pd.Series(index=df.index, dtype="float64")

    for idx, nis in nis_codes.items():
        if pd.isna(nis) or nis not in statbel_lookup.index:
            continue
        row = statbel_lookup.loc[nis]
        if is_apt[idx]:
            commune_median[idx] = row["apt_median"]
            transaction_count[idx] = row["apt_count"]
        else:
            commune_median[idx] = row["house_median"]
            transaction_count[idx] = row["house_count"]

    df["statbel_commune_median"] = commune_median
    df["statbel_transaction_count"] = transaction_count

    matched = commune_median.notna().sum()
    print(f"Enriched with Statbel commune data: {matched}/{len(df)} rows matched")
    return df

# Remove exact duplicate rows that may remain after cleaning
def remove_exact_duplicates(df):
    duplicate_rows = df.duplicated(keep="first")
    df_deduped = df.loc[~duplicate_rows].copy()
    print("Removed exact duplicates")
    return df_deduped

# Orchestrator : Run all feature engineering steps
def engineer_features(df):
    print("[STARTING] Features engineering...")
    df = process_availability(df)
    df = process_category(df)
    df = process_epc(df)
    df = process_flooding_area(df)
    df = process_parkings(df)
    df = process_building_state(df)
    df = enrich_with_statbel(df)
    df = engineer_derived_features(df)
    print("[COMPLETED] Features engineering...")
    return df

# Convert availability into nullable binary available_immediately
def process_availability(df):
    df = df.copy()

    # Converts all value for smooth comparison
    availability_clean = (
        df["availability"]
        .astype("string")
        .str.lower()
        .str.strip()
    )

    # New column created with missing as base value
    df["available_immediately"] = pd.NA

    # If availability == immediately, set to 1
    df.loc[availability_clean.eq("immediately"), "available_immediately"] = 1

    # If availability is != immediately, set to 0
    df.loc[availability_clean.notna() & ~availability_clean.eq("immediately"), "available_immediately"] = 0
    
    # Converts to Int64 to preserve missing data
    df["available_immediately"] = df["available_immediately"].astype("Int64")
    df = df.drop(columns=["availability"])

    print("Processed feature 'availability' into 'available_immediately'")
    return df

# Remove out-of-scope property categories and normalize student housing
def process_category(df):
    df = df.copy()

    categories_to_remove = [
        "garage",
        "investment_property",
        "land",
        "entreprise",
    ]

    # Remove unwanted categories
    df = df[~df["category"].isin(categories_to_remove)].copy()

    # Transform student_house to appartment
    df["category"] = df["category"].replace({
        "student_house": "appartment"
    })

    print("Processed features 'category' & 'property_type'")
    return df

# Map regional EPC labels to a shared quality scale
def process_epc(df):
    df = df.copy()

    epc_map = {
        # Flanders
        "FlandersDoubleA": "excellent",
        "FlandersSingleA": "excellent",
        "FlandersB": "good",
        "FlandersC": "poor",
        "FlandersD": "poor",
        "FlandersE": "bad",
        "FlandersF": "bad",

        # Brussels
        "BrusselsA": "excellent",
        "BrusselsB": "good",
        "BrusselsC": "good",
        "BrusselsD": "poor",
        "BrusselsE": "poor",
        "BrusselsF": "bad",
        "BrusselsG": "bad",

        # Wallonia
        "WalloniaTripleA": "excellent",
        "WalloniaDoubleA": "excellent",
        "WalloniaSingleA": "good",
        "WalloniaB": "good",
        "WalloniaC": "poor",
        "WalloniaD": "poor",
        "WalloniaE": "poor",
        "WalloniaF": "bad",
        "WalloniaG": "bad",
    }

    # Cleaning for safety
    epc = df["epc"].astype("string").str.strip()

    # Checks matching value in the epc_map - missing is left as missing
    df["epc_quality"] = epc.map(epc_map).astype("string")

    # Drop original column
    df = df.drop(columns=["epc"])

    print("Processed feature 'epc' into 'epc_quality'")
    return df

# Normalize flooding area labels and convert unavailable values to missing
def process_flooding_area(df):
    df = df.copy()

    flooding_area_clean = (
        df["flooding_area_type"]
        .astype("string")
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .replace("(information not available)", pd.NA)
    )

    df["flooding_area_clean"] = flooding_area_clean
    df = df.drop(columns=["flooding_area_type"])

    print("Processed feature 'flooding_area_type' to 'flooding_area_clean'")
    return df

# Merge building_state labels into broader categories
def process_building_state(df):
    df = df.copy()

    df["building_state"] = df["building_state"].replace({
        "To be renovated": "To renovate",
        "To demolish":     "To restore",
    })

    print("Processed feature 'building_state': merged sparse labels")
    return df

# Convert indoor/outdoor parking counts into nullable presence flags
def process_parkings(df):
    df = df.copy()

    parking_cols = [
        "indoor_parking",
        "outdoor_parking",
    ]

    for col in parking_cols:
        parking = pd.to_numeric(df[col], errors="coerce")

        df[col] = pd.NA
        df.loc[parking.eq(0), col] = 0
        df.loc[parking.notna() & parking.ne(0), col] = 1

        df[col] = df[col].astype("Int64")

    print("Processed parking features into binary presence flags")
    return df

# Create derived numeric features from existing columns
def engineer_derived_features(df):
    df = df.copy()

    build_year = pd.to_numeric(df["build_year"], errors="coerce")
    post_year = pd.to_numeric(df["post_year"], errors="coerce")

    # Clip to 0 - under construction properties have future build_year
    df["property_age"] = (post_year - build_year).clip(lower=0)

    print("Engineered derived feature 'property_age'")
    return df

# Separate target (price) from model input features
def split_target_features(df):
    print("[STARTING] Target/features split")
    target = "price"

    y = df[target]
    X = df.drop(columns=[target])

    print("Target:", target)
    print("Feature rows:", len(X))
    print("Feature columns:", len(X.columns))
    print("[COMPLETED] Target/features split")

    return X, y

# Split features and target into reproducible train/test sets
def split_train_test(X, y):
    print("[STARTING] Train/test split")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=config.RANDOM_SEED,
    )

    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)
    print("[COMPLETED] Train/test split")

    return X_train, X_test, y_train, y_test

# Run the feature engineering workflow as a standalone script
def main():

    print(f"\n{'=' * 60}")
    print(" features_engineer.py ")
    print(f"\n{'=' * 60}")

    print("[STARTING] Loading and preparing data...")
    df = load_data()
    df = remove_exact_duplicates(df)
    print("[COMPLETED] Initial setup")
    df = engineer_features(df)
    X, y = split_target_features(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

if __name__ == "__main__":
    main()