import pandas as pd
from src import config
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

## SETUP
def load_data():
    df = pd.read_csv(config.DATA_CLN)
    print("Loaded CSV")
    return df

def remove_exact_duplicates(df):
    duplicate_rows = df.duplicated(keep="first")
    df_deduped = df.loc[~duplicate_rows].copy()
    print("Removed exact duplicates")
    return df_deduped

## Feature engineering
def engineer_features(df):
    print("[STARTING] Features engineering...")
    df = process_availability(df)
    df = process_category(df)
    df = process_epc(df)
    df = process_flooding_area(df)
    df = process_binary_missing_values(df)
    print("[COMPLETED] Features engineering...")
    return df

# Process "availability" to a binary "available_immediately"
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

    # Drop original column
    df = df.drop(columns=["availability"])

    print("Processed feature 'availability' into 'available_immediately'")
    return df

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

def process_binary_missing_values(df):
    df = df.copy()

    for col in config.BINARY_MISSING_AS_ZERO:
        df[col] = df[col].fillna(0)
    print("Processed missing values for binary features")
    return df

## Select target and features
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

## Split train/test
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

def main():

    print(f"\n{'=' * 60}")
    print(" training_base.py ")
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