"""
-----------------------------------------
Raw Data Auditor
-----------------------------------------

This script audits the raw listings CSV before model training by:
    1. Loading the raw dataset
    2. Detecting and correcting swapped latitude/longitude values
    3. Nulling coordinates that fall outside Belgian geographic bounds
    4. Removing exact content duplicates, keeping the most complete row
    5. Removing suspicious near-duplicates matched on key identifying columns
    6. Dropping rows with invalid data
    7. Nulling implausible values 
    8. Saving the audited dataset to DATA_AUDITED

Functions:
    - load_data()                    : loads the raw CSV into a dataframe
    - mini_cleaning()                : normalises key columns for duplicate comparison
    - remove_safe_duplicates()       : removes exact content duplicates, 
                                       keeps the row with the most non-missing values
    - remove_suspicious_duplicates() : removes near-duplicate listings matched
                                       on key identifying columns (location, price,
                                       surface, bedrooms, floor)
    - remove_invalid_price()         : drops rows where price is missing, non-positive,
                                       below MIN_VALID_PRICE or above the p99 cap
    - swap_coordinates()             : detects and corrects swapped lat/lon values
    - null_invalid_coordinates()     : nulls coordinates outside Belgian bounds
    - null_invalid_livable_surface() : nulls surface values below MIN_SURFACE
    - null_invalid_garages()         : nulls garage counts above MAX_GARAGES
    - null_invalid_facades()         : nulls facade counts above 4
    - null_invalid_total_floor()     : nulls total floor counts above MAX_FLOOR
    - null_invalid_floor()           : nulls floor values above MAX_FLOOR or
                                       above the building's total floor count
    - main()                         : orchestrates all auditing steps and saves
                                       the result to DATA_AUDITED
"""

# SETUP
import pandas as pd

from src import config

LAT_MIN = 49.5
LAT_MAX = 51.6
LON_MIN = 2.5
LON_MAX = 6.5

SUSPICIOUS_COLUMNS = [
    "longitude",
    "latitude",
    "price",
    "property_type",
    "category",
    "postal_code",
    "street",
    "house_number",
    "livable_surface",
    "bedrooms",
    "floor",
]

# Load CSV file
def load_data():
    df = pd.read_csv(config.DATA_RAW)
    return df

# Simple data type cleaning for duplicates check
def mini_cleaning(df):
    df_clean = df.copy()

    # Converts text columns
    str_cols  = [
        "property_type",
        "category",
        "street",
        "house_number",
        "postal_code",
    ]
    for col in str_cols:
        df_clean[col] = (
            df_clean[col]
            .astype("string")
            .str.strip()
            .str.lower()
            .str.replace(r"\s+", " ", regex=True)
        )
    
    # Converts float columns
    float_cols = [
        "longitude",
        "latitude",
    ]
    for col in float_cols:
        df_clean[col] = df_clean[col].astype(float)
    
    # Converts integer columns
    integer_columns = [
        "price",
        "livable_surface",
        "floor",
    ]
    for col in integer_columns:
        df_clean[col] = pd.to_numeric(df_clean[col]).astype("Int64")

    return df_clean

# Remove duplicates if all the below features are identical
def remove_safe_duplicates(df):
    metadata_cols = [
        "property_id",
        "immovlan_id",
        "url",
        "html_path",
        "scrape_date",
        "date_posted",
    ]

    content_cols = [col for col in df.columns if col not in metadata_cols]

    df_work = df.copy()
    df_work["_date_posted_dt"] = pd.to_datetime(df_work["date_posted"], errors="coerce")
    df_work["_non_missing_count"] = df_work[content_cols].notna().sum(axis=1)
    df_work["_original_order"] = range(len(df_work))

    df_sorted = df_work.sort_values(
        by=["_non_missing_count", "_date_posted_dt", "_original_order"],
        ascending=[False, False, True],
    )

    safe_duplicate_rows = df_sorted.duplicated(
        subset=content_cols,
        keep="first",
    )

    df_deduped = df_sorted.loc[~safe_duplicate_rows].copy()

    df_deduped = df_deduped.sort_values("_original_order")
    df_deduped = df_deduped.drop(
        columns=["_date_posted_dt", "_non_missing_count", "_original_order"]
    )

    print(f"\n{'=' * 60}")
    print("\nInitializing safe duplicates removal")
    print(f"{'=' * 60}")
    print("Rows before:", len(df))
    print("Rows after:", len(df_deduped))
    print("Rows removed:", len(df) - len(df_deduped))

    return df_deduped

# Removes suspicious duplicate rows by 
# grouping on key columns and keeping the most complete, 
# most recent entry per group while dropping the rest.
def remove_suspicious_duplicates(df):
    df_clean = mini_cleaning(df)

    df_work = df.copy()
    df_work["_date_posted_dt"] = pd.to_datetime(df_work["date_posted"], errors="coerce")
    df_work["_non_missing_count"] = df_work.notna().sum(axis=1)
    df_work["_original_order"] = range(len(df_work))

    df_work["_duplicate_group_id"] = (
        df_clean
        .groupby(SUSPICIOUS_COLUMNS, dropna=False)
        .ngroup()
    )

    df_sorted = df_work.sort_values(
        by=[
            "_duplicate_group_id",
            "_non_missing_count",
            "_date_posted_dt",
            "_original_order",
        ],
        ascending=[True, False, False, True],
    )

    suspicious_duplicate_rows = df_sorted.duplicated(
        subset="_duplicate_group_id",
        keep="first",
    )

    df_deduped = df_sorted.loc[~suspicious_duplicate_rows].copy()

    df_deduped = df_deduped.sort_values("_original_order")
    df_deduped = df_deduped.drop(
        columns=[
            "_date_posted_dt",
            "_non_missing_count",
            "_original_order",
            "_duplicate_group_id",
        ]
    )

    print(f"\n{'=' * 60}")
    print("Initializing suspicious duplicates removal")
    print(f"{'=' * 60}")
    print("Rows before:", len(df))
    print("Rows after:", len(df_deduped))
    print("Rows removed:", len(df) - len(df_deduped))

    return df_deduped

# Remove rows where price is missing, non-positive, 
# below MIN_VALID_PRICE (30,000) or above the p99 cap
def remove_invalid_price(df):
    df_work = df.copy()
    df_work["price"] = pd.to_numeric(df_work["price"], errors="coerce")

    max_price = df_work["price"].quantile(0.99)

    missing_price_rows = df_work["price"].isna()
    non_positive_price_rows = df_work["price"] <= 0
    below_min_price_rows = (df_work["price"] > 0) & (df_work["price"] < config.MIN_VALID_PRICE)
    above_max_price_rows = df_work["price"] > max_price

    invalid_price_rows = (
        missing_price_rows
        | non_positive_price_rows
        | below_min_price_rows
        | above_max_price_rows
    )

    df_cleaned = df_work.loc[~invalid_price_rows].copy()

    print(f"\n{'=' * 60}")
    print("Initializing invalid price removal")
    print(f"{'=' * 60}")
    print("Rows before:", len(df))
    print("Rows after:", len(df_cleaned))
    print("Rows removed:", len(df) - len(df_cleaned))
    print("Missing price rows:", int(missing_price_rows.sum()))
    print("Non-positive price rows:", int(non_positive_price_rows.sum()))
    print(f"Rows with price below {config.MIN_VALID_PRICE}:", int(below_min_price_rows.sum()))
    print(f"Rows with price above p99 ({max_price:,.0f}):", int(above_max_price_rows.sum()))

    return df_cleaned

# Swap lon/lat when necessary
def swap_coordinates(df):
    df_work = df.copy()

    latitude = pd.to_numeric(df_work["latitude"], errors="coerce")
    longitude = pd.to_numeric(df_work["longitude"], errors="coerce")

    swapped_coordinates = (
        latitude.between(LON_MIN, LON_MAX)
        & longitude.between(LAT_MIN, LAT_MAX)
    )

    df_work.loc[swapped_coordinates, ["latitude", "longitude"]] = (
        df_work.loc[swapped_coordinates, ["longitude", "latitude"]].values
    )

    print(f"\n{'=' * 60}")
    print("Initializing coordinates swap")
    print(f"{'=' * 60}")
    print("Swapped coordinates found:", int(swapped_coordinates.sum()))

    return df_work 

# Nulls lon/lat fields if they are outside of Belgium's boundaries
def null_invalid_coordinates(df):
    df_work = df.copy()

    df_work["latitude"] = pd.to_numeric(
        df_work["latitude"],
        errors="coerce",
    )
    df_work["longitude"] = pd.to_numeric(
        df_work["longitude"],
        errors="coerce",
    )

    invalid_coordinate_rows = (
        df_work["latitude"].notna()
        & df_work["longitude"].notna()
        & (
            ~df_work["latitude"].between(LAT_MIN, LAT_MAX)
            | ~df_work["longitude"].between(LON_MIN, LON_MAX)
        )
    )

    df_work.loc[invalid_coordinate_rows, ["latitude", "longitude"]] = pd.NA

    print(f"\n{'=' * 60}")
    print("Initializing invalid coordinates nulling")
    print(f"{'=' * 60}")
    print("Rows checked:", len(df))
    print("Rows with coordinates outside Belgium bounds:", int(invalid_coordinate_rows.sum()))

    return df_work

# Nulls livable surface field if < MIN_SURFACE (12m2)
def null_invalid_livable_surface(df):
    df_work = df.copy()
    df_work["livable_surface"] = pd.to_numeric(
        df_work["livable_surface"],
        errors="coerce",
    )

    invalid_livable_surface_rows = (
        df_work["livable_surface"].notna()
        & (df_work["livable_surface"] < config.MIN_SURFACE)
    )

    df_work.loc[invalid_livable_surface_rows, "livable_surface"] = pd.NA

    print(f"\n{'=' * 60}")
    print("Initializing invalid livable surface nulling")
    print(f"{'=' * 60}")
    print("Rows checked:", len(df))
    print(
        f"Rows with livable_surface below {config.MIN_SURFACE}:",
        int(invalid_livable_surface_rows.sum()),
    )

    return df_work   

# Nulls garages field if number of garages is suspiciously high
def null_invalid_garages(df):
    df_work = df.copy()
    df_work["garages"] = pd.to_numeric(
        df_work["garages"],
        errors="coerce",
    )

    invalid_garage_rows = (
        df_work["garages"].notna()
        & (df_work["garages"] > config.MAX_GARAGES)
    )

    df_work.loc[invalid_garage_rows, "garages"] = pd.NA

    print(f"\n{'=' * 60}")
    print("Initializing invalid garages nulling")
    print(f"{'=' * 60}")
    print("Rows checked:", len(df))
    print(
        f"Rows with garages above {config.MAX_GARAGES}:",
        int(invalid_garage_rows.sum()),
    )

    return df_work

# Nulls facades field if number of facades is suspiciously high
def null_invalid_facades(df):
    df_work = df.copy()
    df_work["facades"] = pd.to_numeric(
        df_work["facades"],
        errors="coerce",
    )

    invalid_facade_rows = (
        df_work["facades"].notna()
        & (df_work["facades"] > 4)
    )

    df_work.loc[invalid_facade_rows, "facades"] = pd.NA

    print(f"\n{'=' * 60}")
    print("Initializing invalid facades nulling")
    print(f"{'=' * 60}")
    print("Rows checked:", len(df))
    print("Rows with facades above 4:", int(invalid_facade_rows.sum()))

    return df_work

# Nulls total floor field if number of total floors is suspiciously high
def null_invalid_total_floor(df):
    df_work = df.copy()

    df_work["number_of_floors"] = pd.to_numeric(
        df_work["number_of_floors"],
        errors="coerce",
    )

    invalid_total_floor_rows = (
        df_work["number_of_floors"].notna()
        & (df_work["number_of_floors"] > config.MAX_FLOOR)
    )

    df_work.loc[invalid_total_floor_rows, "number_of_floors"] = pd.NA

    print(f"\n{'=' * 60}")
    print("Initializing invalid total floor nulling")
    print(f"{'=' * 60}")
    print("Rows checked:", len(df))
    print(
        f"Rows with number_of_floors above {config.MAX_FLOOR}:",
        int(invalid_total_floor_rows.sum()),
    )

    return df_work

# Nulls floor field if it is suspiciously high 
# or above total number of floor for that listing
def null_invalid_floor(df):
    df_work = df.copy()

    df_work["floor"] = pd.to_numeric(
        df_work["floor"],
        errors="coerce",
    )
    df_work["number_of_floors"] = pd.to_numeric(
        df_work["number_of_floors"],
        errors="coerce",
    )

    floor_above_max_rows = (
        df_work["floor"].notna()
        & (df_work["floor"] > config.MAX_FLOOR)
    )

    floor_above_total_rows = (
        df_work["floor"].notna()
        & df_work["number_of_floors"].notna()
        & (df_work["floor"] > df_work["number_of_floors"])
    )

    invalid_floor_rows = floor_above_max_rows | floor_above_total_rows

    df_work.loc[invalid_floor_rows, "floor"] = pd.NA

    print(f"\n{'=' * 60}")
    print("Initializing invalid floor nulling")
    print(f"{'=' * 60}")
    print("Rows checked:", len(df))
    print(
        f"Rows with floor above {config.MAX_FLOOR}:",
        int(floor_above_max_rows.sum()),
    )
    print(
        "Rows where floor is above number_of_floors:",
        int(floor_above_total_rows.sum()),
    )
    print("Total floor rows nulled:", int(invalid_floor_rows.sum()))

    return df_work

# Orchestrator that executes all above steps
def main():
    df = load_data()
    df = swap_coordinates(df)
    df = null_invalid_coordinates(df)
    df = remove_safe_duplicates(df)
    df = remove_suspicious_duplicates(df)
    df = remove_invalid_price(df)
    df = null_invalid_livable_surface(df)
    df = null_invalid_garages(df)
    df = null_invalid_facades(df)
    df = null_invalid_total_floor(df)
    df = null_invalid_floor(df)
    df.to_csv(config.DATA_AUDITED, index=False)
    print(f"\nAudited raw dataset saved to: {config.DATA_AUDITED}")
    print("Final rows:", len(df))

if __name__ == "__main__":
    main()
