import config
import pandas as pd

## Readies the raw csv file for data cleaning

## SETUP
LAT_MIN = 49.5
LAT_MAX = 51.6
LON_MIN = 2.5
LON_MAX = 6.5

def load_data():
    df = pd.read_csv(config.DATA_RAW)
    return df

## STEP 1 : Duplicates check
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

def remove_missing_price(df):
    df_work = df.copy()
    df_work["price"] = pd.to_numeric(df_work["price"], errors="coerce")

    invalid_price_rows = df_work["price"].isna() | (df_work["price"] <= 0)
    df_cleaned = df_work.loc[~invalid_price_rows].copy()

    print(f"\n{'=' * 60}")
    print("Initializing missing price removal")
    print(f"{'=' * 60}")
    print("Rows before:", len(df))
    print("Rows after:", len(df_cleaned))
    print("Rows removed:", len(df) - len(df_cleaned))
    print("Missing price rows:", int(df_work["price"].isna().sum()))
    print("Non-positive price rows:", int((df_work["price"] <= 0).sum()))

    return df_cleaned

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

def main():
    df = load_data()
    df = swap_coordinates(df)
    df = remove_safe_duplicates(df)
    df = remove_suspicious_duplicates(df)
    df = remove_missing_price(df)    
    df.to_csv(config.DATA_AUDITED, index=False)
    print(f"\nAudited raw dataset saved to: {config.DATA_AUDITED}")
    print("Final rows:", len(df))

if __name__ == "__main__":
    main()
