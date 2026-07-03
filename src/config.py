"""
-----------------------------------------
Config file
-----------------------------------------
Central configuration for data paths, feature groups, cleaning thresholds,
and small shared utilities used across preprocessing and model training.
"""

from pathlib import Path

## ===== PATHS =====
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_RAW = ROOT / "data" / "raw" / "listings_raw_20260702.csv"
DATA_AUDITED = ROOT / "data" / "raw" / "listings_raw_20260702_audited.csv"
DATA_CLN = ROOT / "data" / "clean" / "listings_clean_20260702.csv"
DATA_STATBEL_COMMUNE = ROOT / "data" / "raw" / "FR_immo_statbel_annee.xlsx"
DATA_POSTAL_NIS = ROOT / "data" / "raw" / "codes-ins-nis-postaux-belgique.csv"
MODEL_PATH = ROOT / "models"

## ===== FEATURES CLASSIFICATION =====
# Features passed to numeric pipeline
NUMERIC_FEATURES = [
    "latitude",
    "longitude",
    "bedrooms",
    "livable_surface",
    "bathrooms",
    "toilets",
    "floor",
    "build_year",
    "facades",
    "garden_surface",
    "terrace_surface",
    "garages",
    "number_of_floors",
    "total_land_surface",
    "nursery_nearest_walk_m",
    "preschool_nearest_walk_m",
    "elementary_school_nearest_walk_m",
    "high_school_nearest_walk_m",
    "bus_stop_nearest_walk_m",
    "tram_stop_nearest_walk_m",
    "train_station_nearest_walk_m",
    "train_station_nearest_drive_m",
    "motorway_nearest_drive_m",
    "airport_nearest_drive_m",
    "charging_station_nearest_drive_m",
    "supermarket_nearest_walk_m",
    "supermarket_count",
    "car_sharing_nearest_walk_m",
    "bike_sharing_nearest_walk_m",
    "post_year",
    "post_month",
    "statbel_commune_median",
    "statbel_transaction_count",
    "property_age",
]

# Binary features 
# Missing values are handled by the two policy lists below
BINARY_FEATURES = [
    "new_construction",
    "vat",
    "currently_leased",
    "furnished",
    "heat_pump",
    "solar_panels",
    "air_conditioning",
    "electrical_certificate",
    "garden",
    "terrace",
    "garage",
    "cellar",
    "swimming_pool",
    "elevator",
    "access_for_disabled",
    "entry_phone",
    "attic",
    "alarm",
    "security_door",
    "rain_water_tank",
    "fireplace",
    "electric_charging_station",
    "hammam_sauna_jacuzzi",
    "domotica",
    "available_immediately",
    "indoor_parking",
    "outdoor_parking",
]

# Nominal categorical features
CATEGORICAL_FEATURES = [
    "property_type",
    "province",
    "region",
    "heating_type",
    "glazing_type",
    "facade_orientation",
    "terrace_orientation",
    "flooding_area_clean",
    "category",
]

# Ordered categorical features
ORDINAL_FEATURES = {
    "epc_quality":       ["bad", "poor", "good", "excellent"],
    "kitchen_equipment": ["Not equipped", "Partially equipped", "Fully equipped", "Super equipped"],
    "building_state":    ["To restore", "To renovate", "Normal", "Fully renovated", "Excellent", "New"],
}

# Categorical features handled with target encoding
TARGET_ENCODED_FEATURES = [
    "postal_code",
]

## ===== MISSING HANDLING FOR BINARY FEATURES =====
# Missing means absence
BINARY_MISSING_AS_ZERO = [
    "heat_pump",
    "solar_panels",
    "air_conditioning",
    "garden",
    "terrace",
    "garage",
    "cellar",
    "swimming_pool",
    "elevator",
    "access_for_disabled",
    "entry_phone",
    "attic",
    "alarm",
    "security_door",
    "rain_water_tank",
    "fireplace",
    "electric_charging_station",
    "hammam_sauna_jacuzzi",
    "domotica",
]

# Missing is preserved as an Unknown category
BINARY_MISSING_AS_UNKNOWN = [
    "new_construction",
    "vat",
    "currently_leased",
    "furnished",
    "electrical_certificate",
    "available_immediately",
    "indoor_parking",
    "outdoor_parking",
]

## ===== MISC =====
RANDOM_SEED = 13
MIN_VALID_PRICE = 30000
MIN_SURFACE = 12
MAX_GARAGES = 4
MAX_FLOOR = 50

## ===== TINY UTILS =====
def format_euros(value):
    return f"{value:,.0f}"
