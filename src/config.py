# src/config.py
from pathlib import Path

## ===== PATHS =====
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATA_RAW = ROOT / "data" / "raw" / "listings_raw_20260629.csv"
DATA_AUDITED = ROOT / "data" / "raw" / "listings_raw_20260629_audited.csv"
DATA_CLN = ROOT / "data" / "clean" / "listings_clean.csv"

## ===== FEATURES CLASSIFICATION =====
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
    "indoor_parking",
    "outdoor_parking",
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
]

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
]

CATEGORICAL_FEATURES = [
    "property_type",
    "province",
    "region",
    "kitchen_equipment",
    "building_state",
    "heating_type",
    "glazing_type",
    "facade_orientation",
    "terrace_orientation",
    "flooding_area_clean",
    "epc_quality",
    "postal_code",
    "category",
]

## ===== MISSING HANDLING FOR BINARY FEATURES =====
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

BINARY_MISSING_AS_UNKNOWN = [
    "new_construction",
    "vat",
    "currently_leased",
    "furnished",
    "electrical_certificate",
    "available_immediately",
]

## ===== MISC =====
RANDOM_SEED = 13
