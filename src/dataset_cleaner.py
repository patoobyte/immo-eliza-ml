import pandas as pd
import config
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_info_columns", 200)
pd.set_option("display.float_format", "{:.3f}".format)

df = pd.read_csv(config.DATA_AUDITED)

# Checks the dataset's shape before processing
# print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
# df.head()
# df.info()

## STEP 1 : Dropping unusable columns ##
COLS_TO_DROP = {
    "property_id" : "Unique identifier",
    "immovlan_id" : "Unique identifier",
    "url" : "Unique identifier",
    "source" : "Common to all for now",
    "scrape_date" : "Metadata",
    "html_path" : "Metadata",
    "transaction_type" : "Common to all for now",
	"cadastral_income" : "Incomplete extraction and low expected relevance",
	"number_of_kitchens" : "Empty data",
	"street" : "Too granular",
	"house_number" : "Too granular",
	"locality" : "Duplicates postal_code",
	"showers" : "Too granular",
    "urbanism_infraction_description" : ">90% missingness",
    "gasoil_tank_certificate_validity" : ">90% missingness",
    "electricity_connection" : ">90% missingness",
    "barbecue" : ">90% missingness",
    "level_e" : ">90% missingness",
    "service_apartment" : ">90% missingness",
    "individual_gas_meter" : ">90% missingness",
    "veranda_surface" : ">90% missingness",
    "adaptations_for_elderlies" : ">90% missingness",
    "individual_electric_meter" : ">90% missingness",
    "total_surface" : ">90% missingness",
    "opportunity_for_professional" : ">90% missingness",
    "dressing_surface" : ">90% missingness",
    "bi_hourly_counter" : ">90% missingness",
    "electrical_certificate_validity" : ">90% missingness",
    "dressing" : ">90% missingness",
    "bike_storage" : ">90% missingness",
    "water_softener" : ">90% missingness",
    "electronic_key" : ">90% missingness",
    "motorized_garage_door" : ">90% missingness",
    "individual_water_meter" : ">90% missingness",
    "gasoil_tank_certificate" : ">90% missingness",
    "attic_surface" : ">90% missingness",
    "garage_surface" : ">90% missingness",
    "office_surface" : ">90% missingness",
    "low_energy_house" : ">90% missingness",
    "cellar_surface" : ">90% missingness",
    "wash_room" : ">90% missingness",
    "solar_thermic_panels" : ">90% missingness",
    "water_sensitive_area" : ">90% missingness",
    "diningroom_surface" : ">90% missingness",
    "balcony" : ">90% missingness",
    "ground_depth" : ">90% missingness",
    "diningroom" : ">90% missingness",
    "garden_orientation" : ">90% missingness",
    "floor_heating" : ">90% missingness",
    "as_build_certificate" : ">90% missingness",
    "buildable_surface" : ">90% missingness",
    "veranda" : ">90% missingness",
    "bedroom_1_surface" : "Part of living surface",
    "bedroom_2_surface" : "Part of living surface",
    "bedroom_3_surface" : "Part of living surface",
	"bedroom_4_surface" : "Part of living surface",
    "bedroom_5_surface" : "Part of living surface",
	"kitchen_surface" : "Part of living surface",
	"primary_energy_consumption" : "EPC offers more reliable data",
	"running_water" : "Weak expected signal",
	"preemption_right" : "Weak expected signal",
	"epc_validity_date" : "Weak expected signal",
	"gas_connection" : "Weak expected signal",
	"sewer_connection" : "Weak expected signal",
    "epc_reference" : "Unique identifier",
	"co2_emission" : "Too many unrealistic values",
	"urbanism_affectation" : "Possible regional bias - Most data come from Flanders",
	"yearly_primary_energy_consumption" : "EPC offers more reliable data",
	"frontage_width" : "Weak expected signal",
	"building_permission_granted" : "Weak expected signal",
	"bathroom_surface" : "Part of living surface",
	"kitchen_type" : "Suspicious - Seems inconsistent and weak expected signal",
	"g_score" : ">85% missing & Possible regional bias - Most data come from Flanders",
	"p_score" : ">85% missing & Possible regional bias - Most data come from Flanders",
	"terrain_width_at_roadside" : "Weak expected signal",
	"living_room_surface" : "Part of living surface",
    "protected_heritage" : "Weak expected signal",
}

print(f"Cleaning - Step 1: Dropping unusable features...")
df = df.drop(columns=COLS_TO_DROP, errors="ignore")
print(f"Dropped {len(COLS_TO_DROP)} column(s): {COLS_TO_DROP}")
print(f"Remaining shape after Step 1: {df.shape}")

## STEP 2 : Converting data types ##
print(f"Cleaning - Step 2: Converting data types...")

# 2-1 : Convert date data to date format
df["date_posted"] = pd.to_datetime(df["date_posted"])

# Extract date : keep year and month, drops day
df["post_year"] = df["date_posted"].dt.year
df["post_month"] = df["date_posted"].dt.month

# Drop original date_posted column
df = df.drop(columns=["date_posted"])
df.info()

#STEP 3 - 
# garages = set above >= 5 as missing (suspicious!)
# convert parkings to yes/no not actual count
# if floor > number_of_floors: set floor to missing
# Availability = immediately, missing or not_immediately (anything else)
# EPC 
# building_state = maybe group more

##  Saving to CSV
path = Path(config.DATA_CLN)
path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(path, index=False, encoding="utf-8")


#Testing area
#df["price"].dtype