import pandas as pd
import config
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_info_columns", 200)
pd.set_option("display.float_format", "{:.3f}".format)

df = pd.read_csv(config.DATA_RAW)

# Checks the dataset's shape before processing
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
df.head()
df.info()

# Remove irrelevant columns
COLS_TO_DROP = {
    "property_id" : "Unique identifier",
    "immovlan_id" : "Unique identifier",
    "url" : "Unique identifier",
    "source" : "Common to all for now",
    "scrape_date" : "Storage data",
    "html_path" : "Storage data",
    "transaction_type" : "Common to all for now",
}

df = df.drop(columns=COLS_TO_DROP, errors="ignore")
print(f"Dropped {len(COLS_TO_DROP)} column(s): {COLS_TO_DROP}")
print(f"Remaining shape: {df.shape}")
df.info()



#Testing area
df["price"].dtype