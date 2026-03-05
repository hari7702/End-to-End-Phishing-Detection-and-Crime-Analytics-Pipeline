import pandas as pd
import os
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = quote_plus(os.getenv("DB_PASSWORD", ""))
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME_PHISHING", "phishing_db")

SQL_URI = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "phishing_url.csv")

def connect_db():
    try:
        engine = create_engine(SQL_URI)
        print("Connected to MySQL!")
        return engine
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def process_pipeline():
    print(" STARTING PHISHING URL PIPELINE (New Dataset) ")
    engine = connect_db()
    if not engine:
        return

    # 1. Load Raw CSV
    print(f"Loading raw data from {DATA_FILE}...")
    try:
        df = pd.read_csv(DATA_FILE)
        print(f"Loaded {df.shape[0]} rows.")
    except FileNotFoundError:
        print("File not found.")
        return

    # Store Raw to MySQL
    print("Storing raw data to MySQL table 'raw_phishing_extra'...")
    df.to_sql("raw_phishing_extra", engine, if_exists="replace", index=False)
    
    # Retrieve from MySQL
    print("Retrieving raw data from MySQL...")
    df_raw = pd.read_sql("SELECT * FROM raw_phishing_extra", engine)
    
    # Data Preprocessing
    print("Preprocessing data...")
    print(f"Duplicates before: {df_raw.duplicated().sum()}")
    print(f"Nulls before: {df_raw.isnull().sum().sum()}")
    
    # Drop specified columns
    drop_cols = ["FILENAME", "URL", "Domain", "Title"]
    existing_drop_cols = [c for c in drop_cols if c in df_raw.columns]
    df2 = df_raw.drop(columns=existing_drop_cols)
    
    # Handle duplicates/nulls 
    df2.drop_duplicates(inplace=True)
    df2.fillna(0, inplace=True) 
    
    print(f"Processed shape: {df2.shape}")

    # Store Processed to MySQL
    print("Storing processed data to MySQL table 'processed_phishing_extra'...")
    df2.to_sql("processed_phishing_extra", engine, if_exists="replace", index=False)
    
    # Retrieve Processed 
    print("Verifying storage...")
    df_check = pd.read_sql("SELECT * FROM processed_phishing_extra LIMIT 5", engine)
    print(df_check.head())
    
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    process_pipeline()
