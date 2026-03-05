import pandas as pd
import numpy as np
from pymongo import MongoClient
from sqlalchemy import create_engine
from urllib.parse import quote_plus
import os
from dotenv import load_dotenv

# Load env
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = quote_plus(os.getenv("DB_PASSWORD", ""))
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME_PHISHING", "phishing_db")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")

SQL_URI = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_PATH = os.path.join(BASE_DIR, "data", "website_phishing.csv")

def connect_sql():
    return create_engine(SQL_URI)

def load_csv(path):
    print(" Loading CSV...")
    return pd.read_csv(path)

def insert_into_mongo(df):
    print("\n Connecting to MongoDB...")
    client = MongoClient(MONGO_URI)
    db = client["phishing_db"]
    collection = db["raw_data"]
    collection.drop()  
    print(" Inserting raw CSV rows into MongoDB...")
    collection.insert_many(df.to_dict(orient="records"))
    print(" Inserted", collection.count_documents({}), "documents into MongoDB!")
    return collection

def read_from_mongo():
    print("\n Reading raw data from MongoDB...")
    client = MongoClient(MONGO_URI)
    db = client["phishing_db"]
    collection = db["raw_data"]
    df = pd.DataFrame(list(collection.find()))
    df = df.drop(columns=["_id"], errors="ignore")
    return df

def clean_transform(df):
    print("\n Cleaning & Transforming data...")
    df = df.dropna()
    # maps to phishing labels
    if "status" in df.columns:
        df["label"] = (df["status"] == "phishing").astype(int)
    
    id_cols = ["url", "status"]
    feature_cols = [col for col in df.columns if col not in id_cols + ["label"]]
    
    # Simple numeric conversion/cleaning
    for col in feature_cols:
         df[col] = pd.to_numeric(df[col], errors='coerce')

    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
    print(" Cleaning Complete! Shape:", df.shape)
    return df

def store_in_mysql(df):
    print("\n Storing cleaned data into MySQL...")
    engine = connect_sql()
    df.to_sql("clean_urls", engine, if_exists="replace", index=False)
    print(" Cleaned data successfully stored in MySQL!")
    return engine

def run_pipeline():
    print(" STARTING WEBSITE PHISHING PIPELINE (Original Dataset) ")
    try:
        df = load_csv(CSV_PATH)
        insert_into_mongo(df)
        df_raw = read_from_mongo()
        df_clean = clean_transform(df_raw)
        store_in_mysql(df_clean)
        print(" PIPELINE FINISHED SUCCESSFULLY")
    except Exception as e:
        print(f"Pipeline failed: {e}")

if __name__ == "__main__":
    run_pipeline()
