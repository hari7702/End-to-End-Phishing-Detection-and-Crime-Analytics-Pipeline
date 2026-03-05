
import json
import pandas as pd
from pymongo import MongoClient
from sqlalchemy import create_engine
from urllib.parse import quote_plus
import os
from dotenv import load_dotenv

# Load env 
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# CONFIGURATION

#  MongoDB 
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = "cyber_crime_db"
RAW_COLLECTION = "crime_raw"
STRUCT_COLLECTION = "crime_structured"

#  SQL Database (MySQL) 
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = quote_plus(os.getenv("DB_PASSWORD", ""))
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME_CRIME", "cyber_db")

SQL_URI = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
SQL_TABLE = "crime_processed"

#  JSON  File 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
JSON_FILE = os.path.join(BASE_DIR, "data", "state_crime.json")


def run_pipeline():
    #  MONGODB CONNECTION
    print("Connecting to MongoDB...")
    mongo_client = MongoClient(MONGO_URI)
    mongo_db = mongo_client[MONGO_DB]
    raw_coll = mongo_db[RAW_COLLECTION]
    struct_coll = mongo_db[STRUCT_COLLECTION]
    print("MongoDB Connected!")

    # LOAD RAW JSON and INSERT INTO MONGODB
    print(f"\nLoading JSON file from {JSON_FILE}...")
    try:
        with open(JSON_FILE, "r") as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print("Error: JSON file not found.")
        return

    print("Inserting raw JSON data into MongoDB...")
    raw_coll.delete_many({})
    if isinstance(raw_data, list):
        raw_coll.insert_many(raw_data)
    else:
        raw_coll.insert_one(raw_data) 
    print("Inserted raw docs:", raw_coll.count_documents({}))

    # RETRIEVE RAW and FLATTEN / STRUCTURE DATA
    print("\nRetrieving raw data from MongoDB...")
    raw_docs = list(raw_coll.find({}, {"_id": 0}))

    def flatten_record(doc):
        # Specific logic for state_crime.json structure
        d = doc.get("Data", {})
        rates = d.get("Rates", {})
        prop = rates.get("Property", {})
        violent = rates.get("Violent", {})
        
        return {
            "State": doc.get("State"),
            "Year": doc.get("Year"),
            "Population": d.get("Population"),
            "Rate_Property_All": prop.get("All"),
            "Rate_Property_Burglary": prop.get("Burglary"),
            "Rate_Property_Larceny": prop.get("Larceny"),
            "Rate_Property_Motor": prop.get("Motor"),
            "Rate_Violent_All": violent.get("All"),
            "Rate_Violent_Assault": violent.get("Assault"),
            "Rate_Violent_Murder": violent.get("Murder"),
            "Rate_Violent_Rape": violent.get("Rape"),
            "Rate_Violent_Robbery": violent.get("Robbery")
        }

    print("Flattening records...")
    if raw_docs:
        structured_docs = [flatten_record(r) for r in raw_docs]

        struct_coll.delete_many({})
        struct_coll.insert_many(structured_docs)
        print("Inserted structured docs:", struct_coll.count_documents({}))

        # LOAD STRUCTURED DATA INTO PANDAS
        print("\nLoading structured data into pandas DataFrame...")
        df = pd.DataFrame(structured_docs)
        
        # DATA PRE-PROCESSING & TRANSFORMATION
        print("\nStarting data preprocessing...")
        df.columns = df.columns.str.lower().str.replace(" ", "_")
        
        numeric_cols = df.select_dtypes(include="number").columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        df.drop_duplicates(subset=["state", "year"], inplace=True)
        
        df["total_crime_rate"] = df["rate_property_all"] + df["rate_violent_all"]
        df["crime_per_capita"] = (df["total_crime_rate"] / df["population"]) * 100000
        df["violent_to_property_ratio"] = df["rate_violent_all"] / df["rate_property_all"]
        df["decade"] = (df["year"] // 10) * 10

        #  STORE PROCESSED DATA INTO MYSQL
        print("\nConnecting to MySQL database...")
        engine = create_engine(SQL_URI)
        print("Writing processed data into MySQL table...")
        df.to_sql(SQL_TABLE, engine, if_exists="replace", index=False)
        print("Stored processed table:", SQL_TABLE)

    else:
        print("No raw documents found.")

    print("\n DATA INGESTION PIPELINE COMPLETED SUCCESSFULLY ")

if __name__ == "__main__":
    run_pipeline()
