from prefect import flow, task
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importing pipelines
from pipelines.crime_pipeline import run_pipeline as crime_etl
from pipelines.phishing_url_pipeline import process_pipeline as phishing_url_etl
from pipelines.website_phishing_pipeline import run_pipeline as website_phishing_etl

@task(name="Crime Data ETL")
def task_crime_etl():
    print(" Starting Crime ETL Task ")
    crime_etl()
    print(" Crime ETL Task Completed ")

@task(name="Phishing URL Data ETL")
def task_phishing_url_etl():
    print(" Starting Phishing URL Dataset ETL Task ")
    phishing_url_etl()
    print(" Phishing URL ETL Task Completed ")

@task(name="Website Phishing Data ETL")
def task_website_phishing_etl():
    print(" Starting Website Phishing Dataset ETL Task ")
    website_phishing_etl()
    print(" Website Phishing ETL Task Completed ")

@flow(name="Main Data Ingestion Flow")
def main_flow():
    
    t1 = task_crime_etl()
    t2 = task_website_phishing_etl()
    t3 = task_phishing_url_etl()
    
    return t1, t2, t3

if __name__ == "__main__":
    main_flow()
