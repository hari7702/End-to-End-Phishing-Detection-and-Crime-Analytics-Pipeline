PROJECT: PHISHING AND CRIME ANALYSIS PREDICTION

OVERVIEW

This project is an analytical dashboard designed to analyze cyber threats (phishing) and US crime statistics.
It uses three distinct datasets, processed through dedicated pipelines, and visualizes the results using a Dash web application.
Machine learning models are integrated for predictive analysis.
Data ingestion and processing leverage Prefect for orchestration.


1. DATASET ANALYSIS

DATASET 1: WEBSITE PHISHING DATA
- Source File: data/website_phishing.csv
- Pipeline Script: pipelines/website_phishing_pipeline.py
- Model Script:models\website_phishing_rf.py

DATASET 2: US CRIME STATISTICS
- Source File: data/state_crime.json
- Pipeline Script: pipelines/crime_pipeline.py
- Model Script:models\crime_model.py


DATASET 3: PHISHING URL DATA
- Source File: data/phishing_url.csv
- Pipeline Script: pipelines/phishing_url_pipeline.py
- Model Script:models\phishing_url_catboost.py



2. PROJECT STRUCTURE

- app.py: Main application file containing the Dash layout, callbacks, and visualization logic.
- orchestration.py: Prefect flow script to trigger and manage all three data pipelines sequentially.
- pipelines/: Directory containing the ETL scripts.
  - crime_pipeline.py: ETL for US Crime data.
  - phishing_url_pipeline.py: ETL for Phishing URL data.
  - website_phishing_pipeline.py: ETL for Website Phishing data.
- models/: Scripts for training Machine Learning models.
  - crime_model.py: Logic for crime rate prediction.
  - phishing_url_catboost.py: CatBoost training script for URL analysis.
  - website_phishing_rf.py: Random Forest training script for website phishing.
- data/: Directory for storing raw CSV/JSON files.
- requirements.txt: List of Python dependencies.
- .env: Configuration file for database credentials.


3. ORCHESTRATION

The project uses Prefect to automate the data ingestion process.
The 'orchestration.py' file defines a flow 'Main Data Ingestion Flow' that executes the three pipelines as tasks:
1. Crime Data ETL
2. Website Phishing Data ETL
3. Phishing URL Data ETL

To run the full pipeline automation:
python orchestration.py


4. INSTALLATION AND USAGE

STEP 1: INSTALL DEPENDENCIES
Run the following command in terminal:
pip install -r requirements.txt

STEP 2: CONFIGURE DATABASE
Ensure MySQL and MongoDB instances are running.
Create a '.env' file with database credentials (DB_USER, DB_PASSWORD, etc.).
MANUALLY create the databases in MySQL before running the pipelines.
Run the following SQL commands in MySQL workbench/terminal:
  CREATE DATABASE phishing_db;
  CREATE DATABASE cyber_db;
MongoDB databases will be created automatically.

STEP 3: RUN DATA PIPELINES
Run the full pipeline using the orchestration script:
python orchestration.py

Alternatively, run scripts individually:
- python pipelines/website_phishing_pipeline.py
- python pipelines/crime_pipeline.py
- python pipelines/phishing_url_pipeline.py

STEP 4: LAUNCH DASHBOARD
Run the application:
python app.py

Access the dashboard in web browser at: http://127.0.0.1:8050/
