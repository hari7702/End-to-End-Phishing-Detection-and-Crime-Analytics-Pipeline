# End-to-End Phishing Detection and Crime Analytics Pipeline

## Overview

This project implements a complete data engineering and machine learning workflow for phishing detection and US crime analytics. 

The system integrates multiple datasets, automated ETL pipelines, database storage, machine learning models, and an interactive dashboard for visualization and analysis.

The objective of the project is to demonstrate how a full data pipeline can be built, starting from raw data ingestion to predictive analytics and dashboard-based reporting.

---

## Key Features

• Automated ETL pipelines for multiple datasets  
• Integration of structured and semi-structured data  
• Database storage using MongoDB and MySQL  
• Machine learning models for phishing detection and crime prediction  
• Interactive analytics dashboard built with Dash and Plotly  
• Pipeline orchestration using Prefect  

---

## Technologies Used

- Python
- MongoDB
- MySQL
- Prefect
- Pandas
- Scikit-learn
- CatBoost
- Dash
- Plotly
- SQLAlchemy

---

## Datasets Used

### 1. Website Phishing Dataset
Contains engineered features extracted from phishing and legitimate websites used for classification.

### 2. US State Crime Dataset
Semi-structured JSON dataset containing crime statistics across US states from 1960–2019.

### 3. Phishing URL Dataset
Structured dataset containing URL-level features used to classify phishing and legitimate URLs.

---

## Project Architecture

The system follows a complete end-to-end data pipeline:

1. Raw datasets are ingested from CSV and JSON sources
2. Data is stored in MongoDB and MySQL databases
3. ETL pipelines clean and transform the data
4. Machine learning models are trained for classification and regression
5. Results are visualized through an interactive dashboard

---

## Project Structure
project/
│
├── app.py
├── orchestration.py
├── requirements.txt
│
├── data/
│ ├── website_phishing.csv
│ ├── phishing_url.csv
│ └── state_crime.json
│
├── pipelines/
│ ├── website_phishing_pipeline.py
│ ├── crime_pipeline.py
│ └── phishing_url_pipeline.py
│
├── models/
│ ├── website_phishing_rf.py
│ ├── crime_model.py
│ └── phishing_url_catboost.py
