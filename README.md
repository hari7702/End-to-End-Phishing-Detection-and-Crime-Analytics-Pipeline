# End-to-End Phishing Detection and Crime Analytics Pipeline

## Overview

This project presents a complete data engineering and machine learning pipeline designed to analyze phishing threats and U.S. crime trends. The system integrates multiple datasets, performs automated ETL processing, stores data using both SQL and NoSQL databases, and applies machine learning models for predictive analytics.

An interactive dashboard built using Dash and Plotly allows users to explore the processed data, visualize trends, and evaluate model performance.

The project demonstrates how a full data workflow can be implemented—from raw data ingestion to machine learning predictions and visual analytics.

---

## Key Features

* Automated ETL pipelines for multiple datasets
* Integration of structured and semi-structured data sources
* Hybrid database architecture using MongoDB and MySQL
* Machine learning models for phishing detection and crime prediction
* Interactive data visualization dashboard
* Pipeline orchestration using Prefect
* Reproducible end-to-end data pipeline

---

## Technologies Used

| Category               | Tools                  |
| ---------------------- | ---------------------- |
| Programming            | Python                 |
| Databases              | MongoDB, MySQL         |
| Data Processing        | Pandas, NumPy          |
| Machine Learning       | Scikit-learn, CatBoost |
| Data Visualization     | Dash, Plotly           |
| Pipeline Orchestration | Prefect                |
| Data Access            | SQLAlchemy, PyMongo    |

---

## Datasets Used

### Website Phishing Dataset

A dataset containing engineered features extracted from website URLs and webpage content to identify phishing attacks.

### U.S. State Crime Dataset

A semi-structured JSON dataset containing crime statistics across U.S. states covering multiple decades.

### Phishing URL Dataset

A structured dataset consisting of URL-based features used to classify phishing and legitimate URLs.

---

## System Architecture

The project follows a complete end-to-end pipeline:

1. Raw data is ingested from CSV and JSON sources
2. Raw data is stored in databases
3. ETL pipelines clean and transform the datasets
4. Processed datasets are stored in MySQL
5. Machine learning models are trained on processed data
6. Results are visualized through an interactive dashboard

---

## Project Structure

```
project/
│
├── app.py
├── orchestration.py
├── requirements.txt
│
├── data/
│   ├── website_phishing.csv
│   ├── phishing_url.csv
│   └── state_crime.json
│
├── pipelines/
│   ├── website_phishing_pipeline.py
│   ├── crime_pipeline.py
│   └── phishing_url_pipeline.py
│
├── models/
│   ├── website_phishing_rf.py
│   ├── crime_model.py
│   └── phishing_url_catboost.py
```

---

## Installation

### 1. Clone the Repository

```
git clone https://github.com/yourusername/phishing-crime-analytics-pipeline.git
cd phishing-crime-analytics-pipeline
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

---

## Database Setup

Ensure the following databases are running:

* MySQL
* MongoDB

Create the required MySQL databases:

```
CREATE DATABASE phishing_db;
CREATE DATABASE cyber_db;
```

MongoDB databases will be created automatically when the pipelines run.

---

## Running the Data Pipelines

Run the Prefect orchestration script to execute all pipelines:

```
python orchestration.py
```

This will execute the following pipelines:

* Crime Data ETL Pipeline
* Website Phishing Data Pipeline
* Phishing URL Data Pipeline

Each pipeline ingests raw data, performs preprocessing, and stores the cleaned datasets in MySQL.

---

## Running Individual Pipelines (Optional)

You can run each pipeline separately if required:

```
python pipelines/website_phishing_pipeline.py
python pipelines/crime_pipeline.py
python pipelines/phishing_url_pipeline.py
```

---

## Launching the Dashboard

Run the Dash application:

```
python app.py
```

Then open your browser and go to:

```
http://127.0.0.1:8050/
```

The dashboard provides visual analytics for phishing detection results and crime statistics.

---

## Machine Learning Models

| Dataset               | Model Used                    | Task                  |
| --------------------- | ----------------------------- | --------------------- |
| Website Phishing      | Random Forest                 | Binary Classification |
| Phishing URL Features | CatBoost                      | Binary Classification |
| U.S. Crime Data       | Decision Tree / Random Forest | Regression            |

---

## Example Outputs

The dashboard provides visualizations such as:

* URL feature distribution analysis
* Crime rate comparisons across states
* Feature importance visualizations
* Machine learning model performance metrics

---

## Model Performance

The models achieved strong performance during evaluation:

* Website Phishing Detection Accuracy: **92.3%**
* Phishing URL Detection Accuracy: **98.5%**
* Crime Prediction Models: **R² up to 0.997**

These results demonstrate the effectiveness of feature-based machine learning approaches for phishing detection and predictive analytics.

---

## Future Improvements

Possible extensions for the project include:

* Real-time streaming data pipelines
* Automated model retraining workflows
* SHAP-based explainable AI integration
* Deployment of the dashboard as a cloud web application
* Model monitoring and drift detection

---

## Author

Hari Prasath; 
MSc Artificial Intelligence
