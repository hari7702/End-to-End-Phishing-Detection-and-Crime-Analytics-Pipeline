# End-to-End-Phishing-Detection-and-Crime-Analytics-Pipeline
End-to-end data engineering and machine learning pipeline for phishing detection and US crime analytics using MongoDB, MySQL, Prefect, and Dash.

phishing-crime-analytics/
│
├── app.py
├── orchestration.py
├── requirements.txt
├── README.md
├── report.pdf
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
│
└── images/
    ├── dashboard1.png
    └── dashboard2.png
