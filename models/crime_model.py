from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

def train_crime_models(df):
    """
    Trains Decision Tree and Random Forest on the crime dataframe.
    Returns a formatted HTML component with metrics.
    """
    from dash import html 

    features = [
        "population", "rate_property_all", "rate_violent_all",
        "crime_per_capita", "violent_to_property_ratio", "decade"
    ]
    target = "total_crime_rate"
    
    # Check features exist
    existing = [f for f in features if f in df.columns]
    if len(existing) < len(features): 
        return html.Div("Missing features data for modeling.")

    X = df[existing]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Decision Tree
    dt = DecisionTreeRegressor(max_depth=10, random_state=42)
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=120,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features="sqrt",
        bootstrap=True,
        random_state=42
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    # Metrics Layout
    return html.Div([
        html.H3("ML Model Metrics", className="text-info"),
        
        html.H5("Decision Tree", className="mt-3"),
        html.P(f"MAE: {mean_absolute_error(y_test, dt_pred):.2f}"),
        html.P(f"RMSE: {np.sqrt(mean_squared_error(y_test, dt_pred)):.2f}"),
        html.P(f"R²: {r2_score(y_test, dt_pred):.3f}"),

        html.H5("Random Forest", className="mt-3"),
        html.P(f"MAE: {mean_absolute_error(y_test, rf_pred):.2f}"),
        html.P(f"RMSE: {np.sqrt(mean_squared_error(y_test, rf_pred)):.2f}"),
        html.P(f"R²: {r2_score(y_test, rf_pred):.3f}"),
    ])
