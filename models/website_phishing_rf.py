import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from dash import html
import plotly.graph_objects as go
import plotly.express as px
from lime.lime_tabular import LimeTabularExplainer


last_trained_model = None
last_features = None

def train_website_phishing_rf(df):
    """
    Trains Random Forest on the Website Phishing dataframe.
    Returns: metrics_html, confusion_matrix_fig, roc_curve_fig, lime_fig, lime_table
    """
    global last_trained_model, last_features
    
    if df.empty:
        return html.Div("No Data"), go.Figure(), go.Figure(), go.Figure(), html.Div()

    remove_cols = [
        "domain_age", "domain_registration_length", "dns_record",
        "google_index", "page_rank", "web_traffic"
    ]

    df_model = df.copy()
    
    for col in remove_cols:
        if col in df_model.columns:
            df_model.drop(columns=[col], inplace=True)

    if "label" not in df_model.columns:
        return html.Div("Label column missing"), go.Figure(), go.Figure(), go.Figure(), html.Div()

    y = df_model["label"]
    drop_cols = ["label", "url", "status", "whois_registered_domain", "entropy"] # Drop non-numeric/targets
    X = df_model.drop(columns=[c for c in drop_cols if c in df_model.columns], errors="ignore")
    X = X.select_dtypes(include=["int64", "float64"])
    
    feat_names = X.columns.tolist()
    last_features = feat_names

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    last_trained_model = model

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 1. Metrics Report
    report = html.Div([
        html.H4("Random Forest Metrics", className="text-info"),
        html.P(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}"),
        html.P(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}"),
        html.P(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}"),
        html.P(f"F1 Score : {f1_score(y_test, y_pred, zero_division=0):.4f}"),
        html.P(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}"),
    ])

    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Pred Legit", "Pred Phishing"], 
        y=["Actual Legit", "Actual Phishing"],
        colorscale="Blues",
        text=cm.astype(str),
        texttemplate="%{text}",
        textfont={"size": 16, "color": "black"}
    ))
    fig_cm.update_layout(title="Confusion Matrix", template="plotly_dark")

    # 3. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC Curve"))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Baseline", line=dict(dash="dash")))
    fig_roc.update_layout(title="ROC Curve", template="plotly_dark")
    
    # 4. LIME Explanation 
    try:
        sample_idx = np.random.randint(0, len(X_test))
        X_sample = X_test.iloc[[sample_idx]]
        
        explainer = LimeTabularExplainer(
            X_train.values, 
            feature_names=feat_names, 
            class_names=['Legit', 'Phishing'], 
            mode='classification'
        )
        exp = explainer.explain_instance(X_sample.values[0], model.predict_proba)
        lime_df = pd.DataFrame(exp.as_list(), columns=["Feature", "Contribution"])
        
        fig_lime = go.Figure()
        fig_lime.add_bar(y=lime_df["Feature"], x=lime_df["Contribution"], orientation="h", 
                         marker_color=["green" if v > 0 else "red" for v in lime_df["Contribution"]])
        fig_lime.update_layout(template="plotly_dark", title=f"LIME Explanation (Sample {sample_idx})")
        
        table_lime = html.Table([
            html.Thead(html.Tr([html.Th("Feature"), html.Th("Contribution")])),
            html.Tbody([html.Tr([html.Td(row["Feature"]), html.Td(f"{row['Contribution']:.6f}")]) for _, row in lime_df.iterrows()])
        ], className="table table-dark")
        
    except Exception as e:
        fig_lime = go.Figure()
        fig_lime.update_layout(title=f"LIME Error: {str(e)}")
        table_lime = html.Div("LIME Failed")

    return report, fig_cm, fig_roc, fig_lime, table_lime
