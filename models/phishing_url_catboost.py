import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from catboost import CatBoostClassifier
from dash import html
import plotly.graph_objects as go
import plotly.express as px

def train_phishing_catboost(df):
    """
    Trains CatBoost on the extended phishing dataset.
    Returns metrics HTML and figures.
    """
    if df.empty:
         return html.Div("No Data Available"), go.Figure(), go.Figure()

    SEED = 42

    df=df.drop(columns=["URLSimilarityIndex","IsHTTPS","LineOfCode","NoOfSelfRedirect","LargestLineLength","NoOfExternalRef","NoOfImage","NoOfOtherSpecialCharsInURL","NoOfSelfRef","SpacialCharRatioInURL","NoOfCSS","LetterRatioInURL","NoOfJS","HasSocialNet","URLLength","NoOfLettersInURL","HasCopyrightInfo"])

    
    if "label" not in df.columns:
        return html.Div("Target column 'label' missing"), go.Figure(), go.Figure()
        
    X = df.drop(columns=["label", "label_str"], errors="ignore")
    y = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop"
    )

    model_cb = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False 
    )

    pipe = Pipeline([
        ("prep", preprocess),
        ("clf", model_cb)
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    # Metrics
    metrics_txt = html.Div([
        html.H4("CatBoost Results", className="text-info"),
        html.P(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}"),
        html.P(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}"),
        html.P(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}"),
        html.P(f"F1-score : {f1_score(y_test, y_pred, zero_division=0):.4f}"),
        html.P(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}"),
    ])
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Pred Phishing", "Pred Legit"], 
        y=["Actual Phishing", "Actual Legit"],
        colorscale="Viridis",
        text=cm.astype(str),
        texttemplate="%{text}",
        textfont={"size": 16, "color": "black"}
    ))
    fig_cm.update_layout(title="Confusion Matrix (CatBoost)", template="plotly_dark")
    
    # Feature Importance (Top 10)
    cb_extracted = pipe.named_steps["clf"]
    importances = cb_extracted.get_feature_importance()
    feat_names = cat_cols + num_cols
    
    if len(importances) == len(feat_names):
        s_imp = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(10) # Changed to 10
        fig_imp = px.bar(x=s_imp.values, y=s_imp.index, orientation='h', title="Top 10 Features", text_auto='.2f') # Added text
        fig_imp.update_layout(template="plotly_dark", yaxis={'categoryorder':'total ascending'})
    else:
        fig_imp = go.Figure()
        fig_imp.update_layout(title="Feature Importance (Size Mismatch)", template="plotly_dark")

    return metrics_txt, fig_cm, fig_imp
