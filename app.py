import dash
from dash import html, dcc, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from joblib import load
import numpy as np
import os
from dotenv import load_dotenv

#  CUSTOM MODULES 

from models.crime_model import train_crime_models
from models.phishing_url_catboost import train_phishing_catboost
from models.website_phishing_rf import train_website_phishing_rf 
from lime.lime_tabular import LimeTabularExplainer

# LOAD ENV
load_dotenv()
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = quote_plus(os.getenv("DB_PASSWORD", "root@123"))
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME_PHISHING = os.getenv("DB_NAME_PHISHING", "phishing_db")
DB_NAME_CRIME = os.getenv("DB_NAME_CRIME", "cyber_db")



#   DATA LOADING


engine1 = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME_PHISHING}")

#  1. WEBSITE PHISHING DATA 
try:
    df1 = pd.read_sql("SELECT * FROM clean_urls", engine1)
except:
    df1 = pd.DataFrame(columns=["label", "url", "status", "entropy"]) 

import math
if not df1.empty and "entropy" not in df1.columns:
    def calc_entropy(s):
        prob = [float(s.count(c)) / len(s) for c in dict.fromkeys(list(s))]
        return -sum(p * math.log(p, 2) for p in prob)
    df1["entropy"] = df1["url"].astype(str).apply(calc_entropy)

if not df1.empty:
    df1["status"] = df1["label"].map({0: "Legitimate", 1: "Phishing"})
    numeric_cols1 = df1.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_cols1 = [c for c in numeric_cols1 if c != "label"]
else:
    numeric_cols1 = []

#  TRAIN WEBSITE PHISHING MODEL 
print("Training Website Phishing Model...")
report_web, cm_web, roc_web, lime_fig_web, lime_table_web = train_website_phishing_rf(df1)


#  2. PHISHING URL DATA 
try:
    df_extra = pd.read_sql("SELECT * FROM processed_phishing_extra", engine1)
    # 1 = Legitimate, 0 = Phishing
    df_extra["label_str"] = df_extra["label"].map({1: "Legitimate", 0: "Phishing"})
except:
    df_extra = pd.DataFrame()

#  TRAIN PHISHING URL MODEL 
print("Training Phishing URL CatBoost Model...")
report_cat, cm_cat, imp_cat = train_phishing_catboost(df_extra)


#  3. CRIME DATA 
engine2 = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME_CRIME}")

try:
    df2 = pd.read_sql("SELECT * FROM crime_processed", engine2)
    states = sorted(df2["state"].unique())
    years = sorted(df2["year"].unique())
except:
    df2 = pd.DataFrame()
    states = []
    years = []



#   APP CONFIG & COMPONENTS


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE], suppress_callback_exceptions=True)
server = app.server

def card_component(title, graph_id=None, content=None, figure=None):
    body_children = [html.H4(title, className="text-info")]
    
    # Prioritize passing a direct figure if available (for static plots)
    if figure:
        body_children.append(dcc.Graph(figure=figure, style={"height": "450px"}))
    elif graph_id:
        body_children.append(dcc.Graph(id=graph_id, style={"height": "450px"}))
        
    if content:
        body_children.append(content)
        
    return dbc.Card(
        dbc.CardBody(body_children),
        className="mb-4 shadow-lg"
    )



#   LAYOUTS


#  1. WEBSITE PHISHING LAYOUT 
website_phishing_layout = dbc.Container([
    html.H1(" Website Phishing Analytics", className="text-center text-light mb-4"),

    dbc.Row([
        dbc.Col(card_component(" Label Distribution", "label-dist"), width=6),
        dbc.Col([
            html.Label("Select Feature:", className="text-light"),
            dcc.Dropdown(id="hist-select", options=numeric_cols1, value=numeric_cols1[0] if numeric_cols1 else None),
            card_component(" Top 5 Correlated Features", "hist-plot")   
        ], width=6),
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Label("X-axis", className="text-light"),
            dcc.Dropdown(id="scatter-x", options=numeric_cols1, value=numeric_cols1[0] if numeric_cols1 else None),
            html.Label("Y-axis", className="text-light"),
            dcc.Dropdown(id="scatter-y", options=numeric_cols1, value=numeric_cols1[1] if len(numeric_cols1) > 1 else None),
            card_component(" Scatter Plot", "scatter-plot")
        ], width=6),

        dbc.Col([
            html.Label("Select KDE Feature:", className="text-light"),
            dcc.Dropdown(id="kde-select", options=numeric_cols1, value=numeric_cols1[0] if numeric_cols1 else None),
            card_component(" KDE Plot", "kde-plot")
        ], width=6)
    ]),

    dbc.Row([
        dbc.Col(card_component(" Entropy vs URL Length", "entropy-length"), width=6),
        dbc.Col(card_component(" Entropy vs Page Rank", "entropy-pagerank"), width=6),
    ]),

    dbc.Row([
        dbc.Col(card_component(" Special Characters", "domain-age"), width=6),
        dbc.Col(card_component(" WHOIS Comparison", "whois-bar"), width=6),
    ]),

    html.Hr(),
    html.H2(" Model Results (Random Forest)", className="text-center text-info"),

    dbc.Card(
        dbc.CardBody([
            html.H4(" ML Metrics Report", className="text-info"),
            html.Div(report_web, className="text-light fs-5") # Inserted directly
        ]),
        className="shadow mb-4"
    ),

    dbc.Row([

        dbc.Col(card_component(" Confusion Matrix", figure=cm_web), width=6),
        dbc.Col(card_component(" ROC Curve", figure=roc_web), width=6),
    ]),

    dbc.Row([
        dbc.Col(card_component(" LIME Explanation", figure=lime_fig_web), width=7),
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H4(" LIME Contribution Table", className="text-info"),
                    html.Div(lime_table_web) # Inserted directly
                ])
            ), width=5
        )
    ]),
], fluid=True)


#  2. CRIME DASHBOARD 
crime_layout = dbc.Container([
    html.H1(" US Crime Dashboard", className="text-center text-light mb-4"),

    dbc.Row([
        dbc.Col([
            html.Label("Select State"),
            dcc.Dropdown(id="state_select", options=[{"label": s, "value": s} for s in states], value="Alabama")
        ], width=5),
        dbc.Col([
            html.Label("Select Year"),
            dcc.Dropdown(id="year_select", options=[{"label": str(y), "value": y} for y in years], value=years[-1] if years else None)
        ], width=5),
    ], className="mb-4 justify-content-center"),

    dbc.Row([
        dbc.Col(dcc.Graph(id="crime_trend"), width=6),
        dbc.Col(dcc.Graph(id="violent_vs_property"), width=6)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(dcc.Graph(id="top_states"), width=6),
        dbc.Col(dcc.Graph(id="crime_pie_chart"), width=6)
    ], className="mb-4"),

    # FIXED LAYOUT at bottom
    dbc.Row([
        dbc.Col([
            html.Label("Violent Crime Breakdown State:"),
            dcc.Dropdown(
                id="violent_state_select",
                options=[{"label": s, "value": s} for s in states],
                value="Alabama",
                className="mb-2"
            ),
            dcc.Graph(id="grouped_crime_chart", style={"height": "400px"}) 
        ], width=7),
        
        dbc.Col([
            # This is populated by callback 'train_crime_cb' based on 'year_select'
            # Since year_select has a default value, this will run on load.
            dbc.Card(
                dbc.CardBody(id="metrics_block", style={"overflowY": "auto", "height": "450px"}),
                className="h-100 shadow"
            )
        ], width=5)
    ])
], fluid=True)


#  3. PHISHING URL LAYOUT 
fig_url_1 = go.Figure()
fig_url_2 = go.Figure()
fig_url_3 = go.Figure()
fig_url_4 = go.Figure()

if not df_extra.empty:
    # 1. Mean URL Length
    df_len = df_extra.groupby("label_str")["URLLength"].mean().reset_index()
    fig_url_1 = px.bar(df_len, x="label_str", y="URLLength", title="Mean URLLength", color="label_str", 
                  template="plotly_dark", text_auto='.2f')

    # 2. HasTitle
    df_title = df_extra.groupby(["HasTitle", "label_str"]).size().reset_index(name="Count")
    fig_url_2 = px.bar(df_title, x="HasTitle", y="Count", color="label_str", barmode="group", 
                  title="HasTitle vs Label", template="plotly_dark", text_auto=True)

    # 3. Letter/Digit Ratio
    fig_url_3 = px.scatter(df_extra, x="LetterRatioInURL", y="DegitRatioInURL", color="label_str", 
                      title="Letter/Digit Ratio", opacity=0.5, template="plotly_dark")
    
    # 4. Top 10 TLDs
    top_tlds = df_extra['TLD'].value_counts().head(10).index
    df_tld = df_extra[df_extra['TLD'].isin(top_tlds)].groupby(["TLD", "label_str"]).size().reset_index(name="Count")
    df_tld = df_tld.sort_values("Count", ascending=False)
    fig_url_4 = px.bar(df_tld, x="TLD", y="Count", color="label_str", barmode="group", 
                  title="Top 10 TLDs", template="plotly_dark", text_auto=True)
    fig_url_4.update_xaxes(categoryorder='total descending')

phishing_url_layout = dbc.Container([
    html.H1(" Phishing URL Analysis", className="text-center text-light mb-4"),
    
    dbc.Row([
        dbc.Col(card_component(" Mean URL Length by Class", figure=fig_url_1), width=6),
        dbc.Col(card_component(" HasTitle vs Label", figure=fig_url_2), width=6),
    ]),
    
    dbc.Row([
        dbc.Col(card_component(" Letter Ratio vs Digit Ratio", figure=fig_url_3), width=6),
        dbc.Col(card_component(" Top 10 TLDs by Label", figure=fig_url_4), width=6),
    ]),
    
    html.Hr(),
    html.H2(" CatBoost Model Results", className="text-center text-info"),
    
    dbc.Row([
        dbc.Col([
            html.Div(report_cat)
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col(card_component(" Confusion Matrix (CatBoost)", figure=cm_cat), width=6),
        dbc.Col(card_component(" Feature Importance (Top 10)", figure=imp_cat), width=6),
    ])
], fluid=True)


#  MAIN TABS 
app.layout = dbc.Container([
    dcc.Tabs([
        dcc.Tab(label=" Website Phishing", children=[website_phishing_layout], 
                style={"background": "#333", "color": "cyan"}, selected_style={"background": "#111", "color": "white"}),
        dcc.Tab(label=" Crime Dashboard", children=[crime_layout], 
                style={"background": "#333", "color": "cyan"}, selected_style={"background": "#111", "color": "white"}),
        dcc.Tab(label=" Phishing URL Analysis", children=[phishing_url_layout], 
                style={"background": "#333", "color": "cyan"}, selected_style={"background": "#111", "color": "white"})
    ])
], fluid=True)



#   CALLBACKS


#  WEBSITE PHISHING CB 
@app.callback(Output("label-dist", "figure"), Input("hist-select", "value"))
def dist(_):
    if df1.empty: return go.Figure()
    temp = df1["label"].value_counts().reset_index()
    temp.columns = ["label", "count"]
    fig = px.bar(temp, x="label", y="count", color="label", title="Label Distribution")
    fig.update_layout(template="plotly_dark")
    return fig

@app.callback(Output("hist-plot", "figure"), Input("hist-select", "value"))
def correlation_plot(_):
    if df1.empty: return go.Figure()
    df_corr = df1.copy()
    df_corr["status_num"] = df_corr["status"].map({"Legitimate": 0, "Phishing": 1})
    num_df = df_corr.select_dtypes(include=["int64", "float64"])
    correlations = num_df.corr()["status_num"].abs().sort_values(ascending=False)
    top5 = correlations[2:7]
    fig = go.Figure()
    fig.add_bar(x=top5.values[::-1], y=top5.index[::-1], orientation="h",
                text=[f"{v:.3f}" for v in top5.values[::-1]], textposition="auto")
    fig.update_layout(title="Top 5 Correlated Features", template="plotly_dark")
    return fig

@app.callback(Output("scatter-plot", "figure"), [Input("scatter-x", "value"), Input("scatter-y", "value")])
def scatter(x, y):
    if df1.empty or not x or not y: return go.Figure()
    fig = px.scatter(df1, x=x, y=y, color="status", title=f"{x} vs {y}")
    fig.update_layout(template="plotly_dark")
    return fig

@app.callback(Output("kde-plot", "figure"), Input("kde-select", "value"))
def kde(feature):
    if df1.empty or not feature: return go.Figure()
    fig = go.Figure()
    fig.add_histogram(x=df1[df1.label == 0][feature], name="Legitimate", opacity=0.5, histnorm="probability density")
    fig.add_histogram(x=df1[df1.label == 1][feature], name="Phishing", opacity=0.5, histnorm="probability density")
    fig.update_layout(barmode="overlay", template="plotly_dark", title=f"KDE Plot — {feature}")
    return fig

@app.callback(Output("domain-age", "figure"), Input("hist-select", "value"))
def domain_age(_):
    if df1.empty: return go.Figure()
    special_cols = ['nb_dots', 'nb_hyphens', 'nb_qm', 'nb_at']
    existing = [c for c in special_cols if c in df1.columns]
    if not existing: return go.Figure(layout={"title": "Data Missing"})
    legit_mean = df1[df1['status'] == 'Legitimate'][existing].mean()
    phish_mean = df1[df1['status'] == 'Phishing'][existing].mean()
    fig = go.Figure()
    fig.add_bar(x=existing, y=legit_mean, name="Legitimate")
    fig.add_bar(x=existing, y=phish_mean, name="Phishing")
    fig.update_layout(title="Special Character Comparison", barmode="group", template="plotly_dark")
    return fig

@app.callback(Output("entropy-length", "figure"), Input("kde-select", "value"))
def entropy_length(_):
    if df1.empty: return go.Figure()
    fig = px.scatter(df1, x="length_url", y="entropy", color="status", title="Entropy vs URL Length")
    fig.update_layout(template="plotly_dark")
    return fig

@app.callback(Output("entropy-pagerank", "figure"), Input("hist-select", "value"))
def entropy_rank(_):
    if df1.empty: return go.Figure()
    fig = px.scatter(df1, x="page_rank", y="entropy", color="status", title="Entropy vs Page Rank")
    fig.update_layout(template="plotly_dark")
    return fig

@app.callback(Output("whois-bar", "figure"), Input("hist-select", "value"))
def whois_bar(_):
    if df1.empty: return go.Figure()
    agg = df1.groupby("whois_registered_domain").agg({"label": ["sum", "count"], "domain_age": "mean", "page_rank": "mean"}).reset_index()
    agg.columns = ["Registrar", "Phishing", "Total", "AvgAge", "AvgRank"]
    agg["Legitimate"] = agg["Total"] - agg["Phishing"]
    top = agg.nlargest(8, "Total")
    fig = go.Figure()
    fig.add_bar(x=top["Registrar"], y=top["Phishing"], name="Phishing")
    fig.add_bar(x=top["Registrar"], y=top["Legitimate"], name="Legitimate")
    fig.add_bar(x=top["Registrar"], y=top["AvgAge"], name="Avg Age")
    fig.add_bar(x=top["Registrar"], y=top["AvgRank"], name="Avg Rank")
    fig.update_layout(barmode="group", template="plotly_dark", title="WHOIS Multi-Metric Comparison")
    return fig


#  CRIME CB 

@app.callback(Output("crime_trend", "figure"), Input("state_select", "value"))
def update_trend(s):
    if df2.empty: return go.Figure()
    d = df2[df2["state"] == s].sort_values("year")
    return px.line(d, x="year", y=["rate_property_all", "rate_violent_all", "total_crime_rate"], title=f"Crime Trend — {s}")

@app.callback(Output("violent_vs_property", "figure"), Input("state_select", "value"))
def update_scatter(s):
    if df2.empty: return go.Figure()
    d = df2[df2["state"] == s]
    return px.scatter(d, x="rate_property_all", y="rate_violent_all", size="population", color="year", title=f"Violent vs Property Crime — {s}")

@app.callback(Output("top_states", "figure"), Input("year_select", "value"))
def update_top(y):
    if df2.empty: return go.Figure()
    d = df2[df2["year"] == y].sort_values("total_crime_rate", ascending=False).head(10)
    return px.bar(d, x="state", y="total_crime_rate", title=f"Top 10 States — {y}", color="total_crime_rate")

@app.callback(Output("crime_pie_chart", "figure"), Output("grouped_crime_chart", "figure"), 
              Input("state_select", "value"), Input("violent_state_select", "value"), Input("year_select", "value"))
def update_extra(main_state, violent_state, year):
    if df2.empty: return go.Figure(), go.Figure()
    d1 = df2[(df2["state"] == main_state) & (df2["year"] == year)]
    d2 = df2[(df2["state"] == violent_state) & (df2["year"] == year)]
    if d1.empty or d2.empty: return go.Figure(), go.Figure()
    
    pie = px.pie(names=["Property Crime", "Violent Crime"], values=[d1["rate_property_all"].values[0], d1["rate_violent_all"].values[0]], title=f"Crime Composition — {main_state}")
    group = go.Figure(data=[
        go.Bar(name=k, x=[violent_state], y=[d2[f"rate_violent_{k.lower()}"].values[0]]) for k in ["Assault", "Robbery", "Murder", "Rape"]
    ])
    group.update_layout(barmode="group", title=f"Violent Crime Breakdown — {violent_state}")
    return pie, group

@app.callback(Output("metrics_block", "children"), Input("year_select", "value"))
def train_crime_cb(_):
    if df2.empty: return "No Data for Modeling"
    return train_crime_models(df2)

if __name__ == "__main__":
    app.run(debug=True)
