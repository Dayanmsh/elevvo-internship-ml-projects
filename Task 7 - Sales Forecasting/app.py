import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import numpy as np

# Load data (user must add CSVs)
try:
    sales = pd.read_csv('train.csv')
    features = pd.read_csv('features.csv')
    stores = pd.read_csv('stores.csv')
except FileNotFoundError:
    sales = features = stores = None

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

def get_store_options():
    if sales is not None:
        return [{'label': f'Store {i}', 'value': i} for i in sorted(sales['Store'].unique())]
    return []

def get_dept_options(store):
    if sales is not None and store is not None:
        return [{'label': f'Department {i}', 'value': i} for i in sorted(sales[sales['Store'] == store]['Dept'].unique())]
    return []

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H2("Walmart Sales Forecasting Dashboard", className="text-primary fw-bold mb-2"), width=8),
        dbc.Col(html.Img(src="https://1000logos.net/wp-content/uploads/2017/06/Walmart-Logo.png", height="48px"), width=4, style={"textAlign": "right"})
    ], align="center", className="mt-3 mb-2"),
    dbc.Row([
        dbc.Col([
            html.Label("Select Store:"),
            dcc.Dropdown(
                id='store-dropdown',
                options=get_store_options(),
                value=(sales['Store'].unique()[0] if sales is not None else None),
                clearable=False
            ),
        ], width=3),
        dbc.Col([
            html.Label("Select Department:"),
            dcc.Dropdown(
                id='dept-dropdown',
                options=get_dept_options(sales['Store'].unique()[0] if sales is not None else None),
                value=None,
                clearable=True,
                placeholder="All Departments"
            ),
        ], width=3),
        dbc.Col([
            html.Label("Date Range:"),
            dcc.DatePickerRange(
                id='date-picker',
                min_date_allowed=(pd.to_datetime(sales['Date']).min() if sales is not None else None),
                max_date_allowed=(pd.to_datetime(sales['Date']).max() if sales is not None else None),
                start_date=(pd.to_datetime(sales['Date']).min() if sales is not None else None),
                end_date=(pd.to_datetime(sales['Date']).max() if sales is not None else None),
            ),
        ], width=4),
        dbc.Col([
            html.Label("Options:"),
            dbc.Checklist(
                options=[{"label": "Show Moving Average", "value": "ma"}],
                value=[],
                id="ma-toggle",
                switch=True,
            ),
        ], width=2)
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([
            dbc.CardGroup([
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Total Sales", className="card-title"),
                        html.H3(id="total-sales", className="text-success")
                    ])
                ]),
                dbc.Card([
                    dbc.CardBody([
                        html.H5("Records", className="card-title"),
                        html.H3(id="record-count", className="text-info")
                    ])
                ]),
            ])
        ], width=12)
    ], className="mb-3"),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='sales-graph', config={"displayModeBar": True})
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.Div(id='summary', className="mt-3")
        ], width=12)
    ]),
    dbc.Row([
        dbc.Col([
            html.Div("Data not found. Please add train.csv, features.csv, and stores.csv.", id="data-warning", className="text-danger", style={"display": "none"})
        ])
    ])
], fluid=True)

@app.callback(
    Output('dept-dropdown', 'options'),
    Output('dept-dropdown', 'value'),
    Input('store-dropdown', 'value')
)
def update_dept_options(store):
    options = get_dept_options(store)
    return options, None

@app.callback(
    [Output('sales-graph', 'figure'), Output('summary', 'children'), Output('total-sales', 'children'), Output('record-count', 'children'), Output('data-warning', 'style')],
    [Input('store-dropdown', 'value'), Input('dept-dropdown', 'value'), Input('date-picker', 'start_date'), Input('date-picker', 'end_date'), Input('ma-toggle', 'value')]
)
def update_dashboard(store, dept, start_date, end_date, ma_toggle):
    if sales is None:
        return go.Figure(), "Data not found. Please add the CSV files.", "-", "-", {"display": "block"}
    df = sales[sales['Store'] == store]
    if dept is not None:
        df = df[df['Dept'] == dept]
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    df = df.sort_values('Date')
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Weekly_Sales'], mode='lines+markers', name='Weekly Sales', line=dict(color='#0074D9')))
    if 'ma' in ma_toggle and len(df) > 2:
        df['MA_4'] = df['Weekly_Sales'].rolling(window=4).mean()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA_4'], mode='lines', name='4-Week MA', line=dict(color='#FF851B', dash='dash')))
    fig.update_layout(
        title=f"Store {store} Sales{' - Dept ' + str(dept) if dept else ''}",
        xaxis_title="Date",
        yaxis_title="Weekly Sales ($)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    total_sales = f"${df['Weekly_Sales'].sum():,.2f}" if not df.empty else "$0.00"
    record_count = f"{len(df):,}" if not df.empty else "0"
    summary = f"<b>Store:</b> {store}"
    if dept:
        summary += f" | <b>Department:</b> {dept}"
    summary += f" | <b>Date Range:</b> {start_date} to {end_date}"
    return fig, summary, total_sales, record_count, {"display": "none"}

if __name__ == '__main__':
    app.run_server(debug=True)
