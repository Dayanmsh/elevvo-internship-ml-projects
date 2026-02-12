import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

# Load data (user must add CSVs)
try:
    sales = pd.read_csv('train.csv')
    features = pd.read_csv('features.csv')
    stores = pd.read_csv('stores.csv')
except FileNotFoundError:
    sales = features = stores = None

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.H1("Walmart Sales Forecasting Dashboard"),
    html.P("Select a store and date range to explore sales trends."),
    dcc.Dropdown(
        id='store-dropdown',
        options=[{'label': f'Store {i}', 'value': i} for i in (sales['Store'].unique() if sales is not None else [])],
        value=(sales['Store'].unique()[0] if sales is not None else None),
        clearable=False
    ),
    dcc.DatePickerRange(
        id='date-picker',
        min_date_allowed=(pd.to_datetime(sales['Date']).min() if sales is not None else None),
        max_date_allowed=(pd.to_datetime(sales['Date']).max() if sales is not None else None),
        start_date=(pd.to_datetime(sales['Date']).min() if sales is not None else None),
        end_date=(pd.to_datetime(sales['Date']).max() if sales is not None else None),
    ),
    dcc.Graph(id='sales-graph'),
    html.Div(id='summary')
], fluid=True)

@app.callback(
    [Output('sales-graph', 'figure'), Output('summary', 'children')],
    [Input('store-dropdown', 'value'), Input('date-picker', 'start_date'), Input('date-picker', 'end_date')]
)
def update_dashboard(store, start_date, end_date):
    if sales is None:
        return go.Figure(), "Data not found. Please add the CSV files."
    df = sales[sales['Store'] == store]
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    fig = px.line(df, x='Date', y='Weekly_Sales', title=f'Store {store} Weekly Sales')
    summary = f"Total sales: ${df['Weekly_Sales'].sum():,.2f} | Records: {len(df)}"
    return fig, summary

if __name__ == '__main__':
    app.run_server(debug=True)
