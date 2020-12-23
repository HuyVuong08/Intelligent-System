# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'fontWeight': 'bold'
}

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'fontWeight': 'bold'
}

app.layout = html.Div([
    html.Div([html.H1('House Price Prediction Using Linear Regression')], style={'text-align': 'center'}),
    dcc.Tabs(id='tabs-example', value='tab-outlierAna', children=[
        dcc.Tab(label='Outlier Analysis', value='tab-outlierAna', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Outlier Analysis Results', value='tab-outlierRes', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Exploratory Data Analytics', value='tab-explore', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Check Correlation Coefficients', value='tab-heatmap', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Error Terms', value='tab-error', style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Model Evaluation', value='tab-evaluate', style=tab_style, selected_style=tab_selected_style),
    ]),
    html.Div(id='tabs-example-content')
])

@app.callback(Output('tabs-example-content', 'children'),
              Input('tabs-example', 'value'))
def render_content(tab):
    if tab == 'tab-outlierAna':
        return html.Div([
            html.H2('Outliers Box Plot'),
            html.Img(src=app.get_asset_url('boxPlotBefore.png')),
        ], style={'text-align': 'center'})
    elif tab == 'tab-outlierRes':
        return html.Div([
            html.H2('After Trimming Outliers'),
            html.Img(src=app.get_asset_url('boxPlot.png')),
        ], style={'text-align': 'center'})
    elif tab == 'tab-explore':
        return html.Div([
            html.H2('Visualize Numeric Variables By Pair Plot'),
            html.Img(src=app.get_asset_url('pairPlot.png')),
            html.H2('Visualize Categorical Variables'),
            html.Img(src=app.get_asset_url('boxPlotCategorical.png')),
        ], style={'text-align': 'center'})
    elif tab == 'tab-heatmap':
        return html.Div([
            html.H2('Heat Map'),
            html.Img(src=app.get_asset_url('heatMap.png')),
        ], style={'text-align': 'center'})
    elif tab == 'tab-evaluate':
        return html.Div([
            html.Br(),
            html.Br(),
            html.Img(src=app.get_asset_url('predictionScatter.png'))
        ], style={'text-align': 'center'})
    elif tab == 'tab-error':
        return html.Div([
            html.Br(),
            html.Br(),
            html.Img(src=app.get_asset_url('errorTerms.png')),
            html.Img(src=app.get_asset_url('errorScatter.png'))
        ], style={'text-align': 'center'})

if __name__ == '__main__':
    app.run_server(debug=True)
