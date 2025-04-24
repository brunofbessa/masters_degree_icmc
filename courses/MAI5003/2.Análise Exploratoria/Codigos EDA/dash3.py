#!/usr/bin/env python
# coding: utf-8

# In[4]:


import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
import os


os.chdir("C:/Users/Benedito/Desktop")
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv('dash_1610.csv')

fig = px.scatter(df, x="avg_income", y="avg_amt_credit",
                 size="default_rate", color="grupos", hover_name="NAME_FAMILY_STATUS",
                 log_x=True, size_max=60)

app.layout = html.Div([
    dcc.Graph(
        id='income-vs-amt-credit',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)

