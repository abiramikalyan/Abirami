import dash 
from dash import dcc, html  
from dash.dependencies import Input, Output 
import plotly.express as px
import pandas as pd

df = pd.read_csv(r'D:\Download\Thyroid Cancer Prediction\Thyroid_Diff.csv')

app = dash.Dash(__name__)

fig1 = px.histogram(df, x="Age", color="Recurred", barmode="overlay", 
                    title="Age Distribution by Recurrence",
                    labels={"Age": "Age", "Recurred": "Recurrence"})

fig2 = px.pie(df, names='Thyroid Function', title="Thyroid Function Distribution")

fig3 = px.scatter(df, x="Age", y="T", color="Recurred", symbol="Stage", 
                  title="Age vs Tumor Stage (T) with Recurrence",
                  labels={"Age": "Age", "T": "Tumor Size (T)"})

app.layout = html.Div([
    html.H1("Thyroid Cancer Dashboard", style={'text-align': 'center'}),
    
    dcc.Graph(id='age-histogram', figure=fig1),
    
    html.Div([
        dcc.Graph(id='thyroid-function-pie', figure=fig2),
        dcc.Graph(id='age-tumor-scatter', figure=fig3)
    ], style={'display': 'flex', 'justify-content': 'space-around'}),
    
    html.Label("Select Tumor Size (T)"),
    dcc.Dropdown(
        id='tumor-dropdown',
        options=[{'label': t, 'value': t} for t in df['T'].unique()],
        value='T1a'
    ),
    
    dcc.Graph(id='tumor-stage-chart')
])

@app.callback(
    Output('tumor-stage-chart', 'figure'),
    [Input('tumor-dropdown', 'value')]
)
def update_chart(selected_tumor):
    filtered_df = df[df['T'] == selected_tumor]
    fig = px.histogram(filtered_df, x="Age", color="Recurred", title=f"Age Distribution for Tumor Size {selected_tumor}")
    return fig

if __name__ == '__main__':
    app.run(debug=True)

