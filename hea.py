import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash,dcc,html
from dash.dependencies import Input, Output, State

#import data
df = pd.read_csv('/Users/dominicwelti/Library/CloudStorage/Dropbox/Master_Thesis/data_npj/Graph_data.csv')

def calculate_rel_e(df:pd.DataFrame):
    '''Calculate normalized error. Returns dataframe with new column ["Normalized error"].'''
    df['Normalized error']=0
    for (i, error_e, error_f) in zip(df.index, df['Error_E'],df['Error_F']):
        df.loc[i,'Normalized error']=(1/2)*( error_f/max(df['Error_F']) + error_e/max(df['Error_E']) )
    return df

df=calculate_rel_e(df)

# add distinctive colours for each model
df['Colour']=''
colours=['blue','red','green','purple','orange', 'turquoise', 'black']

for i, name in enumerate(df['Type'].unique()):
    df.loc[(df['Type']==name),['Colour']]=colours[i]

def buildPlot(df):
    fig = go.Figure(data=[])
    i=0 # index for trace
    traces=[] 
    for engine in df['Engine'].unique():
        for suite in df['MD Suite'].unique():
            for kind in df['Type'].unique():
                subset=df.query('Type==@kind & Engine==@engine & `MD Suite`==@suite')
                if len(subset)>=1:
                    i+=1
                    fig.add_trace(go.Scatter(
                        x=subset['Cost'],
                        y=subset['Normalized error'],
                        mode='markers',
                        name=kind,
                        text=subset['Model']+'<br>'+'Engine: '+subset['MD Suite']+'-'+subset['Engine'],
                        #textposition="center bottom",
                        hovertemplate="%{text}"+"<br>Computational cost: %{x} μs"+"<br>Normalized error: %{y}",
                        legendgroup=kind,
                        marker={'color':subset['Colour']},
                        #legendgrouptitle=kind,
                        #yaxis=subset['y axes'].unique()[0],
                    ))
                    trace={
                        'Trace': i,
                        'Type': kind,
                        'Engine': engine,
                        'MD Suite': suite,
                    }
                    traces.append(trace)

    traces=pd.DataFrame(traces) # convert list of dictionaries to DataFrame

    # Add annotation
    fig.update_xaxes(title_text='Computational cost of inference [<i>μs/atom/step</i>]',type="log",minor_ticks='inside')
    fig.update_layout(yaxis={'title':'Normalized error'},
                      title={'text':'Interactive graph for comparing performances between different machine learning interatomic potentials.'},
                      legend={'title':'Model'},
                    font_family="Serif",font_size=18,
                    template='ggplot2',
                    width=1000,
                    height=600) #,'position':0

    names=set()
    fig.for_each_trace(
        lambda trace:
            trace.update(showlegend=False)
            if (trace.name in names) else names.add(trace.name))
    
    return fig

app = Dash(__name__)
server=app.server

app.layout = html.Div(
    [
        dcc.Dropdown(
            id="md",
            options=[{"label": x, "value": x} for x in df["MD Suite"].unique()]+[{'label': 'All MD suites', 'value': 'all_values'}],
            value='all_values',
        ),
        dcc.Dropdown(
            id="engine",
            options=[{"label": x, "value": x} for x in df["Engine"].unique()]+[{'label': 'All engines', 'value': 'all_values'}],
            value='all_values',
        ),
        dcc.Graph(id="graph1"),
    ]
)

@app.callback(
    Output("graph1", "figure"),
    Input("md", "value"),
    Input("engine", "value"),
)
def update_graph1(md, engine):
    #print(md, engine)
    if ('all' in engine) and ('all' in md):
        return buildPlot(df)
    elif 'all' in engine:
        return buildPlot(df.query('`MD Suite`==@md'))
    elif 'all' in md:
        return buildPlot(df.query('`Engine`==@engine'))
    else:
        return buildPlot(df.query('Engine==@engine & `MD Suite`==@md'))


    #return buildPlot(df.loc[df["MD Suite"].eq(md)])


# Run app and display result inline in the notebook
if __name__ == '__main__':
    app.run_server(mode="inline")