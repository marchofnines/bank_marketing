import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='notebook'

def plot_percentage_barplots(df, subp_titles, legend_title, figure_title='', target='y', row_height=200):
    """
    Create percentage bar plots for each feature in a DataFrame against a target column.
    
    Parameters:
    - df: pandas DataFrame, the dataset containing the features and target
    - subp_titles: list of str, titles for the subplots
    - legend_title: str, title for the legend
    - figure_title: str, title for the entire figure (default is an empty string)
    - target: str, the target column name (default is 'y')
    - row_height: int, height of each subplot row (default is 200)
    
    Returns:
    - None: The function shows the plot.
    """
    # Extract feature names, removing the target column
    features = df.columns.tolist()
    features.remove(target)  
    # Determine the number of rows for subplots
    num_rows = len(features)
    colors = {'yes': px.colors.qualitative.Dark24[0], 'no': px.colors.qualitative.Dark24[1]}

    # Create subplots with specified number of rows
    fig = make_subplots(rows=num_rows, cols=1, subplot_titles=subp_titles)
    
    # Iterate through each feature to create subplots
    for idx, feature in enumerate(features):
        # Calculate the percentages
        percentages = pd.crosstab(df[feature], df[target], normalize='index') * 100
        
        # Create bar plot
        for i, outcome in enumerate(df[target].unique()):
            fig.add_trace(
                go.Bar(name=outcome, x=percentages.index, y=percentages[outcome], marker_color=colors[outcome], 
                      text = round(percentages[outcome],0)),
                row=idx + 1, col=1
            )
            if idx > 0:  # Hide legend for all but the first subplot
                fig.data[-1].showlegend = False

        # Update axis titles
        fig['layout'][f'xaxis{idx + 1}'].title.text = subp_titles[idx]
        fig['layout'][f'yaxis{idx + 1}'].title.text = 'Percentage (%)'
        fig['layout'][f'xaxis{idx + 1}'].tickfont.size = 14  
        fig['layout'][f'yaxis{idx + 1}'].tickfont.size = 14  
    
    # Update layout
    fig.update_layout(
        height=num_rows * row_height,
        width=1000,
        title=figure_title,
        legend_title=legend_title,
        legend=dict(x=1, y=1, xanchor='right', yanchor='bottom')
    )
    # Show plot
    fig.show('notebook')
