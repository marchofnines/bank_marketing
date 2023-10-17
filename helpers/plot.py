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

def dec_boundary_mesh(estimator, X1, y, feature1, feature2):
    xx = np.linspace(X1.iloc[:, 0].min(), X1.iloc[:, 0].max(), 50)
    yy = np.linspace(X1.iloc[:, 1].min(), X1.iloc[:, 1].max(), 50)
    XX, YY = np.meshgrid(xx, yy)
    grid = np.c_[XX.ravel(), YY.ravel()]
    labels = pd.factorize(estimator.predict(grid))[0]
    plt.contourf(xx, yy, labels.reshape(XX.shape), cmap = 'twilight', alpha = 0.6)
    sns.scatterplot(data = X1, x = feature1, y = feature2, hue = y,  palette = 'flare')

def can_cast_to_number(series):
        try:
            series.astype(float)
            return True
        except ValueError:
            return False
     
def tick_magic(param_values, max_ticks, ints=True):
    min_val = param_values.min()
    max_val = param_values.max()
    if ints:
        tick_interval = max((max_val - min_val) // (max_ticks - 1), 1)
    else: 
        tick_interval = max((max_val - min_val) / (max_ticks - 1), 1)
    tick_values = np.arange(min_val, max_val + 1, tick_interval)
    return tick_values

#displot is good for multiple columns on a single plot or for faceting on categorical, not numerical
def plot_distributions(dfo, bins=10, binrange=None, outlier_thresh=3,
                       deskew=False, suptitle='', common_fontsize=17, 
                       cols=3, row_height=5, col_width=6, 
                       banner_y=1.08, pad=1.9, kde=False):
    from scipy.stats import zscore
    df = dfo.copy().dropna().select_dtypes(include=np.number).replace('\D', '', regex=True).dropna()
    num_df_cols = df.shape[1]
    rows = num_df_cols // cols + (1 if num_df_cols % cols != 0 else 0)

    plt.clf()
    plt.figure(figsize=(col_width * cols, row_height * rows))
    sns.set_style('whitegrid')

    for i, col in enumerate(df.columns):
        if outlier_thresh is not None and outlier_thresh !=0:
            try:
                col_zscore = zscore(df[col])
                no_outliers = (np.abs(col_zscore) < outlier_thresh)
                df = df[no_outliers]
            except Exception as e:
                print(f"Couldn't remove outliers for column {col}. Error: {e}")
        
        plt.subplot(rows, cols, i+1)
        
        if deskew:
            data = np.log1p(df[col])
            xlabel = f'Log of {col}'
        else:
            data = df[col]
            xlabel = col
            
        if bins and bins!='auto' and not binrange: 
            tickstep = int(np.ceil((max(data)-min(data))/bins))
            binrange = (min(data),max(data)+tickstep)
        elif bins and bins!='auto' and binrange: 
            tickstep = int(np.ceil((max(data)-min(data))/bins))
        else:
            bins = 'auto'
            binrange = None
            
        # Determine bin settings
        sns.histplot(data, bins=bins, binrange=binrange, kde=kde)
        
        if bins=='auto':
            plt.xticks(fontsize=common_fontsize)
        else: 
            plt.xticks(np.arange(min(data),max(data)+tickstep,tickstep))
            
        plt.xlabel(xlabel, fontsize=common_fontsize)
        plt.ylabel(col, fontsize=common_fontsize)
        plt.yticks(fontsize=common_fontsize)
    
    plt.tight_layout(pad=pad)
    if suptitle:
        plt.suptitle(suptitle, fontsize=common_fontsize + 4, weight='bold', x=0.5, y=banner_y-0.06, ha='center')
    plt.show()


def plot_percentage_barplots(df, subp_titles, legend_title, figure_title='', target='y', row_height=200):
    features = df.columns.tolist()
    features.remove(target)  # Remove the target column from the feature list
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
    fig.show()
