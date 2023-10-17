import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from IPython.core.display import display, HTML

def hello_world():
    print("Hello World")
    print("Hello world world d ")

def mystats(df, nulls=False):
    unique_counts = []
    for col in df.columns:
        unique_values = df[col][~df[col].isna() & (df[col] != '') & (df[col] != ' ')].unique()
        unique_counts.append(len(unique_values))
    #summary_data['unique_values_excluding_null_blank_nan'] = unique_counts
    
    stats_df = pd.DataFrame({'blank_spaces': (df==' ').sum(),
                              'empty_strs': (df=='').sum(),
                             'nulls': df.isnull().sum(), 
                             'null_pct': round(100*(df.isnull().sum()) / len(df),2), 
                             'unique_not_null': df.nunique(),
                             'unique_not_null_blank_empty': unique_counts,
                             'dups': [df[col].duplicated().sum() for col in df.columns]
                            }).sort_values(by='null_pct', ascending=False)
    print(df.shape)
    if nulls:
        return stats_df
    else: 
        return stats_df.query('nulls > 0')
    

#find number of zeroes 
#df_edits.eq(0).sum()
#df.isna().sum()
#print(df_edits.level.dtype)
#print(df_edits.level.isna().sum())

#drop dups - currently we have no dups
#duplicate_rows=raw[raw.duplicated()]
#duplicate_rows
#df_edits = raw.drop_duplicates(inplace=False)

import pandas as pd
import numpy as np

def calculate_iqr_bounds(df):
    """
    Calculate the lower and upper bounds for outliers for all numerical columns in a DataFrame.
    
    Parameters:
    - df: DataFrame containing the data
    
    Returns:
    - A DataFrame with columns as index and lower and upper bounds as columns
    """
    # Initialize an empty DataFrame to store the results
    result = pd.DataFrame(columns=["lower_bound", "upper_bound", "num_within_bounds", "num_below_lower", "num_above_upper"])
    
    # Loop through each numerical column in the DataFrame
    for column in df.select_dtypes(include=[np.number]).columns:
        # Calculate Q1, Q3, and IQR
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Calculate the lower and upper bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        num_below_lower = df[df[column] < lower_bound].shape[0]   
        num_above_upper = df[df[column] > upper_bound].shape[0] 
        num_within_bounds = df[(df[column] >= lower_bound)&(df[column] <= upper_bound)].shape[0] 
        
        # Add the results to the DataFrame
        result.loc[column] = [lower_bound, upper_bound, num_within_bounds, num_below_lower, num_above_upper]
        
    return result

def remove_all_outliers_based_on_IQRs(df, factor=1.5):
    """
    Remove outliers from a DataFrame based on IQR, only considering numeric columns.
    
    Parameters:
        df (DataFrame): The original DataFrame.
        factor (float): Factor to multiply the IQR range to set bounds.
        
    Returns:
        DataFrame: A new DataFrame with outliers removed.
    """
    # Select only numeric columns
    df_numeric = df.select_dtypes(include=['number'])
    
    # Calculate Q1, Q3 and IQR for each numeric column
    Q1 = df_numeric.quantile(0.25)
    Q3 = df_numeric.quantile(0.75)
    IQR = Q3 - Q1
    
    # Define bounds for the outliers
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    # Identify the outliers
    not_outliers = ((df_numeric >= lower_bound) & (df_numeric <= upper_bound)).all(axis=1)
    
    # Remove outliers from the DataFrame based on numeric columns
    df_cleaned = df[not_outliers]
    print(f'Returned DataFrame has: {df_cleaned.shape[0]} samples remainining')
    return df_cleaned

# Create a sampl

def remove_col_outliers_based_on_IQRs(df, column_name):
    """
    Removes outliers from a specific column in a dataframe using the IQR method.
    
    Parameters:
    - df (pd.DataFrame): The input dataframe.
    - column_name (str): The column from which outliers should be removed.
    
    Returns:
    - pd.DataFrame: A dataframe with outliers removed from the specified column.
    """
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    print(f'lower_bound: {lower_bound} and upper_bound: {upper_bound}')
    df_cleaned = df.query(f"{lower_bound} <= {column_name} <= {upper_bound}") 
    #df_cleaned = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]
    print(f'Removed {len(df)- len(df_cleaned)} rows')
    
    return df_cleaned

import pandas as pd
from scipy import stats

def remove_outliers_based_on_z_scores(df, z_threshold=3):
    """
    Remove outliers from a DataFrame based on Z-score, only considering numeric columns.
    
    Parameters:
        df (DataFrame): The original DataFrame.
        z_threshold (float): Z-score threshold for outlier detection.
        
    Returns:
        DataFrame: A new DataFrame with outliers removed.
    """
    
    # Select only numeric columns
    df_numeric = df.select_dtypes(include=['number'])
    
    # Calculate Z-scores for numeric columns
    z_scores = pd.DataFrame(stats.zscore(df_numeric, nan_policy='omit'), columns=df_numeric.columns)
    
    # Get boolean DataFrame where True indicates the presence of an outlier in numeric columns
    outliers = (z_scores.abs() > z_threshold)
    
    # Remove outliers from the DataFrame based on numeric columns
    df_cleaned = df[~outliers.any(axis=1)]
    
    return df_cleaned


def consolidate_values(df, col, list_of_bad_values, good_value):
    df.loc[df[col].isin(list_of_bad_values), col] = good_value #df.col.value_counts().index[0]


def evaluate_scaler_imputers(scalers, models, X,y, scoring, imputers=None, cv=5):
    """
    
    """
    pipelines = []
    cv_scores = []
    if imputers:
        for scaler in scalers:
            for imputer in imputers:
                for model in models:
                    pipe = Pipeline([
                        ('scaler', scaler),
                        ('imputer', imputer),
                        ('model', model)
                    ])
                    #pipelines.append(pipe)
                    scaler_name=pipe.named_steps['scaler'].__class__.__name__
                    imputer_name=pipe.named_steps['imputer'].__class__.__name__
                    model_name=pipe.named_steps['model'].__class__.__name__
                    
                    score = cross_val_score(pipe, X, y, scoring=scoring, n_jobs=-1, cv=cv)
                    cv_scores.append({'scaler': scaler_name, 'imputer': imputer_name, 'model': model_name,
                                    'score_mean': score.mean(), 'score_std': score.std()})
        display(HTML(f'<h4>CV Results with imputation</h4>'))
    else: 
        for scaler in scalers:
            for model in models:
                pipe = Pipeline([
                    ('scaler', scaler),
                    #('imputer', imputer),
                    ('model', model)
                ])
                #pipelines.append(pipe)
                scaler_name=pipe.named_steps['scaler'].__class__.__name__
                #imputer_name=pipe.named_steps['imputer'].__class__.__name__
                model_name=pipe.named_steps['model'].__class__.__name__
                
                score = cross_val_score(pipe, X, y, scoring=scoring, n_jobs=-1, cv=cv)
                cv_scores.append({'scaler': scaler_name, 'imputer': 'N/A', 'model': model_name,
                                'score_mean': score.mean(), 'score_std': score.std()})
        display(HTML(f'<h4>Baseline CV Score with nulls dropped</h4>'))

    
    df_cv_scores = pd.DataFrame(cv_scores).sort_values(by=['score_mean','score_std'],
                                                  ascending=[False, True]).reset_index(drop=True)
    display(df_cv_scores)
    