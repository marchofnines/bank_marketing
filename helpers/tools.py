import sys
sys.path.append('/Users/basilhaddad/jupyter/module17/bank_marketing_repo/')
from importlib import reload
from helpers.reload import myreload
import helpers.preprocessing as pp
import helpers.plot as plots
import helpers.tools as tools


import pandas as pd
import numpy as np
from IPython.display import display
from IPython.core.display import display, HTML

import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt
import plotly.express as px
import plotly.subplots as sp
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='notebook'

from scipy.stats import entropy, randint, uniform
from scipy.linalg import svd

from sklearn.experimental import enable_halving_search_cv, enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from category_encoders import JamesSteinEncoder, BinaryEncoder
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder, OrdinalEncoder, LabelBinarizer
from sklearn.compose import make_column_transformer, make_column_selector, ColumnTransformer, TransformedTargetRegressor
from sklearn.feature_selection import SequentialFeatureSelector, SelectFromModel, RFE
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN

from sklearn.linear_model import LinearRegression, Lasso, HuberRegressor, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.multiclass import OneVsOneClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV

from sklearn.inspection import permutation_importance
from sklearn.metrics import get_scorer, mean_squared_error, mean_absolute_error, accuracy_score, f1_score, recall_score, precision_score
from sklearn.metrics import precision_recall_curve, auc, roc_curve, RocCurveDisplay, log_loss
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.inspection import PartialDependenceDisplay, partial_dependence

def reorder_cols_like(df, like_str, after=True):
    """
    Reorders the columns of a DataFrame based on substring matches, moving them either to the beginning or end.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame whose columns need to be reordered.
    - like_str (str, list): The substring(s) to match against column names.
    - after (bool): Whether to move the matched columns after the others. If False, moves them before.
    
    Returns:
    - pd.DataFrame: A DataFrame with reordered columns.
    """
    
    if isinstance(like_str, list):
        grouped_cols = []
        for s in like_str:
            grouped_cols += [col for col in df.columns if s in col]
    else:
        grouped_cols = [col for col in df.columns if like_str in col]
    
    remaining_cols = [col for col in df.columns if col not in grouped_cols]
    
    if after:
        new_col_order = remaining_cols + grouped_cols
    else: 
        new_col_order = grouped_cols + remaining_cols
    
    return df[new_col_order]


#cv_based_holdout(estimator, X, y, )


def cv_and_holdout(estimator,X, y, test_size=0.25, stratify=None, random_state=42, search_type='random', param_dict=None,
                  scoring=None, refit=None, holdout_tolerance=0, verbose=0, cv=5, n_iter=10, summary=True):
    pd.set_option('display.max_columns', None)
    """
    - For general use case use cv_based_holdout
    - used in pigmee assignment.  Requires classifier with 0 and 1 as classes
    - requires scoring = list and refit = string 
    """
    if refit in scoring:
        refit_scorer = scoring[refit]
    else:
        raise ValueError(f"The refit metric {refit} was not found in the scoring_metrics dictionary.")
    original_pipe=str(estimator)

    # Step 1: Run GridSearchCV or RandomizedSearchCV
    if search_type == 'grid':
        search = GridSearchCV(estimator, param_dict, scoring=scoring, refit=refit, cv=cv)
    else:
        search = RandomizedSearchCV(estimator, param_dict, n_iter=n_iter, scoring=scoring, refit=refit, random_state=random_state, verbose=verbose, cv=cv) #n_jobs=-1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=None, random_state=random_state)
    search.fit(X_train, y_train)
    
    # Step 2: Extract models and results
    cv_results = pd.DataFrame(search.cv_results_)
    #type(list(search.best_estimator_.named_steps.values())[-1]).__name__
    #cv_results = cv_results.drop(columns=cv_results.columns[cv_results.columns.str.startswith('split')])
    #cv_results = cv_results.drop([col for col in cv_results.columns if 'std_' in col], axis=1)
    cv_results = cv_results.drop(columns='params')
    #param_cols = {col: col.split('__')[-1] for col in cv_results.columns if '__' in col}
    #cv_results.rename(columns=param_cols, inplace=True)
    
    holdout_train_scores = []
    holdout_test_scores = []
    overfit_flags = []
    
    # Step 3: Holdout Validation
    best_holdout_estimator = None
    best_holdout_score = 0
    for candidate_params in search.cv_results_['params']:
        estimator.set_params(**candidate_params)
        estimator.fit(X_train, y_train)
        
        holdout_train_score = refit_scorer(estimator, X_train, y_train)
        holdout_test_score = refit_scorer(estimator, X_test, y_test)
        
        if holdout_train_score > (1+holdout_tolerance)*holdout_test_score: 
            if holdout_test_score > best_holdout_score:
                best_holdout_score = holdout_test_score
                best_holdout_estimator = estimator
        
        holdout_train_scores.append(holdout_train_score)
        holdout_test_scores.append(holdout_test_score)
        
        overfit_flags.append(1 if holdout_train_score > (1+holdout_tolerance)*holdout_test_score else 0)
    
    # Step 4: Augment cv_results 
    rank_test_refit_col=f"rank_test_{refit}"
    mean_test_refit_col=f"mean_test_{refit}"
    
    cv_results['holdout_train_score'] = holdout_train_scores
    cv_results['holdout_test_score'] = holdout_test_scores
    cv_results['is_overfit'] = overfit_flags
    cv_results_og = cv_results.copy()
    
    #Step 5: Sort CV Results by decending rank and create a modified copy for displaying best Holdout Results by descending test score
    # Sort by 'is_overfit' ascending and 'holdout_test_score' descending and rank_test_refit_col ascending
    cv_results.sort_values(by=['is_overfit', 'holdout_test_score', rank_test_refit_col], ascending=[True, False, True], inplace=True)
    cv_results.reset_index(drop=True, inplace=True)
    # Create the new rank column based on the new index
    cv_results['holdout_test_rank'] = cv_results.index + 1  

    cv_results_og = cv_results_og.sort_values(by=[rank_test_refit_col, 'holdout_test_score'] , ascending=[True, False])    

    cv_results= reorder_cols_like(cv_results, ['is_overfit', 'holdout_test_rank', rank_test_refit_col, 'holdout_train_score', 'holdout_test_score', mean_test_refit_col, 
                                               'mean_fit_time', 'mean_score_time', 'params'] , after=False)

    #Rearrange columns and map overfit column.  TODO: fix the rename
    cv_results['is_overfit']= cv_results['is_overfit'].map({0: 'No', 1: 'Yes'}) 
    cv_results.rename(columns={
    #'hold_out_train_score': f'train_score_{refit}',
    #'hold_out_train_score': f'test_score_{refit}', 
    mean_test_refit_col: f'cv_score_{refit}'
    }, inplace=True)
    
    #Step 6: Display Best CV Model Details and Best Holdout Model Details
    if summary:
        display(HTML(f'<h4>Model:{original_pipe}</h4>'))
        display(HTML(f'<h5>Top CV Results By Descending {refit} Rank</h5>'))
        display(cv_results_og.iloc[:5,:])
        display(HTML(f'<h5>Best Holdout Validation Models By Overfit Status, Hold Out Test Rank and CV Rank</h5>'))
        display(HTML(cv_results.iloc[:7,:].to_html(index=False)))

    # Step 7: Plotting
    if summary:
        sns.set_style('darkgrid')
        common_fontsize=20
        linewidth=1.8
        markers='o',
        s=80
        plt.clf()
        plt.figure(figsize=(23 , 6))
        
        sns.scatterplot(x=rank_test_refit_col, y='holdout_train_score', label=f'Holdout Train {refit} Score', markers=markers,  s=s, data=cv_results)
        sns.scatterplot(x=rank_test_refit_col, y='holdout_test_score', label=f'Holdout Test {refit} Score', markers=markers,  s=s, data=cv_results)
        
        # Initialize best_model_rank_score_list to None
        best_model_rank_score_list = None

        filtered_cv_results = cv_results.query("holdout_test_rank==1 and is_overfit=='No'")
        if not filtered_cv_results.empty:
            best_model_rank_score_list = filtered_cv_results[[rank_test_refit_col, 'holdout_test_score']].iloc[0].to_list()
            plt.axvline(x=best_model_rank_score_list[0], color='r', linestyle='--', label=f"Best Non-Overfit Model {refit} Test Score: {best_model_rank_score_list[1]:.3f}")
        else:
            print("No non-overfit models were found. Consider re-running the function with a houldout_threshold > 0")
        
        plt.xticks(fontsize=common_fontsize)
        plt.yticks(fontsize=common_fontsize)
        plt.title(f"{refit} Holdout Train and Test Scores", weight='bold', fontsize=common_fontsize+2)
        plt.xlabel(f'{refit} CV Test Rank',fontsize=common_fontsize)
        plt.ylabel("Score", fontsize=common_fontsize)
        plt.grid(True, which='both', linestyle='--', linewidth=0.6)
        plt.legend(fontsize=common_fontsize-5)
        
        plt.tight_layout()
        plt.show()
    return cv_results, best_holdout_estimator





# Modify the function to handle the column name issue
def tune_multiple_imputers(df, imputers, metrics):
    results = []
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df_no_na = val_df.dropna()
    
    target_col = 'target'
    
    for imputer in imputers:
        imputer_name = type(imputer).__name__
        has_transform = hasattr(imputer, "transform")
        
        if has_transform:
            imputer.fit(train_df)
            imputed_val = imputer.transform(val_df_no_na)
            imputed_val = pd.DataFrame(imputed_val, columns=val_df_no_na.columns, index=val_df_no_na.index)
        else:
            train_no_na = train_df.dropna()
            X_train = train_no_na.drop(columns=[target_col])
            y_train = train_no_na[target_col]
            imputer.fit(X_train, y_train)
            imputed_val = imputer.predict(val_df_no_na.drop(columns=[target_col]))
            imputed_val = pd.DataFrame(imputed_val, columns=[target_col], index=val_df_no_na.index)
        
        result_row = {'Imputer': imputer_name}
        
        for metric in metrics:
            if metric == 'RMSE':
                score = np.sqrt(mean_squared_error(val_df_no_na[target_col], imputed_val[target_col]))
            elif metric == 'MSE':
                score = mean_squared_error(val_df_no_na[target_col], imputed_val[target_col])
            elif metric == 'MAE':
                score = mean_absolute_error(val_df_no_na[target_col], imputed_val[target_col])
            else:
                raise ValueError("Invalid metric type. Choose from 'RMSE', 'MSE', 'MAE'")
            
            result_row[metric] = score
        
        results.append(result_row)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by=metrics, ascending=[True]*len(metrics))
    return results_df

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def did_some_models_converge(search, model_name, params_dict):
    best_estimator = search.best_estimator_
    # Check its convergence
    actual_iterations = best_estimator.named_steps['logistic'].n_iter_
    max_iterations = params_dict['logistic__max_iter']
    print(f"Actual iterations for best estimator: {actual_iterations}")
    print(f"Max allowed iterations: {max_iterations}")
    
    if all(iter_count <= max(max_iterations) for iter_count in actual_iterations):
        print("Best model converged for all classes.")
    else:
        print("Best model did not converge for some classes.")


def evaluate_model_with_classification_v3(imputer, params_grid, metrics, n_splits, X, y):
    best_score = -float('inf')
    best_model = None
    best_params = None
    results = []

    # Step 1: K-Fold Cross-Validation
    kf = KFold(n_splits=n_splits)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        for params in params_grid:
            model = clone(imputer)
            model.set_params(**params)
            model.fit(X_train)  # Train the imputer
            
            # Transform the datasets using the trained imputer
            X_train_imputed = model.transform(X_train)
            X_test_imputed = model.transform(X_test)

            # Step 2: Holdout Validation using a Classifier (RandomForestClassifier)
            clf = RandomForestClassifier()
            clf.fit(X_train_imputed, y_train)  # Train the classifier
            y_train_pred = clf.predict(X_train_imputed)
            y_test_pred = clf.predict(X_test_imputed)

            # Calculate train and test errors for each metric
            for metric in metrics:
                train_error = error_metrics[metric](y_train, y_train_pred)
                test_error = error_metrics[metric](y_test, y_test_pred)
                is_overfit = "No" if train_error >= test_error else "Yes"
                
                results.append({
                    'Params': params,
                    'Metric': metric,
                    'Train Error': train_error,
                    'Test Error': test_error,
                    'Overfit': is_overfit
                })

                # Update best model based on test error
                if test_error > best_score:
                    best_score = test_error
                    best_model = clone(model)
                    best_params = params

    # Create DataFrame for the results
    results_df = pd.DataFrame(results)

    return best_model, best_params, results_df


def select_all_col_names_except(df, exclude_list):
    # List of all columns
    all_columns = df.columns.tolist()
    # Columns to exclude
    exclude_columns = exclude_list
    # Columns to keep
    return list(set(all_columns) - set(exclude_columns))



def run_pipelines(pipe_param_pairs, X, y, test_size = 0.25, stratify=None, random_state=42, search_type='random', 
                   scoring=None, refit=None, holdout_tolerance=0, verbose=0, cv=5, n_iter=10, summary=False):
    output, models= [], []
    for pipe, params in pipe_param_pairs:
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=None, random_state=random_state)
        result, model =tools.cv_and_holdout(estimator=pipe,
                                X=X,
                                y=y,
                                param_dict=params,
                                scoring=scoring,
                                refit= refit,  
                                random_state=random_state,
                                search_type= search_type,  
                                n_iter=n_iter,
                                summary=summary)
        #model_name = type(pipe.steps[-1][1]).__name__
        model_name = pipe.steps[-1][0]
        train_score = result.loc[0, 'holdout_train_score']
        test_score = result.loc[0, 'holdout_test_score']
        average_fit_time = result.loc[0, 'mean_fit_time']
        output_row = {'model': model_name, 
                      'train score': train_score,
                      'test score': test_score,
                      'average fit time': average_fit_time}
        output.append(output_row)
        models.append(model)
        output_df = pd.DataFrame(output)
        output_df['model'] = output_df['model'].map({'logistic': 'LogisticRegression', 
                                     'knn': 'KNN',
                                      'svc': 'SVC', 
                                      'dtree': 'DecisionTreeClassifier'})
    return output_df, models

def pipe_help(df, target_col, regressor, X_train= None, imbalanced=False, range_extend=0.2):
    n = df.shape[0]
    m = df.shape[1]
    
    print(f"Target value counts: {df['target_col'].value_counts(normalize=True)}")
    
    if isinstance(regressor, KNeighborsClassifier):
        if imbalanced: 
            params = {
                'knn__n_neighbors': randint(3, np.sqrt(n)+(n*range_extend)), #Rule of Thumb: Sqrt of N
                'knn__weights':  ['uniform', 'distance'], #use distance for imbalanced classes
                'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'knn__leaf_size': randint(5,100),  #default = 30
                'knn__p': [1,2],
                'knn__metric_params': [{'V': np.cov(X_train)}],
                'knn__metric': ['mahalanobis']   
                }   
        else:      
            params = {
            'knn__n_neighbors': randint(3, np.sqrt(n)+(n*range_extend)), #Rule of Thumb: Sqrt of N
            'knn__weights':  ['uniform', 'distance'], #use distance for imbalanced classes
            'knn__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'knn__leaf_size': randint(5,100),  #default = 30
            'knn__p': [1,2],
            'knn__metric': ['minkowski', 'euclidean', 'manhattan']  
            }    
            
        
        
    