import sys
sys.path.append('/Users/basilhaddad/jupyter/module17/bank_marketing_repo/')
from importlib import reload
from helpers.my_imports import * 
from IPython.core.display import HTML

def reorder_cols_in(df, in_str, after=True):
    """
    Reorders the columns of a DataFrame based on substring matches, moving them either to the beginning or end.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame whose columns need to be reordered.
    - in_str (str, list): The substring(s) to match against column names.
    - after (bool): Whether to move the matched columns after the others. If False, moves them before.
    
    Returns:
    - pd.DataFrame: A DataFrame with reordered columns.
    """
    
    #check if in_str belong to column name (handle both lists and individual arguments)
    if isinstance(in_str, list):
        grouped_cols = []
        for s in in_str:
            grouped_cols += [col for col in df.columns if s in col]
    else:
        grouped_cols = [col for col in df.columns if in_str in col]
    #group remaining columns
    remaining_cols = [col for col in df.columns if col not in grouped_cols]
    #order column groups
    
    #if after, place grouped cols after remaining cols and vice versa
    if after:
        new_col_order = remaining_cols + grouped_cols
    else: 
        new_col_order = grouped_cols + remaining_cols
    
    return df[new_col_order]



def cv_and_holdout(estimator,X, y, test_size=0.27, stratify=None, random_state=42, search_type='random', param_dict=None,
                  scoring=None, refit=None, holdout_tolerance=0, verbose=0, cv=5, n_iter=10, summary=True):
    pd.set_option('display.max_columns', None)
    """
    Perform cross-validation and holdout validation on a given estimator.
    
    Parameters:
        estimator: scikit-learn estimator object
        X: Feature matrix
        y: Target vector
        test_size: test_size for train_test_split
        stratify: stratify for train_test_split
        random_state: random_state for train_test_split
        search_type: use grid for GridSearchCV and random for RandomizedSearchCV
        param_dict: Dict of parameters to be used in grid/randomized search
        scoring: Dict of Scoring Metrics to be used in grid/randomized search
        refit: refit scoring metric (required) for grid/randomized search as well as holdout validation
        holdout_tolerance: is overfitting within given tolerance?
        verbose: verbose parameter for grid/randomized search
        cv: cv parameter for grid/randomized search
        n_iter: number of iterations for grid/randomized search
        summary: Show summary of results which include: 
                 - Models ranked by descending CV rank
                 - Models ranked by overfit status and descending holdout test scores
                 - Plot of all the models + vertical line at best non-overfit model
    
    Returns:
        ho_results: DataFrame containing holdout results
        best_holdout_estimator: Best estimator based on holdout validation
    """
    #Validate refit metric was entered
    if refit in scoring:
        refit_scorer = scoring[refit]
    else:
        raise ValueError(f"The refit metric {refit} was not found in the scoring_metrics dictionary.")

    # Step 1: Split Data into Train and Test Sets and Run GridSearchCV or RandomizedSearchCV
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=None, random_state=random_state)

    if search_type == 'grid':
        search = GridSearchCV(estimator, param_dict, scoring=scoring, refit=refit, cv=cv, n_jobs=-1)
    else:
        search = RandomizedSearchCV(estimator, param_dict, n_iter=n_iter, scoring=scoring, refit=refit, random_state=random_state, verbose=verbose, cv=cv, n_jobs=-1)
    
    search.fit(X_train, y_train)
    
    # Step 2: Build Custom Results DataFrame based on cv_results_
    cv_results = pd.DataFrame(search.cv_results_)
    #Unclutter results
    cv_results = cv_results.drop(columns=cv_results.columns[cv_results.columns.str.startswith('split')])
    cv_results = cv_results.drop([col for col in cv_results.columns if 'std_' in col], axis=1)
    cv_results = cv_results.drop(columns='mean_score_time')
    cv_results = cv_results.drop(columns='params')
    
    # Step 3: Holdout Validation using refit score metric
    holdout_train_scores = []
    holdout_test_scores = []
    overfit_flags = []
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
    
    # Step 4: Augment cv_results with holdout testing results 
    #Define column names
    cv_rank_refit_col=f"cv_rank_{refit}"
    cv_test_score_refit_col=f"cv_{refit}" #mean_test!
    ho_rank_refit_col=f"holdout_rank_{refit}"
    ho_train_score_refit_col=f"train_{refit}"
    ho_test_score_refit_col=f"test_{refit}"
    cv_results.rename(columns={
        f'rank_test_{refit}': cv_rank_refit_col,
        f'mean_test_{refit}': cv_test_score_refit_col
    }, inplace=True)
    #Update dataframe with new columns
    cv_results[ho_train_score_refit_col] = holdout_train_scores
    cv_results[ho_test_score_refit_col] = holdout_test_scores
    cv_results['is_overfit'] = overfit_flags
    cv_results['is_overfit']= cv_results['is_overfit'].map({0: 'No', 1: 'Yes'}) 
    #Take snapshot to be used in step 5
    ho_results = cv_results.copy()
    #Sort results by descending CV Rank and reorder columns for visibility
    cv_results = cv_results.sort_values(by=[cv_rank_refit_col, ho_test_score_refit_col] , ascending=[True, False])  
    #Show most important columns first
    cv_results= reorder_cols_in(cv_results, [cv_rank_refit_col, cv_test_score_refit_col, 'is_overfit', ho_train_score_refit_col, ho_test_score_refit_col, 
                                               'mean', 'param'] , after=False) 
    
    
    #Step 5: Define dataframe to display holdout test results ordered by overfit status and descending holdout test scores
    ho_results.sort_values(by=['is_overfit', ho_test_score_refit_col, cv_rank_refit_col], ascending=[True, False, True], inplace=True)
    ho_results.reset_index(drop=True, inplace=True)
    # Create the new holdout rank column based on the new index
    ho_results[ho_rank_refit_col] = ho_results.index + 1  
      
    #Show most important columns first
    ho_results= reorder_cols_in(ho_results, [cv_rank_refit_col, cv_test_score_refit_col, 'is_overfit', ho_train_score_refit_col, ho_test_score_refit_col,  
                                              'mean', 'param'] , after=False) #'mean_fit_time', 'mean_score_time'

    #Step 6: Display Best CV Model Details and Best Holdout Model Details
    if summary:
        display(HTML(f'<h3>Results for {estimator.steps[-1][0]}: </h3>'))
        display(HTML(f'<h5>Models ranked by descending {cv_rank_refit_col}</h5>'))
        display(cv_results.iloc[:4,:].style.hide_index())
        display(HTML(f'<h5>Models ranked by overfit status and descending holdout {ho_test_score_refit_col}</h5>'))
        display(HTML(ho_results.iloc[:4,:].to_html(index=False)))

    # Step 7: Plot Holdout Validation Model Scores and show best non-overfit (or least overfit if threshold > 0) if available
    if summary:
        sns.set_style('darkgrid')
        common_fontsize=23
        linewidth=1.8
        markers='o',
        s=80
        plt.clf()
        plt.figure(figsize=(23 , 6))
        
        sns.scatterplot(x=cv_rank_refit_col, y=ho_train_score_refit_col, label=f'Holdout Train {refit} Score', markers=markers,  s=s, data=ho_results)
        sns.scatterplot(x=cv_rank_refit_col, y=ho_test_score_refit_col, label=f'Holdout Test {refit} Score', markers=markers,  s=s, data=ho_results)
        
        #Show best non-overfit (or least overfit if threshold > 0) if available
        best_model_rank_score_list = None

        filtered_ho_results = ho_results.query(f"{ho_rank_refit_col}==1 and is_overfit=='No'")
        if not filtered_ho_results.empty:
            best_model_rank_score_list = filtered_ho_results[[cv_rank_refit_col, ho_test_score_refit_col]].iloc[0].to_list()
            plt.axvline(x=best_model_rank_score_list[0], color='r', linestyle='--', label=f"Best Non-Overfit Model {refit} Test Score: {best_model_rank_score_list[1]:.3f}")
        else:
            print("No non-overfit models were found. Consider re-running the function with a houldout_threshold > 0")
        
        plt.xticks(fontsize=common_fontsize)
        plt.yticks(fontsize=common_fontsize)
        plt.title(f"{refit} Holdout Train and Test Scores", weight='bold', fontsize=common_fontsize+2)
        plt.xlabel(cv_rank_refit_col, fontsize=common_fontsize) 
        plt.ylabel("Score", fontsize=common_fontsize)
        plt.grid(True, which='both', linestyle='--', linewidth=0.6)
        plt.legend(fontsize=common_fontsize-3)
        
        plt.tight_layout()
        plt.show()
    return ho_results, best_holdout_estimator


def evaluate_models(models, X_train, y_train, X_test,y_test, transformer=None, scaler=None, selector=None):
    """
    Evaluate one or more machine learning models on given data.
    
    Parameters:
    - models: dict or single  model. If dict, keys are model names and values are model instances.
    - X_train, y_train, X_test, y_test: Training and test datasets.
    - transformer: Data transformer (optional).
    - scaler: Data scaler (optional).
    - selector: Feature selector (optional).
    
    Returns:
    - results_df: DataFrame containing performance metrics.
    """
    #Define dict of arrays to store results
    results = {
    'Model': [],
    'Train Time': [],
    'Inference Time': [],
    'Train Accuracy': [],
    'Test Accuracy': [],
    'Train Precision': [],
    'Test Precision': [],
    'Train Recall': [],
    'Test Recall': [],
    'Train f1': [],
    'Test f1': [],
    'Train ROC AUC': [],
    'Test ROC AUC': []
    }
  
    #If not a dict, make it a dict for uniform processing
    if not isinstance(models, dict):
        models = {
            models.__class__.__name__ : models
        }
        
    import time
    for model_name, model in models.items():        
        #build steps for pipeline
        steps = []
        if transformer is not None:
            steps.append(('transformer', transformer))
        if scaler is not None:
            steps.append(('scaler', scaler))
        if selector is not None:
            steps.append(('selector', selector))
        
        #Take start timestamp for training
        start_time = time.time()

        #if it's a pipeline, or if it's a standalone model, just fit it
        if isinstance(model, Pipeline) or (transformer is None and scaler is None and selector is None):
            fit_model = model.fit(X_train, y_train)
        #else add the model to the steps and then fit it 
        else:
            steps.append((model_name, model))
            fit_model = Pipeline(steps).fit(X_train, y_train)

        #Take end timestamp for training
        train_time = time.time() - start_time

        #Take start timestamp for test inference
        start_time = time.time()
        y_test_pred = fit_model.predict(X_test)
        #Take end timestamp for test inference 
        inference_time = time.time() - start_time

        #Compute various score metrics
        train_accuracy = fit_model.score(X_train, y_train)
        test_accuracy = fit_model.score(X_test, y_test)

        y_train_pred = fit_model.predict(X_train)
        #y_test_preds = model.predict(X_test) (already computed)
        y_train_prob = fit_model.predict_proba(X_train)[:, 1]
        y_test_prob = fit_model.predict_proba(X_test)[:, 1]

        train_precision = precision_score(y_train, y_train_pred, average='weighted')
        train_recall = recall_score(y_train, y_train_pred, average='weighted')
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')    
        train_roc_auc = roc_auc_score(y_train, y_train_prob)

        test_precision = precision_score(y_test, y_test_pred, average='weighted')
        test_recall = recall_score(y_test, y_test_pred, average='weighted')
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        test_roc_auc = roc_auc_score(y_test, y_test_prob)

        #Append results to arrays 
        results['Model'].append(model_name)
        results['Train Time'].append(train_time)
        results['Inference Time'].append(inference_time)
        results['Train Accuracy'].append(train_accuracy)
        results['Test Accuracy'].append(test_accuracy)
        results['Train Precision'].append(train_precision)
        results['Test Precision'].append(test_precision)
        results['Train Recall'].append(train_recall)
        results['Test Recall'].append(test_recall)
        results['Train f1'].append(train_f1)
        results['Test f1'].append(test_f1)
        results['Train ROC AUC'].append(train_roc_auc)
        results['Test ROC AUC'].append(test_roc_auc)
    
    #Create results dataframe using the arrays of metrics
    results_df = pd.DataFrame(results)
    return results_df


def select_all_col_names_except(df, exclude_list):
    """
    Select all column names from a DataFrame except those specified in an exclusion list.
    
    Parameters:
    - df: pandas DataFrame
    - exclude_list: list of column names to exclude
    
    Returns:
    - List of column names to keep
    """
    # List of all columns
    all_columns = df.columns.tolist()
    # Columns to exclude
    exclude_columns = exclude_list
    # Columns to keep
    return list(set(all_columns) - set(exclude_columns))


def run_pipelines(pipe_param_pairs, X, y, test_size = 0.25, stratify=None, random_state=42, search_type='random', 
                   scoring=None, refit=None, holdout_tolerance=0, verbose=0, cv=5, n_iter=10, summary=True):
    """
    Run multiple pipelines with different hyperparameter settings into cv_and_holdout function and collect the results.
    
    Parameters:
        pipe_param_pairs: Pairs of pipelines and parameter dics
        X: Feature matrix
        y: Target vector
        test_size: test_size for train_test_split
        stratify: stratify for train_test_split
        random_state: random_state for train_test_split
        search_type: use grid for GridSearchCV and random for RandomizedSearchCV
        param_dict: Dict of parameters to be used in grid/randomized search
        scoring: Dict of Scoring Metrics to be used in grid/randomized search
        refit: refit scoring metric (required) for grid/randomized search as well as holdout validation
        holdout_tolerance: is overfitting within given tolerance?
        verbose: verbose parameter for grid/randomized search
        cv: cv parameter for grid/randomized search
        n_iter: number of iterations for grid/randomized search
        summary: Show summary of results which include: 
                 - Models ranked by descending CV rank
                 - Models ranked by overfit status and descending holdout test scores
                 - Plot of all the models + vertical line at best non-overfit model
                 
        Returns:
        - Dataframe containing the performance metrics for the best model from each pipe/param pair
        - An array of the best models 
    """
    output, models= [], []
    for pipe, params in pipe_param_pairs:
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=None, random_state=random_state)
        result, model =cv_and_holdout(estimator=pipe,
                                X=X,
                                y=y,
                                stratify=stratify,
                                param_dict=params,
                                scoring=scoring,
                                refit= refit,  
                                random_state=random_state,
                                search_type= search_type,  
                                n_iter=n_iter,
                                holdout_tolerance=holdout_tolerance,
                                cv=cv,
                                verbose=verbose, 
                                summary=summary)
       #Create summary dict of best model from each pipe/param pair
        output_row = {'model': pipe.steps[-1][0], 
                      f'train {refit} score': result.loc[0, f'train_{refit}'],
                      f'test {refit} score': result.loc[0, f'test_{refit}'], 
                      'mean fit time': result.loc[0, 'mean_fit_time']}
        # Append columns that start with 'mean_test' to output_row
        mean_test_cols = [col for col in result.columns if col.startswith('mean_test')]
        for col in mean_test_cols:
            output_row[col] = result.loc[0, col]
        # Append to output and models arrays 
        output.append(output_row)
        models.append(model)
    #Display summary DataFrame containing best models from each pipe/param pair    
    output_df = pd.DataFrame(output)    
    display(HTML(f'<h3>Best Models From Each Grid/Random Search: </h3>'))
    display(output_df.style.hide_index())

    return output_df, models

        
    