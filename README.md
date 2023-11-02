# Success of a Contact: Will a client subscribe? 
[Link to bank_marketing.ipynb Jupyter Notebook](https://github.com/marchofnines/bank_marketing/blob/main/bank_marketing.ipynb)

## PLEASE NOTE
- The code includes plotly plots that need to be rerun in order be visible 
- I included a PDF version of the Jupyter Notebook to make it easier to see the plotly plots 
- **Please scroll to the VERY END of the Jupyter Notebook or PDF past all the gridsearch warnings in order to see the full analysis**



## Problem 1: Understanding the Data
There were 17 different campaigns carried out between May 2008 and November 2010 corresponding to a total of 79354 contacts.   

### To better understand the problem, I summarized the paper below:  
##### - Business Goal:
-  Find the best model that can explain the success of a contact i.e. will the client subscribe to a deposit?  By identifying the main characterstics that affect success of a contact, the bank can increase its marketing efficiency and achieve a given number of successes for a lesser number of contacts thus saving time and resources.

##### Iterations: 
- First Iteration:  The research focused on whether the client will subscribe not regarding the deposit amount (which was initially an output).  Then in addition to campaign data, client personal information was collected resulting in a total of 59 attributes.  A first rough model was then built.  
- Second Iteration: The target classes were simplified to just two outcomes and graphical tools were used to visualize and analyze the data.  Features that had equal proportions of Success and Failures were eliminated resulting in 29 input variables. Testing seemed to support the removal of the excess features.
- Third Iteration: 
    - Nulls were removed and a third iteration of models were built and holdout validation was conducted.  
    - AUC-ROC curves and Cumulative LIFT curves were plotted to compare the different models.  
    - A sensitivity analysis to show the most important features and gain feedback on how best to conduct future campaigns.  For instance, call duration was found to be the most important along with the specific month of contact.  

#### Next steps 
- Test the existing model in the real word
- Collect more client based data and test contact-less campaign


## Problem 2: Read in the Data
- To begin with we will read in the data into a dataframe called raw
- We will assign df_edits for the cleaning stage 
- We will define df_viz which will be a copy of df_edits but with the target encoded to aid in visualizations
-  The final dataframe we will use for modeling will be df_clients

## Problem 3: Understanding the Features
### General Observations
- Since pdays has a value of 999 to indicate that a client was not previously contacted, treating it as a continuous variable would distort the distribution and is very likely to affect modeling performance.  We will convert the feature to categorical and update the 999s to 'Not contacted' 
- All other features of type 'object' will be converted to 'category' for more efficient memory usage and processing.  This would also allow us to define an order for visualizations. 
### Observations through Visualizations (and Pandas) 
- The target column shows **imbalance** with an overall subscription rate of 11.27%
#### Benchmark Feature: Duration
- This will not be used for modeling, however the following observations were made:
    - Duration column shows that for calls shorter than 70s, no subscriptions are made
    - The mode of the duration is around 90-100s with only few clients subscribing
    -  As the duration increases the subscription rate increases strongly
#### Heatmap correlation observations
- The economic and employment indicators have strong positive correlations with each other 
- Very low correlation among features related to the previous campaigns

#### Features that have a lot of variability in the subscription rate and are likely to have some importance: 
- ##### age
    - Highest subscription rates below ~20 and especially above ~60
    - There are very few clients above 60
    - Estimated importance: Moderate to high
- ##### default
    - Clients who previously defaulted had a 0% subscription rate
    - We note the presence of an unknown category
    - Estimated importance: High
- ##### contact
    - customers contacted on their cell phone were 3 times more likely to subscribe
    - Estimated importance: High
- ##### month
    - Estimated importance: High
- ##### campaign (number of calls during this campaign)
    - Most clients were contacted less than 8 times
    - For 1-8 campaigns, as number of Campaigns/Calls increases, subscription rate generally decreases 
    - Little to no subscriptions when client was contacted more than ~17 times
    - Estimated importance: High
- ##### pdays (days since last contacted from a previous campaign - repeat bus)
    - Vast majority of clients were clients who were never previously contacted and they had low subscription rates
    - Estimated importance: Moderate to High
- ##### previous (number of contacts performed before this campaign - prev campaign)
    - As number of previous contacts increases, the success rate generally increases
    - Estimated importance: High, Relationship is Linear 
- ##### poutcome outcome of the previous marketing campaign
    - Having been contacted in a previous campaign increases chance of success even in the event of a previous failure 
    - This feature likely has some interaction with previous and pdays
    - Estimated importance: Very high
- ##### emp.var.rate
    - As employment variation rate increases past ~-0.5, success rates fall drastically even though banks have tried harder to make more calls 
    - Estimated importance: Moderate to high
- ##### cons.price.idx and cons.conf.idx
    - These features show some variations 
    - Estimated importance: Moderate
- ##### nr.employed
    - As the number of employees increases we see a general decrease in success rates
    - Somewhat linear relationship with target
    - Estimated importance: Moderate

#### Features that have less variability in subscription rates and likely to play a lesser role:
- ##### job
- ##### marital
- ##### education
- ##### housing
- ##### loan
- ##### day_of_week

## Problem 4: Understanding the Task 
#### Business Objective:
Find the main characterstics that affect the success of a contact so that the bank can **increase its marketing efficiency**. i.e. Help the bank achieve a given number of successes for a lesser number of contacts thus saving it time and resources

The selected model will be evaluated against 2 criteria:
- **Performance**
    - We will use a carefully selected scoring metric and consider train/inference time
- **Explainability**
     - Ability to identify the most important features and explain their impact on the target.  Depending on the selected model,we will do this through one or more of the following:
     - Interpreting model coefficients (inferential statistics)
     - Using Partial Dependence and ICE Plots
     - Model representations
     - LIFT curves
     - CounterFactuals
     - Actionable items for a non-technical audience


## Problem 5 Engineering Features
#### Encoding choice
- For housing, loan and default, OneHotEncoding is the obvious choice. 
- For Education and Job we could consider Binary Encoding or James Stein Encoding, however, Job has the highest number of dimensions and that is 12.  At the same time, we are using only 7 features (or less) and have a large dataset (40k+ samples), so our cardinality is not very high.  Because of this and to improve our chances of getting higher scores, we will use one hot encoding for all categorical features 

## Problem 7: A Baseline Model 
- Using a DummyClassifier we achieved:
    - Train Accuracy of: 0.8873
    - Test Accuracy of: 0.8873

## Problem 8: A Simple Model
We created a Simple Model using:  
- OneHotEncoder of categorical features
- StandardScaler
- A simple LogisticRegression Model

## Problem 9: Score the Model
We evaluated the simple model across several metrics. Here are the results:
| Model   | Train Time | Inference Time | Train Accuracy | Test Accuracy | Train Precision | Test Precision | Train Recall | Test Recall | Train f1 | Test f1 | Train ROC AUC | Test ROC AUC |
|---------|------------|----------------|----------------|---------------|-----------------|----------------|--------------|-------------|----------|---------|---------------|--------------|
| Pipeline| 0.112887   | 0.012601       | 0.887346       | 0.887346      | 0.787383        | 0.787383       | 0.887346     | 0.887346    | 0.834381 | 0.834381| 0.650991      | 0.655391     |


## Problem 10: Model Comparisons
| Model               | Train Time | Inference Time | Train Accuracy | Test Accuracy | Train Precision | Test Precision | Train Recall | Test Recall | Train ROC AUC | Test ROC AUC |
|---------------------|------------|----------------|----------------|---------------|-----------------|----------------|--------------|-------------|---------------|--------------|
| Logistic Regression | 0.130102   | 0.014986       | 0.887346       | 0.887346      | 0.787383        | 0.787383       | 0.887346     | 0.887346    | 0.650991      | 0.655891     |
| KNN                 | 0.051160   | 0.267852       | 0.891198       | 0.878994      | 0.862973        | 0.826457       | 0.891198     | 0.878994    | 0.787690      | 0.568524     |
| Decision Trees      | 0.111717   | 0.010421       | 0.917775       | 0.868466      | 0.918043        | 0.819098       | 0.917775     | 0.868466    | 0.919315      | 0.579553     |
| SVM                 | 74.828012  | 7.243961       | 0.887670       | 0.887152      | 0.878236        | 0.825048       | 0.887670     | 0.887152    | 0.629428      | 0.552591     |

Note even though we see a high score for Decision Trees and KNN, these models are quite overfit and therefore we pick the Logistic Regression Model as the model to beat.  It also does well in terms of Training Time.  

### Comparison of 4 models 
In general, we prefer Logistic Regression and Decision Trees because they are the most interpretable models and also come with probabilities which can provide levels of confidence.  That said, below is my analysis for comparing these models: 
 - The highest accuracy score belongs to Decision Trees followed by SVM and KNN which are both overfit and finally Logistic Regression
 - However, as will be discussed below in Problem 11, we would like to use ROC AUC as our performance metric
 - When we look at ROC AUC the only one that is not overfit by a lot is Logistic Regression
 - On top of this SVM ihas a very long training time 
 - Lastly, Logistic Regression will be easy to explain so we will continue to use that as the reference model to bea.  But we are not done.  We will continue to cross-validate using all models.  

## Problem 11: Improving the Model

### Updating our Performance Metric
We will use ROC-AUC as our primary scoring metric for the following reasons:
 - It handles imbalance well.   According to [this article by Machine Learning Mastery](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-imbalanced-classification/), *"ROC analysis does not have any bias toward models that perform well on the minority class at the expense of the majority class—a property that is quite attractive when dealing with imbalanced data."*.  The article also states it is the most commonly used metric for imbalanced problems. 
 - We would be able to compare our results to the results of the study from the university of Lisbon
 - ROC AUC can leverage predict_probas so we can not only predict the success rate but also select samples where the level of confidence is above a threshold
 - However, secondarily we will also like to keep an eye on ****Precision** because we are trying to improve the efficiency of the campaign and therefore we want to **minimize the False Positives**.

### Trying SMOTE
Before doing feature engineering we tried to address the class imbalance using oversampling on the minority class [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.pipeline.Pipeline.html)

#### Conclusion Regarding using SMOTE
- The  roc auc score to beat was for the simple Logistic Regression model: 0.6554 
- Using SMOTE with the same model we achieved a roc auc score of:  0.6572 (and similar precision)
- While we did see a slightly better score with SMOTE, the improvement was negligible and since SMOTE complicates the model, we abandoned the idea

### Feature Engineering
#### 1. Examine LogisticRegression Coefficients
- The **housing** and **loan** coefficients are low.  This confirms what we saw in the visualizations which is that the success rate stayed relatively stable for all values of these features. We will drop these two features and reevaluate the simple model. 
- Based on the coefficients, the most important features are default, job and age.  
#### 2. Partial Dependence Plot
This was not the most helpful since we could only see the Partial Dependence Plot for the age feature but it did confirm that as age increases, so does the model's likelihood of predicting a success. In other words it confirmed that age is an important feature to keep. 

#### 3. SelectFromModel + LogisticRegression with Regularization 
The features that were removed using SelectFromModel are shown below.  We do see that ohe__housing_yes and ohe__loan_no were eliminated but this is inconclusive because ohe__housing_no and ohe__loan_yes remain.
{'ohe__education_basic.6y',
 'ohe__housing_yes',
 'ohe__job_self-employed',
 'ohe__job_technician',
 'ohe__loan_no',
 'ohe__marital_married'}

#### 4.  Visuzalize the regularization of the features 
Unfortunately the graph was too busy to be helpful! 

#### 5. Permutation Importance
roc_auc
    default       0.057 +/- 0.006
    job           0.050 +/- 0.005
    marital       0.011 +/- 0.002
    education      0.011 +/- 0.003
    age           0.010 +/- 0.003

#### Conclusion of feature engineering:  Eliminate housing and loan features
- We looked at:
  - Logistic Regression Coefficents
  - SelectFromModel + Logistic Regression + L1 Regularization
  - Permutation Importance
  - Partial Dependence Plot of the Age column

All the results were helpful but the clearest results came from the Permutation Importance Results.  We will remove the **housing and loan** because they have minimal impact on the predictiveness of the model.


### MODELING, CROSS-VALIDATION AND HOLDOUT VALIDATION:  
In this next phase, we used different modeling sets.  For each set we called the function **run_pipelines** which does the following: 
- Takes multiple pipelines and for each pipeline: 
    - Performs Cross-Validation on each pipeline with a wide range of parameters 
    - Shows the best Cross-Validation in descending order of rank
    - Performs Holdout Validation on each model in the RandomizedSearch
    - Shows the best Holdout-Validation  models by overfit status and holdout test score 
        - If holdout_tolerance > 0, then it will select the best models within the tolerance
        - We will generally work with a holdout_tolearance of 2%
    - Shows a plot of all the models and where the best non-overfit model lies 
- Produces a summary DataFrame of the best holdout models from all the pipelines 

#### Description of Modeling Sets

##### Keep Duplicates:  For First 3 sets we keep duplicates on purpose.  We think this makes sense for this dataset because the duplicates reflect the fact that many clients have the same profile and tells us something about the tendancies of those clients
- Set 1: Balanced Pipelines (no class_weights), 5 features, no imputing, no duplicates dropped
- Set 2: Weighted Pipelines, 5 features, no imputing, no duplicates dropped
- Set 3: Weighted Pipelines, 5 features, Simple Imputer, for all features except default, no duplicates dropped

##### Drop Duplicates: Just to make sure we will also run searches with dropped duplicates 
- Set 4: Weighted Pipelines, 5 features, no imputing, duplicates dropped
- Set 5: Weighted Pipelines, 5 features, Simple Imputer for all features except default, duplicates dropped

#### Reasoning for choosing not to impute 'default' column:
The reason we try imputing all features except the default column is that we noticed that our imputer set all the values of default to 'no'.  It is doubtful that this is a good imputation.  

### SUMMARY OF MODELING RESULTS  
#### Simple Model
Once again our Simple Model performed as follows:
- Test ROC AUC of 0.6554
- CV Test Precision: 0.7874
- Fit Time: 0.1129

 #### SET 1 

| model        | train roc_auc score | test roc_auc score | mean fit time | mean test accuracy | mean test precision |
|--------------|--------------------|-------------------|---------------|--------------------|---------------------|
| lgr          | 0.664728           | 0.661370          | 19.993211     | 0.514152           | 0.845869            |
| lgr_saga_l1l2| 0.668225           | 0.660663          | 23.327606     | 0.610104           | 0.840228            |
| lgr_saga_elastic | 0.650284       | 0.653682          | 21.010838     | 0.456347           | 0.845139            |
| knn          | 0.672715           | 0.650773          | 1.302407      | 0.887584           | 0.787806            |
| dtree        | 0.500000           | 0.500000          | 1.360314      | 0.112416           | 0.012637            |
| svc          | 0.516418           | 0.516664          | 17.550135     | 0.445371           | 0.796574            | 

- With just 5 features, we are able to achieve similar scores as the simple model but our secondary metric (Precision is much improved!)
- The best non-overfit model came from lgr saga with elasticnet:
    - Test ROC AUC: 0.6653 
    - Precision: 0.8451
    - Mean Fit Time: 21.01s

#### SET 2
| model           | train roc_auc score | test roc_auc score | mean fit time | mean_test_accuracy | mean_test_precision |
|-----------------|--------------------|--------------------|---------------|--------------------|--------------------|
| lgr             | 0.652331           | 0.656672           | 1.488931      | 0.563874           | 0.838321           |
| lgr_saga_l1l2   | 0.650890           | 0.655896           | 1.511839      | 0.564206           | 0.838226           |
| lgr_saga_elastic| 0.662118           | 0.663104           | 23.392615     | 0.887584           | 0.787806           |
| knn             | 0.672715           | 0.657073           | 1.912423      | 0.887584           | 0.787806           |
| dtree           | 0.500000           | 0.500000           | 1.170062      | 0.112416           | 0.012637           |
| svc             | 0.643583           | 0.651736           | 24.007791     | 0.887584           | 0.787806           |

-lgr saga with elasticnet penalty performed the best according to our primary metric ROC AUC but since we are also keeping an eye on precision, we select regular lgr:
    - Test ROC AUC: 0.6567 (Non-Overfit) 
    - CV Test Precision: 0.8383
    - Mean Fit Time: 1.48s 

#### SET 3
| model          | train_roc_auc_score | test_roc_auc_score | mean_fit_time | mean_test_accuracy | mean_test_precision |
|----------------|---------------------|--------------------|---------------|--------------------|--------------------|
| knn            | 0.673319            | 0.657396           | 1.735170      | 0.887584           | 0.787806           |
| lgr            | 0.651818            | 0.657200           | 1.585244      | 0.560947           | 0.838054           |
| lgr_saga_l1l2  | 0.650543            | 0.656226           | 1.709505      | 0.563042           | 0.837821           |
| lgr_saga_elastic| 0.664408           | 0.664616           | 19.141769     | 0.887584           | 0.821644           |
| dtree          | 0.500000            | 0.500000           | 0.915330      | 0.112416           | 0.012637           |
| svc            | 0.642986            | 0.652167           | 19.950604     | 0.887584           | 0.787806           |

-lgr performed had the best non-overfit model based on our primary metric ROC AUC but also had a high precision and a low mean fit time:
    - Test ROC AUC: 0.6572 (Non-Overfit) 
    - CV Test Precision: 0.8381
    - Mean Fit Time: 1.58s 

#### SET 4 and SET 5
I will omit these results because:
- We could not get CV scores for precision and accuracy
- When we tried to score the best models using holdout validation, these models turned out to have much lower Precision Scores and/or were overfit
- As mentioned before this was just a test but it makes more sense to KEEP the duplicates in order to assign larger weights to common types of clients  


### Comparing the cross validation of all 30 grid searches (5 sets of 6):
- Once again preference is given to decision trees and logistic regression because they are the most interpretable.  That said: 
  - Decision Tree model ROC AUC had to drop to 0.5 to find non-overfit model.  There is one exception but that score is well below our best Logistic Regression Results
  - SVM did produce non-overfit models but those scores were still below Logistic Regression and the fit time was consistently among the highest 
  - KNN performed marginally better than Logistic Regression in some cases however, this is not enough to offset the fact that it had longer training times and is less interpretable than Logistic Regression
  - From the Logistic Regression Models, the plain logistic regression model from set 3 performed the best when we look at score and training time together


### Final Model Selection:  Compare Best CV Model with Simple Model using 5 features only 
From CV we would select the Logistic Regression Model from Set 3 because it perform much faster than the best model and has very similar scores 

However it turns out that when we reconstruct our initial simple transformer to handle just 5 features, it performs about as good as our best CV model but it is also faster and does not include the complexity of PolynomialFeatures or Imputation.  We will use this as it will be the easiest to interpret! That was a lot of number crunching for nothing!! :) 

| Model   | Train Time | Inference Time | Train Accuracy | Test Accuracy | Train Precision | Test Precision | Train Recall | Test Recall | Train f1 | Test f1 | Train ROC AUC | Test ROC AUC |
|---------|------------|----------------|----------------|---------------|-----------------|----------------|--------------|-------------|----------|---------|---------------|--------------|
| Pipeline| 0.302159   | 0.009486       | 0.887346       | 0.887346      | 0.787833        | 0.787833       | 0.887346     | 0.887346    | 0.843481 | 0.843481| 0.650879      | 0.654356     |


### Examining our Model
#### Analysis of Confusion Model 
- Even though we have 0 False Negatives which is what we wanted to minize, we have 0 True Positives indicating that the model is not predicting any of the success!  
- Since our business goal is to increase the efficiency of the marketing campaign by achieving higher success rates for the same number of contacts, we normally would try and raise the confidence level of predicting successes to ~80% for instance but this does nothing since our TP and FP are already equal to 0.
- This tells us that the model is not predicting on the positive class let alone doing so with high confidence.  This is probably due to the imbalance in the dataset and the fact that we did not use campaign or economic related features.  Below we examine the distribution of the probabilities to see if we can increase the confidence level of the no class

#### Observations on probability distributions
- Again our business goal is to increase the efficiency of the campaign
- Based on the distribution above, we cannot raise the threshold of positive predictions above 50% 
- If our goal was just to find more customers who would subscribe then it would make sense to lower our threshold of positive predictions to ~15% but that is not what we want to do! 
- The only thing we can do is raise the threshold of negative predictions above 80%.  This would help the campaign ensure they do not waste their time on contacts who are not likely to subscribe! 

#### Interpretation of Coefficients
Below are the most important coefficents along with their interpretation.  Note that for each of these we assume that all other variables in the model are held constant:
- Job:  
    - Students are 3.09 times more likely to subscribe than clients who are admins 
    - Retirees are 2.1 times more likely to subscribe than clients who are admins
    - Unemployed people are 1.33 times more likely to subscribe than clients who are admins
    - Clients with unknown jobs are 1.15 times more likely to subscribe than clients who are admins
- Education: 
    - Clients who are illiterate are 2.57 times more likely to subscribe than clients with a basic.4y education
    - Clients with unkonwn education are 1.25 times more likely to subscribe than clients with a basic.4y 
    - Clients with a university degree are 1.18 times more likely to subscribe than clients with a basic.4y
- Marital: 
    - Clients whose marital status is unknown are 1.91 times more likely to subscribe than divorced clients
    - Clients who are single are 1.57 times more likely to subscribe than divorced clients
    - Clients who are married are 1.15 times more likely to subscribe than divorced clients
- Age: 
    - For every unit increase in age, the client is 1.013 times more likely to subscribe which means that for every 1 year increase,  the odds the client will subscribe increase by 1.3%.  

#### CounterFactuals Analysis:  
- If you have a group of clients who are of age 41 that have never defaulted, then you'd want to pick the ones that are students and illiterate or retired and illiterate to improve your chances of finding a client who will subscribe 
- This matches what we found when looking at the coefficients! 

### Findings for non-technical audience
- In your campaigns try to target students then retirees and then unemployed clients.  Avoid admins.  
- Try to target illiterate clients,then clients whose education status is unknown then those with a university degree.  Avoid clients with basic4.y
- Try to target clients whose marital status is unknown then those who are single then those who are  married.  Avoid clients who are divorced.  
- Also target older clients.  The older the better. 
- Lastly we will be sending you a spreadsheet containing profiles to avoid.  We know with high confidence that clients with these profiles are unlikely to subscribe.  We recommend avoiding them 

### Next Steps and Recommendations 
- More needs to be done to favor the positive class and address the imbalance in the dataset.  Ideas include:
    - Carrying out more campaigns targetting the clients that have profiles as described in our recommendations 
    - We did try to oversample with SMOTE but you may want to try other sampling techniques to address the imbalance of this dataset
    - Use additional features related to the campaign and/or economic features 
    - Try to use other types of imputers
    - Continue to run grid searches with updated datasets
    - More exploration of counterfactuals