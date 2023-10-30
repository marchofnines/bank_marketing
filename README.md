# Success of a Contact: Will a client subscribe? 
[Link to bank_marketing.ipynb Jupyter Notebook](https://github.com/marchofnines/used_car_prices/blob/main/used_car_prices.ipynb)

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

#### 

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
| Model     | Train Time | Inference Time | Train Accuracy | Test Accuracy | Train Precision | Test Precision | Train Recall | Test Recall | Train ROC AUC | Test ROC AUC |
|-----------|------------|----------------|----------------|---------------|-----------------|----------------|--------------|-------------|---------------|--------------|
| Pipeline  | 0.200806   | 0.02099        | 0.887346       | 0.887346      | 0.787363        | 0.787363       | 0.887346     | 0.887346    | 0.843681      | 0.655391     |


## Problem 10: Model Comparisons
| Model               | Train Time | Inference Time | Train Accuracy | Test Accuracy | Train Precision | Test Precision | Train Recall | Test Recall | Train ROC AUC | Test ROC AUC |
|---------------------|------------|----------------|----------------|---------------|-----------------|----------------|--------------|-------------|---------------|--------------|
| Logistic Regression | 0.130102   | 0.014986       | 0.887346       | 0.887346      | 0.787383        | 0.787383       | 0.887346     | 0.887346    | 0.650991      | 0.655891     |
| KNN                 | 0.051160   | 0.267852       | 0.891198       | 0.878994      | 0.862973        | 0.826457       | 0.891198     | 0.878994    | 0.787690      | 0.568524     |
| Decision Trees      | 0.111717   | 0.010421       | 0.917775       | 0.868466      | 0.918043        | 0.819098       | 0.917775     | 0.868466    | 0.919315      | 0.579553     |
| SVM                 | 74.828012  | 7.243961       | 0.887670       | 0.887152      | 0.878236        | 0.825048       | 0.887670     | 0.887152    | 0.629428      | 0.552591     |

Note even though we see a high score for Decision Trees and KNN, these models are quite overfit and therefore we pick the Logistic Regression Model as the model to beat.  It also does well in terms of Training Time.  

### Problem 11: Improving the Model

#### Updating our Performance Metric
We will use ROC-AUC as our primary scoring metric for the following reasons:
 - It handles imbalance well.   According to [this article by Machine Learning Mastery](https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-imbalanced-classification/), *"ROC analysis does not have any bias toward models that perform well on the minority class at the expense of the majority class—a property that is quite attractive when dealing with imbalanced data."*.  The article also states it is the most commonly used metric for imbalanced problems. 
 - We would be able to compare our results to the results of the study from the university of Lisbon
 - ROC 
 - Secondarily we will also like to keep an eye on Precision because we are trying to improve the efficiency of the campaign and therefore we want to **minimize the False Positives**.

