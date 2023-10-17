# Success of a Contact: Will a client subscribe? 
[Link to bank_marketing.ipynb Jupyter Notebook](https://github.com/marchofnines/used_car_prices/blob/main/used_car_prices.ipynb)

## Problem 1: Understanding the Data
For my benefit I summarized the paper.  There were 17 different campaigns carried out between May 2008 and November 2010 corresponding to a total of 79354 contacts.

### Business Goal:
Find a model that can explain the success of a contact i.e. will the client subscribe to a deposit? 
By identifying the main characterstics that affect success of a contact, the bank can increase its marketing efficiency and achieve a given number of successes for a lesser number of contacts thus saving time and resources.

### First Iteration
The research focused on whether the client will subscribe not regarding the deposit amount (which was initially an output).  Then in addition to campaign data, client personal information was collected resulting in a total of 59 attributes.  A first rough model was then built.  

### Second Iteration 
The target classes were simplified to just two outcomes and graphical tools were used to visualize and analyze the data.  Features that had equal proportions of Success and Failures were eliminated resulting in 29 input variables. Testing seemed to support the removal of the excess features.

### Third Iteration
- Nulls were removed and a third iteration of models were built and holdout validation was conducted.  
- AUC-ROC curves and Cumulative LIFT curves were plotted to compare the different models.  
- A sensitivity analysis to show the most important features and gain feedback on how best to conduct future campaigns.  For instance, call duration was found to be the most important along with the specific month of contact.  

### Next steps 
- Test the existing model in the real word
- Collect more client based data and test contact-less campaign


## Problem 2: Read in the Data
- To begin with we will read in the data into a dataframe called raw
- We will assign df_edits for the cleaning stage 
- We will define df_viz which will be a copy of df_edits but with the target encoded to aid in visualizations
-  Ultimately we will call the datafram df_campaign

## Problem 3: Understanding the Features
### General Observations
- Since pdays has a value of 999 to indicate that a client was not previously contacted, treating it as a continuous variable would distort the distribution and is very likely to affect modeling performance.  We will convert the feature to categorical and update the 999s to 'Not contacted' 
- All other features of type 'object' will be converted to 'category' for more efficient memory usage and processing.  This would also allow us to define an order for visualizations. 
### Observations through Visualizations 
- The target column shows **imbalance** with an overall subscription rate of 11.27%
