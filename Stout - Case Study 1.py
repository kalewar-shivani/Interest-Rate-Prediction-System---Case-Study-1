#!/usr/bin/env python
# coding: utf-8

# # Interest Rate Prediction:

# In[1]:


#importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#reading the data set
loans = pd.read_csv("loans_full_schema.csv")


# In[3]:


#exploring the data set dimensions
loans.shape


# # Data Cleaning:

# In[4]:


#checking the % of missing value for further cleaning
nan_feature = [features for features in loans.columns if loans[features].isnull().sum()>1]

for feature in nan_feature:
  print(feature, np.round(loans[feature].isnull().mean(), 2), '% missing values')


# In[5]:


#dropping the columns with more than 30% missing values
nan = []
for x in loans.columns:
     if loans[x].isnull().mean() > 0.3:
            nan.append(x)
            
loans.drop(nan, axis = 1, inplace = True)            


# In[6]:


#Again, inspecting the data set for missing values
nan_feature = [features for features in loans.columns if loans[features].isnull().sum()>1]

for feature in nan_feature:
  print(feature, np.round(loans[feature].isnull().mean(), 2), '% missing values')


# In[7]:


#replacing na values with respective column median
loans["emp_length"].fillna(value = loans["emp_length"].median(), inplace = True)
loans["debt_to_income"].fillna(value = loans["debt_to_income"].median(), inplace = True)
loans["months_since_last_credit_inquiry"].fillna(value = loans["months_since_last_credit_inquiry"].median(), inplace = True)
loans["num_accounts_120d_past_due"].fillna(value = loans["num_accounts_120d_past_due"].median(), inplace = True)


# In[8]:


#dropping the remaining na values (if any)
loans.dropna(axis = 1, inplace = True)


# In[9]:


#final inspection of the data set for any missing values
nan_feature = [features for features in loans.columns if loans[features].isnull().sum()>1]

for feature in nan_feature:
  print(feature, np.round(loans[feature].isnull().mean(), 2), '% missing values')


# In[10]:


#exploring the data type of each attribute within the data set
loans.dtypes


# In[11]:


#converting the data type of time attributes to 'datetime'
loans[['earliest_credit_line', 'issue_month']] = loans[['earliest_credit_line', 'issue_month']].apply(pd.to_datetime)


# In[12]:


loans_copy = loans


# In[13]:


loans_copy['interest_rate'].describe()


# In[14]:


#Discretization: interest_rate (only for exploration purposes) 
loans_copy['Interest_Rate_Cat'] = pd.cut(loans_copy['interest_rate'],6,labels=['5-10%','11-15%','16-20%','21-25%','26-30%','more than 30%'])


# In[15]:


#exploring the discretized variable
loans_copy['Interest_Rate_Cat'].value_counts()


# # Data Exploration:

# In[16]:


#heatmap of correlations of variables with each other
plt.subplots(figsize=(20, 20))
corr = loans_copy.corr()
#mask = np.triu(np.ones_like(corr, dtype=bool))
#cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, annot=True)


# In[17]:


#exploring the correlation of attributes with the target variable ('interest_rate')
loans_copy.corrwith(loans_copy['interest_rate']).sort_values(ascending = False)
loans_copy.corrwith(loans_copy['interest_rate']).sort_values(ascending = False)


# From the above results, we can observe the correlation of each pf the predictors with the target variable;        
# 'paid_interest' is highly positively correlated variable with the target variable (interest_rate), whereas      
# 'total_debit_limit' is highly negatively correlated variable with the target variable.

# In[18]:


#scatterplot: highest positively correlated vs highest negatively correlated attributes
import plotly.express as px
px.scatter(loans_copy, x='paid_interest', y='total_debit_limit', color = 'Interest_Rate_Cat')


# From the above scatte, the correlation with the target variable is evident.
# For lower interest rates, the total_debit_limit is high as compared to those for higher interest rates.
# Similarly, for lower interest rates the paid interest is also low and as the rate increase the paid_interest value also increases.
# 
# Ofcourse, there are exceptions to this analysis, but that can be justified by the moderate correlation value.

# In[19]:


#importing necessary libraries for interactive plots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


# In[20]:


#interactive box-plot - 1 (highest positive correlation)
loans_copy.pivot(columns='Interest_Rate_Cat', values='paid_interest').iplot(kind='box', yTitle='Paid Interest', title='Interest Rate vs Paid Interest')


# In[21]:


#interactive box-plot - 2 (highest negative correlation)
loans_copy.pivot(columns = 'Interest_Rate_Cat', values = 'total_debit_limit').iplot(kind='box', xTitle='Interest Rate', yTitle='Total Debit Limit', title='Interest Rate vs Total Debit Limit')


# In[22]:


#interactive scatter plot - 1 (highly positively correlated attributes to target variable)
import plotly.figure_factory as ff
figure1 = ff.create_scatterplotmatrix(loans_copy[['term', 'debt_to_income', 'Interest_Rate_Cat']], diag='histogram', index='Interest_Rate_Cat')
figure1


# In[23]:


#interactive scatter plot - 2 (highly negatively correlated attributes to target variable)
figure2 = ff.create_scatterplotmatrix(loans_copy[['num_mort_accounts', 'total_credit_limit','Interest_Rate_Cat']], diag='histogram', index='Interest_Rate_Cat')
figure2


# # Final Clean Data Set:

# In[24]:


#storing all the categorical attributes in a seperate data set
categorical_var = loans.select_dtypes(include='object')

#encoding the categorical attributes for further use
categorical = pd.get_dummies(data = categorical_var)


# In[25]:


#storing all the numeric attributes in a seperate data set
numeric_var = loans.select_dtypes(include=('int64', 'float64'))     

#storing all the date time attributes in a seperate data set
datetime_var = loans.select_dtypes(include='datetime64[ns]')


# In[26]:


#concatinating all the necessary data sets together for further model development
loans_final = pd.concat([numeric_var, categorical, datetime_var], axis = 1)


# In[27]:


#exploring the final data set
loans_final.head()


# In[28]:


loans_final.dtypes.value_counts()


# # Data Transformation:

# In[29]:


#scaling the numeric attributes of the data set, using MinMaxScaler function
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

loans_final.iloc[:, :-2] = scaler.fit_transform(loans_final.iloc[:, :-2])


# # Data Split:

# In[30]:


#splitting the data set into predictors (features) and outcome variable (target)
loans_features = loans_final.loc[:, loans_final.columns != 'interest_rate']
loans_target = loans_final['interest_rate'] 


# In[31]:


#splitting the features and target set into train and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(loans_features, loans_target, test_size = 0.2, random_state = 0)


# # Decision Tree Regressor:

# In[32]:


#fitting Decision Tree Regressor on the train set
from sklearn.tree import DecisionTreeRegressor
DTR = DecisionTreeRegressor()

DTR_model = DTR.fit(x_train.iloc[:, :-2], y_train)


# In[33]:


#predicting the train and test set target variable
DTR_pred = DTR.predict(x_train.iloc[:, :-2])
DTR_test_pred = DTR.predict(x_test.iloc[:, :-2])


# In[34]:


#checking the 'r2 score' of the model on the train and test set prediction results
from sklearn.metrics import r2_score
print('train:', r2_score(y_train, DTR_pred))
print('test:', r2_score(y_test, DTR_test_pred))


# Clearly, the model is overfitting on the train set, hence we tune the hyperparameters and as per the results develp a new model

# In[35]:


#tuning the DTR model to handle the overfitting, using GridSearchCV
from sklearn.model_selection import GridSearchCV

param_dict = {
    "splitter": ['best', 'random'],
    "max_depth": range(1,10),
    "min_samples_split": range(2,10),
    "min_samples_leaf": range(2,5)
        }

dtr = DecisionTreeRegressor()

#using the GridSearchCV function to iterate through the predefined hyperparameters of the model function
#k-fold cross-validation = 5 fold cross-validation
grid = GridSearchCV(dtr, param_grid= param_dict, cv = 5, verbose= 1, n_jobs= -1, scoring= 'neg_mean_squared_error', error_score = 'raise')

#fitting the estimator model on the train set
grid.fit(x_train.iloc[:, :-2], y_train)


# In[36]:


#finding the best paramteres values for the DTR model
grid.best_params_


# In[37]:


#building a new DTR model by tuning the parameters as per the above results
DTR_grid = DecisionTreeRegressor(max_depth = 9, min_samples_leaf = 2, min_samples_split = 4, splitter = 'random').fit(x_train.iloc[:, :-2], y_train)

#predicting the train set labels using the pruned DT Classifer model
DTR_pred = DTR_grid.predict(x_train.iloc[:, :-2])
DTR_test_pred = DTR_grid.predict(x_test.iloc[:, :-2])


# In[38]:


#inspecting how well the new model has performed on the train and test set
print('train:', r2_score(y_train, DTR_pred))
print('test:', r2_score(y_test, DTR_test_pred))


# In[39]:


#displaying the actual and predicted 'interest_rate' values
DTR_results = pd.DataFrame({'Actual (interest_rate)': y_test, 'Predicted (interest_rate)': DTR_test_pred})
print(DTR_results)


# In[40]:


#visualizing the model results
import plotly.express as px
px.scatter(DTR_results, x = 'Actual (interest_rate)', y = 'Predicted (interest_rate)')


# Considering the 0.99 r2_score of the parameter tuned DTR model, the above interactive scatter plot explains the linear regression between the actual test set and the predicted interest rate values.

# # Random Forest Regressor:

# In[41]:


#fitting Random forest Regressor on the data set
from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor(n_estimators = 10, random_state = 0)

RFR_model = RFR.fit(x_train.iloc[:, :-2], y_train)


# In[42]:


#predicting the train and test set target variable
RFR_pred = RFR.predict(x_train.iloc[:, :-2])
RFR_test_pred = RFR.predict(x_test.iloc[:, :-2])


# In[43]:


#model performance evaluation
print('train:', r2_score(y_train, RFR_pred))
print('test:', r2_score(y_test, RFR_test_pred))


# In[44]:


#exploring parameters in use before parameter tuning
print('Parameters currently in use:\n')
print(RFR.get_params())


# In[45]:


#tuning the parameters of the RFR model, using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

#creating the random grid using above values/variables
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print(random_grid)


# In[46]:


#using the RandomizedSearchCV function to iterate through the predefined hyperparameters of the model function
#k-fold cross validation = 5 fold cross-validation
RFR_cv = RandomizedSearchCV(estimator = RFR, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

RFR_cv.fit(x_train.iloc[:, :-2], y_train)


# In[47]:


#finding best parameter values for RFR model
RFR_cv.best_params_


# In[48]:


#developing a new RFR model by using parameter values from the above results
RFR_CV = RandomForestRegressor(n_estimators = 500, min_samples_split = 5, min_samples_leaf = 4, max_features = 'auto', max_depth = 100, bootstrap = True).fit(x_train.iloc[:, :-2], y_train)

#predicting the train and test set target variables
RFR_pred = RFR_CV.predict(x_train.iloc[:, :-2])
RFR_test_pred = RFR_CV.predict(x_test.iloc[:, :-2])


# In[49]:


#performance evaluation of the hyperparameter tuned RFR model
print('train:', r2_score(y_train, RFR_pred))
print('test:', r2_score(y_test, RFR_test_pred))


# In[50]:


#displaying the actual and predicted 'interest_rate' values
RFR_results = pd.DataFrame({'Actual (interest_rate)': y_test, 'Predicted (interest_rate)': RFR_test_pred})
print(RFR_results)


# In[51]:


#visualizing the model results
px.scatter(RFR_results, x = 'Actual (interest_rate)', y = 'Predicted (interest_rate)')


# Considering the 0.99 r2_score of the parameter tuned RFR model, the above interactive scatter plot explains the linear regression between the actual test set and the predicted interest rate values.
