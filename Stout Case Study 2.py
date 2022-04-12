#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import random

# setting maximum visibility of rows and columns
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# In[2]:


case = pd.read_csv('casestudy.csv')


# In[3]:


case.head()


# In[4]:


case.drop('Unnamed: 0', axis=1, inplace=True)


# In[5]:


case.dtypes


# In[6]:


case.describe()


# In[7]:


# Q1
print('Total revenue for the current year:', 
      np.round(case.loc[case['year'] == 2017]['net_revenue'].sum(), 2))


# In[8]:


# sorting year in ascending order
case = case.sort_values(by= 'year')


# In[9]:


# creating a new column for new and existing customers
case= case.assign(Occurence=np.where(~case['customer_email'].duplicated(),'New','Existing'))


# In[10]:


case['Occurence'].value_counts()


# In[11]:


# Q2

print('New customer revenue for the year 2017:', 
      case[(case['year'] == 2017) & (case['Occurence'] == 'New')]['net_revenue'].sum())

print('New customer revenue for the year 2016:', 
      case[(case['year'] == 2016) & (case['Occurence'] == 'New')]['net_revenue'].sum())


# In[12]:


# Q3

print('Existing customer growth for 2017:', 
      case[(case['year'] == 2017) & (case['Occurence'] == 'Existing')]['net_revenue'].sum() - case[(case['year'] == 2016) & (case['Occurence'] == 'Existing')]['net_revenue'].sum())

print('Existing customer growth for 2016:', 
      case[(case['year'] == 2016) & (case['Occurence'] == 'Existing')]['net_revenue'].sum() - case[(case['year'] == 2015) & (case['Occurence'] == 'Existing')]['net_revenue'].sum())


# In[13]:


# Q4

revenue_lost15_16 = case[case['year'] == 2015]['net_revenue'].sum() - case[(case['year'] == 2016) & (case['Occurence'] == 'Existing')]['net_revenue'].sum()
revenue_lost16_17 = case[case['year'] == 2016]['net_revenue'].sum() - case[(case['year'] == 2017) & (case['Occurence'] == 'Existing')]['net_revenue'].sum()

total_revenue_lost = revenue_lost15_16 + revenue_lost16_17

print('Total revenue lost due to attrition:', total_revenue_lost)


# In[14]:


# Q5
print('Existing customer revenue of the current year:', case[(case['year'] == 2017) & 
                                                             (case['Occurence'] == 'Existing')]['net_revenue'].sum())


# In[15]:


# Q6
print('Existing customer revenue of the prior year:', case[(case['year'] == 2016) & 
                                                             (case['Occurence'] == 'Existing')]['net_revenue'].sum())


# In[16]:


# Q7
print('Total customers in the current year:', case[case['year']==2017].shape[0])


# In[17]:


# Q8
print('Total customers in the prior year:', case[case['year']==2016].shape[0])


# In[18]:


# Q9
print('New customers in the year 2016:', case[(case['year']== 2016) & (case['Occurence'] == 'New')].shape[0])
print('New customers in the year 2017:', case[(case['year']== 2017) & (case['Occurence'] == 'New')].shape[0])


# In[19]:


# Q10
print('Lost customers in the year 2016:', case[case['year'] == 2015].shape[0]-case[(case['year'] == 2016) & 
                                                                                   (case['Occurence'] == 'Existing')].shape[0])
print('Lost customers in the year 2017:', case[case['year'] == 2016].shape[0]-case[(case['year'] == 2017) & 
                                                                                   (case['Occurence'] == 'Existing')].shape[0])


# In[20]:


# Around 88% of the customers have made interactions just once
case.groupby('customer_email').size().value_counts()


# In[21]:


sns.countplot(case['year'], hue = case['Occurence'])


# From the above plot we understand the following:     
# From the year 2016 to 2017, the number of New customers drastically increased, where as the number of existing customers dropped. This tells us that the company lost many customers in the year 2017, but also gained new customers.   
# We cannot say anything about the year 2015, as the previous year's data has not been provided.

# In[22]:


df = case.groupby('customer_email').size().value_counts()
df


# In[23]:


plt.bar([1,2,3], [df[1], df[2], df[3]])
plt.xlabel('Number of transacations')
plt.ylabel('Number of customers')


# From the above plot we understand, the number of repitative transactions performed by the customers.   
# Around 88% of the customers made transactions only once, this might explain the high churning rate.

# In[24]:


sns.boxplot(case['year'], case['net_revenue'], hue = case['Occurence'])


# As it was seen earlier the number of new customers was comparitively very high than the existing customers.   
# If you see the box-plot, it can be observed that the net revenue of both type of customers (Existing, New) is similar.   
# This also tells us that the expenditure of the new customers is not as high as the existing customers, assuming as these customers are new to the company enough trust has not been built yet.
