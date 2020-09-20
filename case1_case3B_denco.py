# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 21:41:10 2020

@author: Adithya Madhavan
"""

#Topic ---- Case Study - Denco - Manufacturing Firm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%case details
#%%Objective
#Expand Business by encouraging loyal customers to Improve repeated sales
#Maximise revenue from high value parts
#%%Information Required
#Who are the most loyal Customers - Improve repeated sales, Target customers with low sales Volumes
#Which customers contribute the most to their revenue - How do I retain these customers & target incentives
#What part numbers bring in to significant portion of revenue - Maximise revenue from high value parts
#What parts have the highest profit margin - What parts are driving profits & what parts need to build further
#%%%
#see all columns
pd.set_option('display.max_columns',15)
#others - max_rows, width, precision, height, date_dayfirst, date_yearfirst
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:.2f}'.format
#read data
url='https://raw.githubusercontent.com/DUanalytics/datasets/master/csv/denco.csv'
df = pd.read_csv(url)
df
#Sort revenue in descending order
df.describe()

df_loyal = df.groupby('custname').count().sort_values(by='region',ascending = 0)
df_loyal.iloc[0: ,0:1].head()
df_revenue = df.groupby('custname').count().sort_values(by='revenue',ascending = 0)
df_revenue.iloc[0: ,0:1].head()
#df.sort_values(by=['revenue'], inplace = True,ascending = False)
#df.sort_values(by=['revenue'], inplace = True,ascending = False) 
df_loyal
df_revenue
df
df_loyalty.head(10)

#Frequency of each customer
df['custname'].value_counts()


