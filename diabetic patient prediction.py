#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_percentage_error


# In[3]:


df = pd.read_csv(r'C:\Users\hp\Desktop\github\diabetes.csv')
df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.columns


# In[7]:


y = df['DiabetesPedigreeFunction']
X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'Age', 'Outcome']]


# In[8]:


X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=2529)


# In[9]:


knr = KNeighborsRegressor()


# In[10]:


knr.fit(X_train,y_train)


# In[11]:


y_pred = knr.predict(X_test)


# In[12]:


y_pred


# In[13]:


mean_absolute_percentage_error(y_test,y_pred)


# In[ ]:




