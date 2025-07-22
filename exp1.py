#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("train_and_test2.csv")
print("first 5 rows",df.head())
print("Data info:",df.info())


# In[10]:


df['Age']=df['Age'].fillna(method='ffill').fillna(method='bfill')


# In[11]:


df['Age']=df['Age'].fillna('unknown')
print('cabin with unknown')
print(df['Age'].head())


# In[12]:


df=df.drop_duplicates()


# In[13]:


from sklearn.preprocessing import LabelEncoder, StandardScaler
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])


# In[14]:


scaler = StandardScaler()
df['Fare'] = scaler.fit_transform(df[['Fare']])


# In[15]:


sns.pairplot(df[['Pclass', 'Sex', 'Age', 'sibsp']])
plt.suptitle("Pair Plot of Selected Features", y=1.02)
plt.show()


# In[16]:


plt.figure(figsize=(8, 6))
corr_matrix = df[['Pclass', 'Age', 'sibsp', 'Parch', 'Fare']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# In[ ]:




