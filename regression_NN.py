

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.datasets import mnist
from keras.optimizers import RMSprop
from keras import regularizers

from sklearn.model_selection import train_test_split

from lib_classifNN import *
from lib_visu_classif import *

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import os

import pandas as pd
## In this file, we create a neural network with one hidden layer and apply it to the mnist dataset in order to address the classification problem.


# In[2]:


path = 'data/FiveCitiePMData'
data = []
for f in os.listdir(path):
    print(f)
    data.append(pd.read_csv(path + '/'+ f))


# ## Cleaning the dataset 
# 
# We will try to make a regression on the PM_US Post values with some continuous and categorical variables in the dataset

# In[3]:


beijing = data[2]
print(list(beijing))
np.sum(pd.isna(beijing['PM_Nongzhanguan']))
# removing the 
missing_uspost = [not(e) for e in list(pd.isna(beijing['PM_US Post'], ))]
beijing_cl = beijing[missing_uspost]

features = ['year', 'month', 'day', 'hour', 'season', 'DEWP', 'HUMI', 'PRES', 'TEMP','Iws', 'precipitation', 'Iprec']


# In[4]:


# check for nas
for e in features:
    print(e, ':', np.sum(pd.isna(beijing_cl[e])), 'nas')


# In[6]:


# removing the rows for which there are NA
beijing_cl2 = beijing_cl[features + ['PM_US Post' ]].dropna(how = 'any')
X_df = beijing_cl2[features]
y_df = beijing_cl2['PM_US Post']


# In[7]:


# checking another time for nas
for e in features:
    print(e, ':', np.sum(pd.isna(X_df[e])), "nas")

print('PM_US Post :', np.sum(pd.isna(y_df)), 'nas')


plt.figure(figsize= (10,10))

for i,e in enumerate(list(X_df)):
    plt.subplot(3,4,i+1)
    plt.hist(X_df[e], )
    plt.title(e)
plt.show()

plt.hist(y_df)
plt.title("PM_US Post")
plt.show()



# converting to numpy arrays
X = X_df.values
y = y_df.values


nb_samples = 1e4
indices = np.random.shuffle(np.arange(nb_samples))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print('X train shape: ', x_train.shape)
print('y train shape: ', y_train.shape)
print('X test shape: ', X_test.shape)
print('y test shape: ', y_test.shape)

