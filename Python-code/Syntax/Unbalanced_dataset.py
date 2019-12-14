#%% -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 13:43:56 2019

@author: chahida
Email: abderrazak.chahid@gmail.com
"""
#% Packages and functions
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#%% ##################################################################################################
# input parameters

# Remove 'id' and 'target' columns
labels = df_train.columns[2:]

X = df_train[labels]
y = df_train['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
#%%   ----------------------  PART I  -------------------------
#% Split the test into sentences



#%% Test Script CELL ################################################################################
print('The END.')

