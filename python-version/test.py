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

from Python_lib.Shared_Functions import *

#%% ##################################################################################################
# input parameters


#%%   ----------------------  PART I  -------------------------
#% Split the test into sentences
target_count0=np.sum(y_train==0)
target_count1=np.sum(y_train==1)

print('Class 0:', target_count0)
print('Class 1:', target_count1)
print('Proportion:', round(target_count0 / target_count1, 2), ': 1')

#%% Test Script CELL ################################################################################
print('The END.')

