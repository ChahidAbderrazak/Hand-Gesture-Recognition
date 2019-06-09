# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 16:43:55 2019

@author: chahida
Email: abderrazak.chahid@gmail.com
"""
#% Packages and functions

import os
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
import timeit

#from __future__ import division
import matlab.engine
import io
out = io.StringIO()
err = io.StringIO()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from random import randint
import time


from Python_lib.Shared_Functions import *
from Python_lib.Feature_Generation import *

#%%##################################################################################################
# results destination
save_excel=1

# PWM-based parameters
list_M=[i for i in range(6,7,2)]            # quantzation number of intervals
list_k=1.2#[i for i in range(0.8,1.3,0.1)]  # Quantizatin resolution
feature_type='MC_mPWM_'

# experiemnt  parameters
list_Subjets=[i for i in range(1,2)]
list_Gestures=[i for i in range(1,9)]        # The eight hand gestures classes
Trial_split=[[1,3,5,7,9],[2,4,6,8,10]]       #Trial_split=list(np.random.permutation(trials_labels).reshape(( 2, len(trials_labels)//2)))

# The classifier pipeline
names = ["Logistic Regression",
         "Nearest Neighbors", "Linear SVM","RBF SVM",
         "Decision Tree", "Random Forest",
         "Neural Net", "AdaBoost","Naive Bayes"]

classifiers = [LogisticRegression(),#(random_state=0, solver='lbfgs',multi_class='multinomial'),
    KNeighborsClassifier(3), SVC(kernel="linear", C=0.025),SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),AdaBoostClassifier(),GaussianNB()]


#%%  ---------------------   CLASSES + ENTRIES PREPPARATION   --------------------------
X, Gesture,Subject, Trial, X_train, y_train, X_test, y_test, mat_filename=Load_split_dataset(list_Subjets,list_Gestures,Trial_split)

Mdl0_raw=Classification_Train_Test(names, classifiers, X_train, y_train, X_test, y_test)

#%%  ---------------------    METHOD II: matlab functions   --------------------------
M=6#[i for i in range(6,7,2)]            # quantzation number of intervals
k=0.33#[i for i in range(0.8,1.3,0.1)]  # Quantizatin resolution
#  Quatization
Q_train,mu,sigma,Level_intervals=Quantization( k,M, X_train,"NA","NA", y_train)         # Quantize the training set
Q_test,mu,sigma,Level_intervals=Quantization( k,M, X_test,mu,sigma)                     # Quantize the testing set

#%%  ---------------------   QuPMWM-based FEATURE GENERATION   --------------------------
#% Build the nPWM matrices for different kMers
m=3                                         # kmers order
mPWM_structure=Build_QuPWM_matrices(m, Q_train, y_train,Level_intervals)

#%% Generate the Training/Testing features
mPWM_feature_train=Generate_QuPWM_features(mPWM_structure, Q_train)
mPWM_feature_test =Generate_QuPWM_features(mPWM_structure, Q_test)

Xf_train, QuPWM_names, QuPWM_sizes=QuPWM_feature_selection(mPWM_feature_train,selected_feature);

#%%  manually Select fPWM features

selected_feature_op,Xf_train_op,Xf_test_op,acc_max= Feature_selection_using_Scanning_kMers(mPWM_feature_train,y_train, mPWM_feature_test, y_test, names, classifiers)


from itertools import combinations

list_QuPWM_types=[i for i in range(6,12)]
acc_max=0
for k in range(3,6):
    for selected_feature in combinations(list_QuPWM_types , k) :
        print(list(selected_feature))
#        selected_feature=[1,5,8]# [0,1,2]#
        Xf_train=QuPWM_feature_selection(mPWM_feature_train,selected_feature);
        Xf_test =QuPWM_feature_selection(mPWM_feature_test, selected_feature);
        Mdl_score_op=Classification_Train_Test(names, classifiers, Xf_train, y_train, Xf_test, y_test )
        acc=Mdl2_score['Logistic Regression'][0]
        print('Acc_max=',acc_max,'---  Acc=',acc)

        if acc>acc_max :
            selected_feature_op=selected_feature
            Xf_train_op=Xf_train
            Xf_test_op=Xf_test
            acc_max=acc




#%%   Select fPWM features using TREE
indices,importances=Feature_selection_using_Tree(Xf_train,y_train)
Mdl_score=Classification_Train_Test(names, classifiers, X_train, y_train, X_test, y_test )
#%%   ----------------------------  Train/Test Split  ----------------------------
names,classifiers = ["Logistic Regression"],[LogisticRegression()]
for k in range(1,5000,100):
    Mdl2_score=Classification_Train_Test(names, classifiers, Xf_train[:,0:k], y_train, Xf_test[:,0:k], y_test )
    print('\n ', k, '\n')

#%% ---------------------------- Save results ----------------------------
folder_name='./Results'
Save_Mdl_performance(save_excel, Mdl_score, folder_name, feature_type, mat_filename)

#%% Test Script CELL ################################################################################
print('The END.')



#X = trials_labels
#kf = KFold(n_splits=5,shuffle=True)
#kf.get_n_splits(X)
#for train_index, test_index in kf.split(X):
#    print("TRAIN:", train_index, "TEST:", test_index)
#    X_train, X_test = X[train_index], X[test_index]
#    y_train, y_test = y[train_index], y[test_index]
