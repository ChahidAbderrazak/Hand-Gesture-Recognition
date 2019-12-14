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
import h5py


from Python_lib.Shared_Functions import *
from Python_lib.Feature_Generation import *

#%%##################################################################################################
# input parameters
save_excel=0;
project_name='MEG_QuPWM'
mat_filename='Blcd_Patients_9Datasize_531885_2340_EA001EA002EA003EA004EA005EA006EA007EA008EA009_CHs26_L90_Step2'#'Patients8_L90_balanced'#'dataset_MEG90.mat' # QUPWM features

# PWM-based parameters
list_M=[i for i in range(6,7,2)]            # quantzation number of intervals
list_k=1.2#[i for i in range(0.8,1.3,0.1)]  # Quantizatin resolution
feature_type='mPWM_'



#The classifier pipeline
names = ["Logistic Regression","Nearest Neighbors", "Linear SVM",
        "RBF SVM", "Decision Tree", "Random Forest", "Neural Net",
        "Naive Bayes", "Quadratic Discriminant Analysis"]

classifiers = [LogisticRegression(), KNeighborsClassifier(3), SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=2),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    RandomForestClassifier(max_depth=7, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

#names,classifiers = ["Logistic Regression"],[LogisticRegression()]

##################################################################################################
#%% The classifier pipelinLoad mat files
#loaded_mat_file = scipy.io.loadmat('./mat/'+mat_filename)
loaded_mat_file = scipy.io.loadmat('R:/chahida/Projects-Dataset/KFMC/Extracted_data/Patients/EA001EA002EA003EA004EA005EA006EA007EA008EA009/Balanced/'+mat_filename)

X = loaded_mat_file['X']
y = loaded_mat_file['y'].ravel()
y_patient = loaded_mat_file['y_patient'].ravel()
Frame_Step = loaded_mat_file['Frame_Step'].ravel()

#%%
#X=X - np.min(X)
#X=X/np.max(X)
#X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=.2, random_state=42)

Mdl0_raw_data=Classification_Train_Test(names, classifiers, X_train, y_train, X_test, y_test )

#%%  ---------------------    METHOD II: matlab functions   --------------------------
M=6#[i for i in range(6,7,2)]            # quantzation number of intervals
k=0.33#[i for i in range(0.8,1.3,0.1)]  # Quantizatin resolution
#  Quatization
Q_train,mu,sigma,Level_intervals=Quantization( k,M, X_train,"NA","NA", y_train)         # Quantize the training set
Q_test,mu,sigma,Level_intervals=Quantization( k,M, X_test,mu,sigma)                     # Quantize the testing set

#%%  ---------------------   QuPMWM-based FEATURE GENERATION   --------------------------
#% Build the nPWM matrices for different kMers
m=1                                          # kmers order
mPWM_structure=Build_QuPWM_matrices(m, Q_train, y_train,Level_intervals)

#%% Generate the Training/Testing features
mPWM_feature_train=Generate_QuPWM_features(mPWM_structure, Q_train)
mPWM_feature_test =Generate_QuPWM_features(mPWM_structure, Q_test)


#%%  manually Select fPWM features

selected_feature=-1#[0,1,2]#[1,5,8]#-1#

Xf_train, QuPWM_names, QuPWM_sizes,sel_feat = QuPWM_feature_selection(mPWM_feature_train,selected_feature)
Xf_test,  QuPWM_names, QuPWM_sizes,sel_feat = QuPWM_feature_selection(mPWM_feature_test,selected_feature);

Mdl1_score_select=Classification_Train_Test(names, classifiers, Xf_train, y_train, Xf_test, y_test )

#%% optimally select the feature

selected_feature_op,Xf_train_op,Xf_test_op,acc_max= Feature_selection_using_Scanning_kMers(mPWM_feature_train,y_train, mPWM_feature_test, y_test, names, classifiers)


from itertools import combinations

list_QuPWM_types=[i for i in range(6,12)]
acc_max=0
for k in range(3,6):
    for selected_feature in selected_feature:#combinations(list_QuPWM_types , k) :
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




##%%   Select fPWM features using TREE
#indices,importances=Feature_selection_using_Tree(Xf_train,y_train)
#Mdl_score=Classification_Train_Test(names, classifiers, X_train, y_train, X_test, y_test )
##%%   ----------------------------  Train/Test Split  ----------------------------
#for k in range(1,5000,100):
#    Mdl2_score=Classification_Train_Test(names, classifiers, Xf_train[:,0:k], y_train, Xf_test[:,0:k], y_test )
#    print('\n ', k, '\n')
#
##%% ---------------------------- Save results ----------------------------
#folder_name='./Results'
#Save_Mdl_performance(save_excel, Mdl_score, folder_name, feature_type, mat_filename)
#
##%% Test Script CELL ################################################################################
#print('The END.')
#
#
#
##X = trials_labels
##kf = KFold(n_splits=5,shuffle=True)
##kf.get_n_splits(X)
##for train_index, test_index in kf.split(X):
##    print("TRAIN:", train_index, "TEST:", test_index)
##    X_train, X_test = X[train_index], X[test_index]
##    y_train, y_test = y[train_index], y[test_index]
