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
m=4                                         # kmers order
list_M=[i for i in range(6,7,2)]            # quantzation number of intervals
list_k=1.2#[i for i in range(0.8,1.3,0.1)]  # Quantizatin resolution
feature_type='MC_mPWM_'

# experiemnt  parameters
list_Subjets=[i for i in range(1,2)]
list_Gestures=[i for i in range(1,9)]            # The eight hand gestures classes

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

# load mat file where the data are saved
filepath = browse_file()

# Use matlab scripts
eng = matlab.engine.start_matlab()
eng.eval("addpath('.\Matlab_lib');")
eng.eval("addpath('.\Matlab_lib\mPWM');")

[Gesture, Subject, Trial, X, DB]=eng.Load_sEMG_data(filepath,nargout=5,stdout=out,stderr=err)

X=np.asarray(X);
Gesture=np.asarray(Gesture).ravel();Subject=np.asarray(Subject).ravel(); Trial=np.asarray(Trial).ravel()

# Displays
blcd, classes, data_dic=Explore_dataset(Gesture)
# explore the dataset:  number f subjects, trails , gestures...
trials_labels=np.unique(Trial); print(trials_labels)

#list_Subjets=np.unique(Subject); print(list_Subjets)
#list_Gestures=np.unique(Gesture); print(list_Gestures)

#%% Split the data into training and testing
score=list();

#for trial_permutation in range(1):
# Get random training/testing split
#Trial_split=list(np.random.permutation(trials_labels).reshape(( 2, len(trials_labels)//2)))
Trial_split=[[1,3,5,7,9],[2,4,6,8,10]]

Trial_TR=list();Trial_TS=list();
for gesture in list_Gestures:
    Trial_TR.extend(list([ i for i in range(len(Gesture)) if Gesture[i]==gesture and Subject[i] in list_Subjets and Trial[i] in  Trial_split[0]]))
    Trial_TS.extend(list([ i for i in range(len(Gesture)) if Gesture[i]==gesture and Subject[i] in list_Subjets and Trial[i] in  Trial_split[1]]))



# Genrate the Training/Testing splits
y=Gesture
X_train= X[Trial_TR];  X_train_mat=matlab.double(X_train.tolist()); print(X_train)
y_train= y[Trial_TR];  y_train_mat=matlab.double(y_train.tolist())
X_test = X[Trial_TS];  X_test_mat=matlab.double(X_test.tolist())
y_test = y[Trial_TS]



# Displays
print('\n\n==> Genrate the Training/Testing splits ')
trials_labels_TR=np.unique(Trial[Trial_TR]); print('Training size', len(y_train) ,' ,  trials labels: ',trials_labels_TR)
trials_labels_TS=np.unique(Trial[Trial_TS]); print('Testing size', len(y_test) ,' ,  trials labels: ',trials_labels_TS)


#%%  ---------------------   QuPMWM-based FEATURE GENERATION   --------------------------
m=3                                         # kmers order
M=6#[i for i in range(6,7,2)]            # quantzation number of intervals
k=1.2#[i for i in range(0.8,1.3,0.1)]  # Quantizatin resolution

#%%  ---------------------    METHOD I: matlab scripts   --------------------------
#[fPWM_train,fPWM_test,mPWM_structure]=eng.mPWM_features_generation(M,k,m,X_train_mat,y_train_mat, X_test_mat,nargout=3, stdout=out,stderr=err)
#print(err.getvalue()); print(out.getvalue())
#
#Xf_train=np.asarray(fPWM_train)
#Xf_test=np.asarray(fPWM_test)



#%%  ---------------------    METHOD II: matlab functions   --------------------------
#  Quatization
#mu,sigma=np.mean(X_train), np.std(X_train)                       # Get    normal Distribution N(mu, sigma)
mu,sigma=eng.Split_Multi_classes_samples(X_train_mat,y_train_mat,nargout=2,stdout=out,stderr=err);
Levels, Level_intervals=Set_levels_Sigma_py(k,M,mu,sigma);
Level_intervals_mat=matlab.double(np.asarray(Level_intervals).tolist());

Q_train= mapping_levels(X_train,Level_intervals, Levels);Q_train_mat=matlab.double(Q_train.tolist());
Q_test = mapping_levels(X_test, Level_intervals, Levels);Q_test_mat=matlab.double(Q_test.tolist())


#%% Build the nPWM matrices for different kMers
print('\n-->  Generate the mPWM matrices')

mPWM_structure=eng.Build_mPWMs_Structure(m, Q_train_mat, y_train_mat, Level_intervals_mat,nargout=1,stdout=out,stderr=err);
print(out.getvalue()); print(err.getvalue());
mPWM_structure['Levels']

#%% Generate the Training features
print('\n-->  Generate the Training features  ')

mPWM_feature_train=eng.Generate_mPWM_features(mPWM_structure, Q_train_mat,nargout=1,stdout=out,stderr=err);
print(out.getvalue()); print(err.getvalue());

#%% Generate the Testing features
print('\n-->  Generate the Testing features  ')
Q_test_mat=matlab.double(Q_test.tolist())

mPWM_feature_test=eng.Generate_mPWM_features(mPWM_structure, Q_test_mat,nargout=1,stdout=out,stderr=err);

#%% Select fPWM features
name_features=list(mPWM_feature_train['C1'].keys())
name_classes =list(mPWM_feature_train.keys())

selected_feature=[0,1,2]#[1,5,8]

for C in name_classes:
    for f in name_features:
         f_new0=mPWM_feature_train[C][f]
         try:
             Xf_train=np.concatenate((Xf_train,f_new0), axis=1)
         except NameError:
             Xf_train=np.asarray(f_new0)

         f_new1=mPWM_feature_test[C][f]
         try:
             Xf_test=np.concatenate((Xf_test,f_new1), axis=1)
         except NameError:
             Xf_test=np.asarray(f_new1)



#%%   ----------------------------  Train/Test Split  ----------------------------
for name, clf in zip(names, classifiers):
    print('Train/Test Split using:',name)
    #% model training
    start_time = timeit.default_timer()
    clf.fit(Xf_train, y_train)
    time_train = timeit.default_timer() - start_time

    #% model testing
    start_time = timeit.default_timer()
    y_predicted= clf.predict(Xf_test)
    time_test = timeit.default_timer() - start_time

    #% model testing one sample
    start_time = timeit.default_timer()
    y1= clf.predict(Xf_test[0:1])
    time_test1 = timeit.default_timer() - start_time

    #% model evaluation
    accuracy,sensitivity, specificity, precision, recall, f1, AUC=Get_model_performnace(y_test,y_predicted)


    score.append(list([accuracy, sensitivity, specificity,precision, recall, f1, AUC, time_train, time_test]))
    print('accuracy=',accuracy, 'sensitivity=',sensitivity,'specificity=', specificity,'sensitivity=',sensitivity,
          'precision=', precision , 'recall=', recall , 'F1-Score=', f1 , 'AUC=',AUC,'time_train=', time_train, 's time_test=', time_test , 's')


    # ROC
#    fpr, tpr, AUC=Get_ROC_Curve(y_test,y_predicted)

Clf_score = pd.DataFrame(np.asarray(score).T, columns=names)
Clf_score['Scores']=   list(['Accuracy','Sensitivity', 'Specificity','Precision', 'Recall','F1-score', 'ROC-AUC','time_train(s)','time_test(s)'])

print('Train/Test Split  results :\n\n',Clf_score )


#%% ---------------------------- Save results ----------------------------

if save_excel==1:
    path='./Results'
    if not os.path.exists(path):
        # Create target Directory
        os.mkdir(path)

    path=path+'/'+feature_type

    if not os.path.exists(path):
        # Create target Directory
        os.mkdir(path)
    Clf_score.to_csv(path+'/train_test_'+ feature_type+ mat_filename[:-4]+'.csv', sep=',')



#%% Test Script CELL ################################################################################
print('The END.')
#
#X = trials_labels
#kf = KFold(n_splits=5,shuffle=True)
#kf.get_n_splits(X)
#for train_index, test_index in kf.split(X):
#    print("TRAIN:", train_index, "TEST:", test_index)
#    X_train, X_test = X[train_index], X[test_index]
#    y_train, y_test = y[train_index], y[test_index]
