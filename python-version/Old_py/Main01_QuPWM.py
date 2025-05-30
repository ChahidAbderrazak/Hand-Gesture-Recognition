# -*- coding: utf-8 -*-
"""
Created on  Jun 4 16:43:55 2019

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
#clear_all()
# results destination
save_excel=1

# PWM-based parameters
m=3                                         # kmers order
list_M=[i for i in range(6,7,2)]            # quantzation number of intervals
list_k=1.2#[i for i in range(0.8,1.3,0.1)]  # Quantizatin resolution
feature_type='MC_mPWM_'

# experiemnt  parameters
list_Subjets=[i for i in range(1,2)]
list_Gestures=[i for i in range(1,9)]            # The eight hand gestures classes

#%%

root_folder="R:/chahida/Projects-Results/Hand-Gesture-Recognition/QuPWM_ALL_feature"

project_name='MC_QuPWM_all_features'

# The classifier pipeline
names = ["SVM",# "RBF SVM",#"Linear SVM",
         "Logistic Regression","Nearest Neighbors",
         "Decision Tree", "Random Forest",
         "Neural Net", "AdaBoost","Naive Bayes"]

classifiers = [ SVC(),#   SVC(gamma=2, C=1),#SVC(kernel="linear", C=0.025),
               LogisticRegression(),#(random_state=0, solver='lbfgs',multi_class='multinomial'),
               KNeighborsClassifier(),#3
               DecisionTreeClassifier(max_depth=6),RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
               MLPClassifier(alpha=2),AdaBoostClassifier(),GaussianNB()]

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
list_Subjets=np.unique(Subject); print(list_Subjets)
#list_Gestures=np.unique(Gesture); print(list_Gestures)

#%% Split the data into training and testing
score=list();

#for trial_permutation in range(1):
# Get random training/testing split
Trial_split=list(np.random.permutation(trials_labels).reshape(( 2, len(trials_labels)//2)))

Trial_TR=list();Trial_TS=list();
for gesture in list_Gestures:
    Trial_TR.extend(list([ i for i in range(len(Gesture)) if Gesture[i]==gesture and Subject[i] in list_Subjets and Trial[i] in  Trial_split[0]]))
    Trial_TS.extend(list([ i for i in range(len(Gesture)) if Gesture[i]==gesture and Subject[i] in list_Subjets and Trial[i] in  Trial_split[1]]))



# Genrate the Training/Testing splits
y=Gesture
X_train= X[Trial_TR]
y_train= y[Trial_TR]
X_test = X[Trial_TS]
y_test = y[Trial_TS]


#%% Classification models
m=3        # the kPWM order
score=list(); accuracy=list();sensitivity=list(); specificity=list(); precision=list(); recall=list(); f1=list(); AUC=list()

for name, clf in zip(names, classifiers):

    #%----------------------------  Train/Test Split  -----------------------------------------
    print('Train/Test Split using:',name)
    #% model training
    start_time = timeit.default_timer()
    clf.fit(X_train, y_train)
    time_train = timeit.default_timer() - start_time

    #% model testing
    start_time = timeit.default_timer()
    y_predicted= clf.predict(X_test)
    time_test = timeit.default_timer() - start_time

    #% model testing one sample
    start_time = timeit.default_timer()
    y1= clf.predict(X_test[0:1])
    time_test1 = timeit.default_timer() - start_time

    #% model evaluation
    accn,sensn, specn, precn, recalln, f1n, AUCn=Get_model_performnace(y_test,y_predicted)


    accuracy.append(accn); sensitivity.append(sensn);specificity.append(specn); precision.append(precn); recall.append(recalln);f1.append( f1n); AUC.append(AUCn)
    score.append(list([ name,  accn,sensn, specn, precn, recalln, f1n, AUCn, time_train, time_test]))


# Compute the average accuracy of all epoches
accuracy=np.mean(accuracy); sensitivity=np.mean(sensitivity);specificity=np.mean(specificity);precision= np.mean(precision)
recall=np.mean(recall); f1=np.mean(f1); AUC=np.mean(AUC)
score.append(list([name,  accuracy, sensitivity, specificity, precision, recall, f1, AUC, time_train, time_test]))

print('accuracy=',accuracy, 'sensitivity=',sensitivity,'specificity=', specificity,'sensitivity=',sensitivity,'precision=', precision , 'recall=', recall , 'F1-Score=', f1 , 'AUC=',AUC,'time_train=', time_train, 's time_test=', time_test , 's')

    # ROC
#    fpr, tpr, AUC=Get_ROC_Curve(y_test,y_predicted)

Clf_score = pd.DataFrame(np.asarray(score), columns=list(['Classifier','Accuracy','Sensitivity', 'Specificity','Precision', 'Recall','F1-score', 'ROC-AUC','time_train(s)','time_test(s)']))#names_clf)

print('Train/Test Split  results :\n\n',Clf_score )


#%% ---------------------------- Save results ----------------------------
if save_excel==1:

path='./Results'
if not os.path.exists(path):
    # Create target Directory
    os.mkdir(path)

path=path+'/'+project_name

if not os.path.exists(path):
    # Create target Directory
    os.mkdir(path)

Clf_score.to_csv(path+'/train_test_'+ mat_filename[:-4]+'.csv', sep=',')



#%% ---------------------------- Run one classifeir  ----------------------------
clf, name=LogisticRegression() , 'Logistic Regression  '
y_predicted, time_train, time_test=Run_Training_test_Classification(clf, name,  X_train, y_train,X_test)
 #% model evaluation
accuracy, sensitivity, specificity, precision, recall, f1, AUC=Get_model_performnace(y_test,y_predicted)
print('accuracy=',accuracy, 'sensitivity=',sensitivity,'specificity=', specificity,'sensitivity=',sensitivity,
      'precision=', precision , 'recall=', recall , 'F1-Score=', f1 , 'AUC=',AUC,'time_train=', time_train, 's time_test=', time_test , 's')

sel = VarianceThreshold(threshold=(.8 * (1 - .8)))

sel.fit_transform(X_train)

#%% ---------------------------- Tuning Hyper parametes  ----------------------------
## Set the parameters by cross-validation
#CV=0
#
##clf_model=SVC()
##tuned_parameters = [{'kernel': ['linear','rbf'], 'gamma': [1 ,2],'C': [ 1, 2]}]
#
## Tuning Hyper parametes
#clf_op_param, clf_op=Tuning_hyper_parameters(clf_model, tuned_parameters, CV,X_train, y_train)
#print('\n\n ************ Test  using the best model parameters ***************\n')
#y_true, y_pred = y_test, clf_op.predict(X_test)
#accuracy, sensitivity, specificity, precision, recall, f1, AUC=Get_model_performnace(y_test,y_pred)
#print('\n\naccuracy=',accuracy, 'sensitivity=',sensitivity,'specificity=', specificity,'sensitivity=',sensitivity,
#              'precision=', precision , 'recall=', recall , 'F1-Score=', f1 , 'AUC=',AUC)


#clf=SVC(gamma=2, C=1);
#clf.fit(X_train, y_train)
#y_predicted= clf.predict(X_test)
#
#yy=list(y_test)
#yy.append(y_predicted)
#err=np.sum(np.abs(y_test-y_predicted))