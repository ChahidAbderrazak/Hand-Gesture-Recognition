import os
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from Python_lib.Shared_Functions import *

 import matlab.engine
import io
out = io.StringIO()
err = io.StringIO()
#%%

names = ["SVM", "RBF SVM",#"Linear SVM",
         "Logistic Regression","Nearest Neighbors",
         "Decision Tree", "Random Forest",
         "Neural Net", "AdaBoost","Naive Bayes"]

classifiers = [ SVC(),   SVC(gamma=2, C=1),#SVC(kernel="linear", C=0.025),
               LogisticRegression(),#(random_state=0, solver='lbfgs',multi_class='multinomial'),
               KNeighborsClassifier(),#3
               DecisionTreeClassifier(max_depth=6),RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
               MLPClassifier(alpha=2),AdaBoostClassifier(),GaussianNB()]



#%% ---------------------------------------------------------------------------

y_predicted, time_train, time_test=Run_Training_test_Classification(clf[5], name[5],  X_train, y_train,X_test)

#% model evaluation
accuracy, sensitivity, specificity, precision, recall, f1, AUC=Get_model_performnace(y_test,y_predicted)
print('accuracy=',accuracy, 'sensitivity=',sensitivity,'specificity=', specificity,'sensitivity=',sensitivity,
      'precision=', precision , 'recall=', recall , 'F1-Score=', f1 , 'AUC=',AUC,'time_train=', time_train, 's time_test=', time_test , 's')
#%% ---------------------------- Run one classifeir  ----------------------------
Idx_sel=indices[0:1000]
clf, name=[SVC(kernel="linear"), LogisticRegression()] , ['SVM-Lin','Logistic Regression']
y_predicted, time_train, time_test=Run_Training_test_Classification(clf[1], name[1],  X_train[:,Idx_sel], y_train,X_test[:,Idx_sel])

#% model evaluation
accuracy, sensitivity, specificity, precision, recall, f1, AUC=Get_model_performnace(y_test,y_predicted)
print('accuracy=',accuracy, 'sensitivity=',sensitivity,'specificity=', specificity,'sensitivity=',sensitivity,
      'precision=', precision , 'recall=', recall , 'F1-Score=', f1 , 'AUC=',AUC,'time_train=', time_train, 's time_test=', time_test , 's')


#%%
# Start matlab
eng = matlab.engine.start_matlab()

print(err.getvalue())
print(out.getvalue())

eng.eval("addpath('.\Matlab_lib');")
eng.simple_script(nargout=0,stdout=out,stderr=err)

a,b,c=eng.f1(1,2,nargout=3,stdout=out,stderr=err)
print(a)
print(b)
print(c)
[a,b,c]=f1(a1,a2)

#ret = eng.sqrt(4.0)
#print(ret)
#a = eng.workspace['a']
#print(a)
#eng.quit()

#%%
# Loading the Digits dataset
digits = datasets.load_digits()

# To apply an classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target

# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
CV=5
clf_model=SVC()
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

clf_op_param, clf_op=Tuning_hyper_parameters(clf_model, tuned_parameters, CV,X_train, y_train)

print('\n\n ************ Test  using the best model parameters ***************\n')

y_true, y_pred = y_test, clf_op.predict(X_test)

accuracy,sensitivity, specificity, precision, recall, f1, AUC=Get_model_performnace(y_true, y_pred)
score=list([accuracy, sensitivity, specificity, precision, recall, f1, AUC])

# Note the problem is too easy: the hyperparameter plateau is too flat and the
# output model is the same for precision and recall with ties in quality.