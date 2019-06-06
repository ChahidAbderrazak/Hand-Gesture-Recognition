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
from sklearn.metrics import *

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit

from Shared_Functions import *

#%% 

#%%#################################################################################################
# input parameters 
save_excel=0;
project_name='SCSA'
mat_filename='SCSA_features0.1.mat'


#The classifier pipeline 
names = ["Nearest Neighbors", "Linear SVM",
        "RBF SVM", "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
        "Naive Bayes", "Quadratic Discriminant Analysis"]

classifiers = [KNeighborsClassifier(3), SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

##################################################################################################
#%% The classifier pipelinLoad mat files
loaded_mat_file = scipy.io.loadmat('./mat/'+mat_filename)
X = loaded_mat_file['X']
y = loaded_mat_file['y'].ravel()



# Multi Class Datset  Loading the Digits dataset
#digits = datasets.load_digits()
#
## To apply an classifier on this data, we need to flatten the image, to
## turn the data in a (samples, feature) matrix:
#n_samples = len(digits.images)
#X = digits.images.reshape((n_samples, -1))
#y = digits.target

#%% Simulated data 
#X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
#                           random_state=1, n_clusters_per_class=1)
#rng = np.random.RandomState(2)
#X += 2 * rng.uniform(size=X.shape)
#linearly_separable = (X, y)
#
#datasets = [make_moons(noise=0.3, random_state=0),
#            make_circles(noise=0.2, factor=0.5, random_state=1),
#            linearly_separable
#            ]
#X, y = datasets[0]

#%%-------------------------------------------------------------------------------------------
blcd, classes, data_dic=Explore_dataset(y)
score=list();
#X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=.2, random_state=42)
    
#  ---------------------------- Train/Test Split  ----------------------------
 
for name, clf in zip(names, classifiers):
    print('Train/Test Split using:',name)
    clf.fit(X_train, y_train)
    y_predicted= clf.predict(X_test)
    accuracy,sensitivity, specificity, precision, recall, f1, AUC=Get_model_performnace(y_test,y_predicted)
    score.append(list([accuracy, sensitivity, specificity,precision, recall, f1, AUC]))

    # ROC
#    fpr, tpr, AUC=Get_ROC_Curve(y_test,y_predicted)
    
Clf_score = pd.DataFrame(np.asarray(score).T, columns=names)
Clf_score['Scores']=   list(['Accuracy','Sensitivity', 'Specificity','Precision', 'Recall','F1-score', 'ROC-AUC'])
print('Train/Test Split  results :\n\n',Clf_score )

##%% ---------------------------- Save results  ----------------------------

if save_excel==1: 
        
    path='./Results'
    if not os.path.exists(path):
        # Create target Directory
        os.mkdir(path)
        
    path=path+'/'+project_name
    
    if not os.path.exists(path):
        # Create target Directory
        os.mkdir(path)
    Clf_score.to_csv(path+'/train_test_'+ project_name+ mat_filename[:-4]+'.csv', sep=',')



   
#%% ---------------------------- Cross validation  ----------------------------
#clf=classifiers[0]
CV_score_avg=pd.DataFrame()
for name, clf in zip(names, classifiers):
    print('Cross-Validation classification using:',name)

    if len(set(y))==2:
        used_scores = {'Accuracy':'accuracy','Precision':'precision', 'Recall':'recall','F1-score':'f1', 'ROC-AUC':'roc_auc'}
    else:
        used_scores = {'Accuracy':'accuracy','Precision':'precision_macro', 'Recall':'recall_macro','F1-score':'f1_macro'}

    CV = ShuffleSplit(n_splits=5, test_size=0.2)#, random_state=0)    
    CV_scores = cross_validate(clf, X, y, cv=CV, scoring=used_scores, return_train_score=False)   
    sorted(CV_scores.keys())
    avg_scores=np.mean(np.asanyarray(list(CV_scores.values())),axis=1)
    new_score_avg = pd.DataFrame(avg_scores.reshape(-1, len(avg_scores)),columns=list(CV_scores.keys()) )
    
    CV_score_avg=CV_score_avg.append(new_score_avg) 

# Add cissfiers column    
CV_score_avg['classifier']=names
print('Cross-Validation results :\n\n',CV_score_avg )

##%% ----------------------------  Save results ----------------------------
if save_excel==1: 
    path='./Results'
    if not os.path.exists(path):
        # Create target Directory
        os.mkdir(path)
        
    path=path+'/'+project_name
    
    if not os.path.exists(path):
        # Create target Directory
        os.mkdir(path)
    CV_score_avg.to_csv(path+'/CV_'+ project_name+ mat_filename[:-4]+'.csv', sep=',')


##%% ---------------------------- cell  ----------------------------
