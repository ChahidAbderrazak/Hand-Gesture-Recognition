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

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_validate
from sklearn.model_selection import ShuffleSplit
import timeit

from Python_lib.Shared_Functions import *

#%%##################################################################################################
# input parameters
save_excel=1
#project_name = 'mPWM'
#mat_filename='./mat/Mier1_features.mat';
root_folder="R:/chahida/Projects-Results/Hand-Gesture-Recognition/QuPWM_ALL_feature"
project_name='MC_m3QuPWM_all'

for k in [4]:#range(1,5):
#    mat_filename='QuPWM_m3-Subj'+ str(k)+'-p1fPWM-m2fPWM1-m3fPWM1_pro.mat'
    mat_filename='epoch0QuPWM_m3-Subj4-m1fPWM-p1fPWM-m1fPWM1-m2fPWM-p2fPWM-m2fPWM1-m3fPWM-p3fPWM-m3fPWM1'
    print(mat_filename)

    # The classifier pipeline
    names = ["Logistic Regression",
             "Nearest Neighbors", "Linear SVM","RBF SVM",
             "Decision Tree", "Random Forest",
             "Neural Net", "AdaBoost","Naive Bayes"]

    classifiers = [LogisticRegression(),#(random_state=0, solver='lbfgs',multi_class='multinomial'),
        KNeighborsClassifier(3), SVC(kernel="linear", C=0.025),SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1),AdaBoostClassifier(),GaussianNB()]

    ##################################################################################################
    #%% The classifier pipelinLoad mat files
    loaded_mat_file = scipy.io.loadmat(root_folder +"/"+ mat_filename)
    X_train = loaded_mat_file['fPWM_train']#['X_train']#
    y_train = loaded_mat_file['y_train'].ravel()

    X_test = loaded_mat_file['fPWM_test']#['X_test']#
    y_test = loaded_mat_file['y_test'].ravel()
    blcd, classes, data_dic=Explore_dataset(y_test)


    fPWM_size = loaded_mat_file['fPWM_sizes'].ravel()
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
    #X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=.2, random_state=42)
    #blcd, classes, data_dic=Explore_dataset(y_test)

    #%%------------------------------------------------------------------------------------
    score=list();

    #  ----------------------------  Train/Test Split  ----------------------------
    clf, name=[SVC(kernel="linear"), LogisticRegression()] , ['SVM-Lin','Logistic Regression']

#    names, classifiers=["Linear SVM","RBF SVM"],[SVC(kernel="linear", C=0.025),SVC(gamma=2, C=1)]
    #names, classifiers=["SVM-RBF0","SVM-RBF1", "SVM-RBF2"],[SVC(gamma=1, C=10),SVC(gamma=1, C=200),SVC(C=200)]
#    project_name=project_name+'ZZ'
#    clf=classifiers[1]

    for name, clf in zip(names, classifiers):
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

        path=path+'/'+project_name

        if not os.path.exists(path):
            # Create target Directory
            os.mkdir(path)

        Clf_score.to_csv(path+'/train_test_'+ project_name+ mat_filename[:-4]+'.csv', sep=',')


#%% ---------------------------- cell  ----------------------------
## Set the parameters by cross-validation
#CV=0
#clf_model=SVC()
#tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1 ,2],'C': [ 1, 2]}]
#clf_op_param, clf_op=Tuning_hyper_parameters(clf_model, tuned_parameters, CV,X_train, y_train)
#print('\n\n ************ Test  using the best model parameters ***************\n')
#y_true, y_pred = y_test, clf_op.predict(X_test)


#clf=SVC(gamma=2, C=1);
#clf.fit(X_train, y_train)
#y_predicted= clf.predict(X_test)
#
#yy=list(y_test)
#yy.append(y_predicted)
#err=np.sum(np.abs(y_test-y_predicted))