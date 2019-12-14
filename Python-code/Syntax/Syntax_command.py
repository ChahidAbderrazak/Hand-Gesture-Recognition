# Start matlab
eng = matlab.engine.start_matlab()
eng.eval("addpath('.\Matlab_lib');")
eng.eval("addpath('.\Matlab_lib\mPWM');")
 ret = eng.dec2base(2**60,16,stdout=out,stderr=err)
print(err.getvalue())
print(out.getvalue())



#%% The classifier pipelinLoad mat files
loaded_mat_file = scipy.io.loadmat('./mat/'+mat_filename)
X_train = loaded_mat_file['fPWM_train']
y_train = loaded_mat_file['y_train'].ravel()
X_test = loaded_mat_file['fPWM_test']
y_test = loaded_mat_file['y_test'].ravel()
#%%  ---------------------   CLASSES + ENTRIES PREPPARATION   --------------------------

Mdl1_raw=Classification_Train_Test(names, classifiers, X_train, y_train, X_test, y_test)

#indices_frac,importances_frac=Feature_selection_using_Tree(X_train,y_train)
for nb_feature in range(1,6):
    Mdl2_raw=Classification_Train_Test(names, classifiers, X_train[:,list(indices_frac[:nb_feature])], y_train, X_test[:,list(indices_frac[:nb_feature])], y_test)

