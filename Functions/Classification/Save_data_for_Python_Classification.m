function Save_data_for_Python_Classification(mat_file, fPWM_train,fPWM_test ,y_train, y_test)
X_train=fPWM_train;
X_test=fPWM_test;

save(strcat(mat_file,'_pro.mat'),'X_train','X_test','y_train','y_test')