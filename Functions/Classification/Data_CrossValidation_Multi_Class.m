%% K-Folds Cross validation using mltiple classifiers classifiers
% type_clf: the classifier { 'LR', 'SVM' }
function [Accuracy,Avg_Accuracy,Avg_sensitivity,Avg_specificity,Avg_precision,Avg_gmean,Avg_f1score,Avg_AUC]=Data_CrossValidation_Multi_Class(X, y,CV_type, K,type_clf)

if strcmp(CV_type,'LOO')==1
    C = cvpartition(y, 'LeaveOut');
    
    fprintf('------------------------------------------------------------------\n')
    fprintf('            Leave-One-Out Cross Validation using %s           \n',type_clf )
    fprintf('------------------------------------------------------------------\n')


elseif strcmp(CV_type,'KFold')==1
    C = cvpartition(y, 'KFold',K);
    
    fprintf('------------------------------------------------------------------\n')
    fprintf('            The %d-Folds Cross Validation using %s           \n',K,type_clf )
    fprintf('------------------------------------------------------------------\n')



else
    fprintf('\n --> Error: undefined Cross-Validation : %s',CV_type);

end


%     C = cvpartition(y, 'KFold',K,'Stratify',true);

for num_fold = 1:C.NumTestSets
    trIdx = C.training(num_fold);
    teIdx = C.test(num_fold);
    
    
    X_train= X(trIdx,:);              y_train= y(trIdx);
    X_test = X(teIdx,:);              y_test = y(teIdx);
    
   
    %% Get the positive and negative training samples to build PWM matrices
%     Xp=X_train(y_train==1,:);   Np=size(Xp, 1);
%     Xn=X_train(y_train==0,:);   Nn=size(Xn, 1); 
%     
%         if abs(Np-Nn)>2
%             fprintf('Non balanced testing data\n\n')
%             CV_Status=No_blanced; 
% 
%         end

        [Mdl,Accuracy(num_fold),ytrue,yfit]=Classify_Multi_Class_Data(type_clf, X_train, y_train, X_test, y_test);
 
end
 
Avg_Accuracy = sum(Accuracy)/C.NumTestSets;
Avg_sensitivity = -1;
Avg_specificity = -1;
Avg_precision = -1;
Avg_f1score = -1;
Avg_gmean = -1;
Avg_AUC=-1;

Accuracy;
Avg_Accuracy;




