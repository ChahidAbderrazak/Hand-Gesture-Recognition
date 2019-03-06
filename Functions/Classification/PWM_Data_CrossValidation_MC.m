 %% Apply Leave One Out CV with PWM features using diffferent classifers
% X: The data  sample
% y  The Class
% clf: The calssifier:{'nbayes','logisticRegression','SVM','','',''}
% function [accuracy1,sz_fPWM]= Classify_LeaveOut_PWM(X,y,clf)
function [sz_fPWM, Avg_Accuracy,Avg_sensitivity,Avg_specificity,Avg_precision,Avg_gmean,Avg_f1score,Avg_AUC]=PWM_Data_CrossValidation_MC(X, y,CV_type, K,type_clf)

global Levels Level_intervals 


if strcmp(CV_type,'LOO')==1
    C = cvpartition(y, 'LeaveOut');
    
    fprintf('------------------------------------------------------------------\n')
    fprintf('           Multi-Class Leave-One-Out Cross Validation using %s           \n',type_clf )
    fprintf('------------------------------------------------------------------\n')


elseif strcmp(CV_type,'KFold')==1
    C = cvpartition(y, 'KFold',K);%,'Stratify',true);
    
    fprintf('------------------------------------------------------------------\n')
    fprintf('           The Multi-Class  %d-Folds Cross Validation using %s           \n',K,type_clf )
    fprintf('------------------------------------------------------------------\n')



else
    fprintf('\n --> Error: undefined Cross-Validation : %s',CV_type);

end
Bi_classes=  unique(y);


test_tedex=[];

for num_fold = 1%:C.NumTestSets
    clearvars  PWM*
    
%     trIdx = C.training(num_fold);  trIdx=find(trIdx==1);
%     teIdx = C.test(num_fold);      teIdx=find(teIdx==1);
%     Idx= find(teIdx);
    
    Idxp=Get_the_Class_Samples(list_Subjets,list_Trials,class_p,X_table);

    X_train= X(trIdx,:);                                                     X_test= X(teIdx,:); 
    y_train= y(trIdx);                                                       y_test= y(teIdx);
    
    %% Quantization
    Q_train= mapping_levels(X_train,Level_intervals, Levels);
    Q_test= mapping_levels(X_test, Level_intervals, Levels);
    
%     test_tedex=[test_tedex;teIdx];
    %% Get the positive and negative training samples to build PWM matrices
    PWM3Ds=Multi_PWM3D(Q_train,y_train);  
    
%     if abs(Np-Nn)>2
%         fprintf('Non balanced Training data\n\n')
%         CV_Status=No_blanced; 
% 
%     end
%     
    
d=1;
    
%     %% Ballaned testing 
%     Xtest_p=X_test(y_test==1,:);   Ntst_p=size(Xtest_p, 1);
%     
%     if abs(Ntst_p-Ntst_n)>0
%         fprintf('Non balanced testing data\n\n')
%         CV_Status=No_blanced; 
% 
%     end
    
     
    %% PWM features generation
    fPWM3D_train= Generate_PWM3D_features(Q_train, PWM3Ds);
    fPWM3D_test= Generate_PWM3D_features(Q_test, PWM3Ds);
   
    %% Perform the CV for the kth fold

%         [Mdl,Accuracy(num_fold),sensitivity(num_fold),specificity(num_fold),precision(num_fold),gmean(num_fold),f1score(num_fold),AUC(num_fold),ytrue,yfit]=Classify_Data(type_clf, fPWM3D_train, y_train, fPWM3D_test, y_test);
        [Mdl,Accuracy(num_fold),ytrue,yfit]=Classify_Multi_Class_Data(type_clf, fPWM3D_train, y_train, fPWM3D_test, y_test);

    
end

%% Average Accuracy 
Avg_Accuracy = sum(Accuracy)/C.NumTestSets;
Avg_sensitivity = -1;
Avg_specificity = -1;
Avg_precision = -1;
Avg_f1score =-1;
Avg_gmean = -1;
Avg_AUC=-1;;
Accuracy
Avg_Accuracy;
sz_fPWM=size(fPWM3D_train,2);



end



function PWM_letters()
    N_levels=size(Levels,2);
    % Assign to each level a letter
    Seq_letter=char([65:90 97:122  char(194:194+N_levels-52) ]); N_letters=size(Seq_letter,2); %   or Seq_letter='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'    
    
    for lv=1:N_levels
    Levels_ABC(lv)=Seq_letter(lv)
    
    end
    
    
    %% Convert seignal to levels
    Xp= mapping_levels(Xp,Level_intervals, Levels_ABC);
    Xn= mapping_levels(Xn,Level_intervals, Levels_ABC);
    %% Build the PWM matrices
    PWM_P = Generate_PWM_matrix(Xp, Levels_ABC);
    PWM_N = Generate_PWM_matrix(Xn, Levels_ABC);   
    
    
end


