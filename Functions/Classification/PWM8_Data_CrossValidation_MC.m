 %% Apply Leave One Out CV with PWM features using diffferent classifers
% X: The data  sample
% y  The Class
% clf: The calssifier:{'nbayes','logisticRegression','SVM','','',''}
% function [accuracy1,sz_fPWM]= Classify_LeaveOut_PWM(X,y,clf)
function [sz_fPWM, Avg_Accuracy,Avg_sensitivity,Avg_specificity,Avg_precision,Avg_gmean,Avg_f1score,Avg_AUC]=PWM8_Data_CrossValidation_MC(X, y,CV_type, K,type_clf)

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

for num_fold = 1:C.NumTestSets
    clearvars  PWM_* XP Xn
    
    
    trIdx = C.training(num_fold);                                            teIdx = C.test(num_fold);
    Idx= find(teIdx);
    
    X_train= X(trIdx,:);                                                     X_test= X(teIdx,:); 
    y_train= y(trIdx);                                                       y_test= y(teIdx);
    
    
    %% Quantization
    Q_train= mapping_levels(X_train,Level_intervals, Levels);
    Q_test= mapping_levels(X_test,Level_intervals, Levels);


%       Q_train=Q_test; y_train=y_test; save('workk.mat'); load('workk.mat')

    %% Mono-Mers  Position Weight Matrix-BASED FEATURES
    [Q1_Mer_train,name_Mer1] = Extract_Miers1(Q_train,Levels); 
    [Q1_Mer_test,name_Mer10] = Extract_Miers1(Q_test,Levels); 
     % Geneate the coresponding PWMs for eack 1-mer pattern
    PWM4Ds_Mer1=Multi_PWM4D_mers(Q1_Mer_train,y_train);[Na, Nb,NClasses,NMers1]=size(PWM4Ds_Mer1);
     
    
    %% Di-Mers  Position Weight Matrix-BASED FEATURES
    [Q2_Mer_train,name_Mer2] = Extract_Miers2(Q_train,Levels); 
    [Q2_Mer_test,name_Mer20] = Extract_Miers2(Q_test,Levels); 
     % Geneate the coresponding PWMs for eack 1-mer pattern
    PWM4Ds_Mer2=Multi_PWM4D_mers(Q2_Mer_train,y_train);[Na, Nb,NClasses,NMers2]=size(PWM4Ds_Mer2);
     
   
    %% Feature generation
    fPWM_Mer1_train= Generate_PWM3D8_features(Q1_Mer_train, PWM4Ds_Mer1);%,PWM3D8_Mer2);   
    fPWM_Mer1_test = Generate_PWM3D8_features(Q1_Mer_test,  PWM4Ds_Mer1);%,PWM3D8_Mer2);

    fPWM_Mer2_train= Generate_PWM3D8_features(Q2_Mer_train, PWM4Ds_Mer2);%,PWM3D8_Mer2);   
    fPWM_Mer2_test = Generate_PWM3D8_features(Q2_Mer_test,  PWM4Ds_Mer2);%,PWM3D8_Mer2);
    

     
    %% Classification  PWM features
    fPWM3D8_train=[fPWM_Mer1_train fPWM_Mer2_train];
    fPWM3D8_test=[fPWM_Mer1_test fPWM_Mer2_test];
    [Mdl,Accuracy(num_fold),ytrue,yfit]=Classify_Multi_Class_Data(type_clf, fPWM3D8_train, y_train, fPWM3D8_test, y_test);
  
    
    Mdl_SVM_mPWM=Mdl;
    
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

%% Funtions



% Quantization
function X=mapping_levels(X,Level_intervals, Levels)
    for i=1:size(X,1)
        for j=1:size(X,2)
             X(i,j)=Get_level(X(i,j),Level_intervals,Levels);    
        end
        
    end
d=1;
end


function L=Get_level(Vx,Level_intervals,Levels)  

    idx=find(Vx<=Level_intervals);

    if size(idx,2)==0
        L=Levels(end);
    else
       l=idx(1);
       L=Levels(l);

    end


    d=1;
end





