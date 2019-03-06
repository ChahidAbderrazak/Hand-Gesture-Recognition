 %% Apply Leave One Out CV with PWM features using diffferent classifers
% X: The data  sample
% y  The Class
% clf: The calssifier:{'nbayes','logisticRegression','SVM','','',''}
% function [accuracy1,sz_fPWM]= Classify_LeaveOut_PWM(X,y,clf)
function [sz_fPWM, Avg_Accuracy,Avg_sensitivity,Avg_specificity,Avg_precision,Avg_gmean,Avg_f1score,Avg_AUC]=PWM8_Data_CrossValidation(X, y,CV_type, K,type_clf)

global Levels Level_intervals 


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
Bi_classes=  unique(y);

for num_fold = 1:C.NumTestSets
    clearvars  PWM_* XP Xn
    
    
    trIdx = C.training(num_fold);                                            teIdx = C.test(num_fold);
    Idx= find(teIdx);
    
    X_train= X(trIdx,:);                                                     X_test= X(teIdx,:); 
    y_train= y(trIdx);                                                       y_test= y(teIdx);
    
    
    %% Get the positive and negative training samples to build PWM matrices
    Xp=X_train(y_train==1,:);   Np=size(Xp, 1);
    Xn=X_train(y_train==0,:);   Nn=size(Xn, 1);
    
    %% Quantization
    Q_train= mapping_levels(X_train,Level_intervals, Levels);
    Q_test= mapping_levels(X_test,Level_intervals, Levels);


    %% Build the PWM matrices   Mers1
%      [PWMp_Mer1,PWMn_Mer1]= Generate_PWM8_matrix(Q_train,y_train);
%     fPWM_train= Generate_PWM8_features(Q_train, PWMp_Mer1, PWMn_Mer1);       
%     fPWM_test = Generate_PWM8_features(Q_test,  PWMp_Mer1, PWMn_Mer1);
      
    %% Build the PWM matrices   Mers1 Mer2
   [PWM3D_Mer1, PWM3D_Mer2]= Generate_PWM3D8_matrix(Q_train,y_train);
   
       
   %% extract the features  

    fPWM3D8_train= Generate_PWM3D8_features(Q_train, PWM3D_Mer1,PWM3D_Mer2);       
    fPWM3D8_test = Generate_PWM3D8_features(Q_test,  PWM3D_Mer1,PWM3D_Mer2);

    %% Perform the CV for the kth fold

    [Mdl,Accuracy(num_fold),sensitivity(num_fold),specificity(num_fold),precision(num_fold),gmean(num_fold),f1score(num_fold),AUC(num_fold),ytrue,yfit,score]...
    =Classify_Data(type_clf, fPWM3D8_train, y_train, fPWM3D8_test, y_test);
    
    Mdl_SVM_mPWM=Mdl;
    
end

%% Average Accuracy 
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
sz_fPWM=size(fPWM3D8_train,2);



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






function fPWM= Generate_PWM3D8_features(Input_Sequence, PWM_P, PWM_N,PWMp_Mer2,PWMn_Mer2)
global Levels

    [Mer1_Seq,name_Mer1] = Extract_Miers1(Input_Sequence,Levels);
    fPWM1 = Apply_General_PWM_feature_generator(Mer1_Seq, PWM_P, PWM_N); 
    fPWM=fPWM1;
    [Mer2_Seq, name_Mer2] = Extract_Miers2(Input_Sequence,Levels);
    fPWM2 = Apply_General_PWM_feature_generator(Mer2_Seq, PWMp_Mer2,PWMn_Mer2);
    
%     fPWM2 = Apply_General_PWM_feature_generator3D(Mer2_Seq, PWMp_Mer2,PWMn_Mer2);

    fPWM=[fPWM1 fPWM2];
    
    
%     [Mer3_Seq, name_Mer2] = Extract_Miers3(Input_Sequence,Levels);
%     fPWM3 = Apply_General_PWM_feature_generator(Mer3_Seq, PWMp_Mer3,PWMn_Mer3);


    d=1;
    
end

function PWM_letters()
[ X_1Mer,name_1Mer] = Extract_Miers1(X,Levels); 
[ X_2Mer,name_1Mer] = Extract_Miers2(X,Levels); 

 fPWM1 = Apply_General_PWM_feature_generator(X_1Mer, PWM_P, PWM_N); 
 fPWM2 = Apply_General_PWM_feature_generator(X_2Mer, PWMp_2Mer,PWMn_2Mer);
 fmPWM=[fPWM1 fPWM2];

        
    fPWM_train= Generate_PWM8_features(Q_train, PWMp_Mer1, PWMn_Mer1,PWMp_Mer2,PWMn_Mer2);       

    N_levels=size(Levels,2);
    % Assign to each level a letter
    Q_letter=char([65:90 97:122  char(194:194+N_levels-52) ]); N_letters=size(Q_letter,2); %   or Q_letter='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'    
    
    for lv=1:N_levels
    Levels_ABC(lv)=Q_letter(lv)
    
    end
    
    
    %% Convert seignal to levels
    Xp= mapping_levels(Xp,Level_intervals, Levels_ABC);
    Xn= mapping_levels(Xn,Level_intervals, Levels_ABC);
    %% Build the PWM matrices
    PWM_P = Generate_PWM_matrix(Xp, Levels_ABC);
    PWM_N = Generate_PWM_matrix(Xn, Levels_ABC);   
end  
    