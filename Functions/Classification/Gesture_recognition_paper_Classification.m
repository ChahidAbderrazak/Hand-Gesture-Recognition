    
%% ###############   Hand Gesture Detection using EMG 2019   ############################
% This script applies classification using Position Weight Matrices (PWM)

%% ###########################################################################
%  Author:
%  Abderrazak Chahid (abderrazak.chahid@gmail.com)
% Done: Feb,  2019
%
%% ###########################################################################
global Levels Level_intervals 

type_clf= 'LR';%'SVM';%

%% Split the data into training and testing
    list_Subjets=1;
    list_Trials_TR=1:2:9;
    list_Trials_TS=2:2:10;
    list_Gesture=1:8;
    
    [X_train,y_train, X_test,y_test]=Split_Training_Testing_sets(Data,list_Subjets,list_Gesture,list_Trials_TR,list_Trials_TS);
    
%% Get the statistical properties of the data
    [Data_pd,mu,sigma]=Split_Multi_classes_samples(X,y);

for M= 10%list_M                % Number of levels
    cnt = 1;
    list_k=1.65/M;

    for k=list_k*1.2%[0.8:0.1:1.2]
        
        M
        k
%% get the normal Distribution N(mu, sigma)
    [Levels, Level_intervals]=Set_levels_Sigma(k,M,mu,sigma);

%% get the  Optimal quatizer 
% [Level_intervals,codebook] = lloyds(X_train(:),M);Levels=1:M;

%% Quantization
    Q_train= mapping_levels(X_train,Level_intervals, Levels);
    Q_test= mapping_levels(X_test, Level_intervals, Levels);

    
%% PWM based features

    %% Get the positive and negative training samples to build PWM matrices
    PWM3Ds=Multi_PWM3D(Q_train,y_train);  
     
%% PWM1 features generation
    fPWM3D_train= Generate_PWM3D_features(Q_train, PWM3Ds);
    fPWM3D_test= Generate_PWM3D_features(Q_test, PWM3Ds);
   
    %% Perform the MultiLabels classification
%     type_clf='SVM';
%     [Mdl_SVM,Accuracy_SVM,ytrue,yfit_SVM]=Classify_Multi_Class_Data(type_clf, fPWM3D_train, y_train, fPWM3D_test, y_test);
    
    [Mdl_LR1,Accuracy_LR1,ytrue,yfit_LR1]=Classify_Multi_Class_Data(type_clf, fPWM3D_train, y_train, fPWM3D_test, y_test);

%% PWM2  features generation
    fPWM3D2_train= Generate_PWM3D2_features(Q_train, PWM3Ds);
    fPWM3D2_test= Generate_PWM3D2_features(Q_test, PWM3Ds);
   
    %% Perform the MultiLabels classification
%     type_clf='SVM';
%     [Mdl_SVM,Accuracy_SVM,ytrue,yfit_SVM]=Classify_Multi_Class_Data(type_clf, fPWM3D_train, y_train, fPWM3D_test, y_test);
    
    [Mdl_LR2,Accuracy_LR2,ytrue,yfit_LR2]=Classify_Multi_Class_Data(type_clf, fPWM3D2_train, y_train, fPWM3D2_test, y_test);
    
%% PWM 1+2  features generation
    fPWM3D12_train=[fPWM3D_train fPWM3D2_train];    %heatmap(fPWM3D2_train);
    fPWM3D12_test=[fPWM3D_test fPWM3D2_test];
    
    [Mdl_LR12,Accuracy_LR12,ytrue,yfit_LR12]=Classify_Multi_Class_Data(type_clf, fPWM3D12_train, y_train, fPWM3D12_test, y_test);
    

    

%% mPWM based features
% 
%     %% Mono-Mers  Position Weight Matrix-BASED FEATURES
%     [Q1_Mer_train,name_Mer1] = Extract_Miers1(Q_train,Levels); 
%     [Q1_Mer_test,name_Mer10] = Extract_Miers1(Q_test,Levels); 
%      % Geneate the coresponding PWMs for eack 1-mer pattern
%     PWM4Ds_Mer1=Multi_PWM4D_mers(Q1_Mer_train,y_train);[Na, Nb,NClasses,NMers1]=size(PWM4Ds_Mer1);
%      
%    
%     %% Mer1 Feature generation
%     fPWM_Mer1_train= Generate_PWM3D8_features(Q1_Mer_train, PWM4Ds_Mer1);  
%     fPWM_Mer1_test = Generate_PWM3D8_features(Q1_Mer_test,  PWM4Ds_Mer1);
%     
%     %% Perform the MultiLabels classification
% 
%     [Mdl_m1LR,Accuracy_m1LR,ytrue,yfit_m1LR]=Classify_Multi_Class_Data(type_clf, fPWM_Mer1_train, y_train, fPWM_Mer1_test, y_test);
% 
%         
%         
%     %% Di-Mers  Position Weight Matrix-BASED FEATURES
%     [Q2_Mer_train,name_Mer2] = Extract_Miers2(Q_train,Levels); 
%     [Q2_Mer_test,name_Mer20] = Extract_Miers2(Q_test,Levels); 
%      % Geneate the coresponding PWMs for eack 1-mer pattern
%     PWM4Ds_Mer2=Multi_PWM4D_mers(Q2_Mer_train,y_train);[Na, Nb,NClasses,NMers2]=size(PWM4Ds_Mer2);
%         
%     %% Mer2 Feature generation
%     fPWM_Mer2_train= Generate_PWM3D8_features(Q2_Mer_train, PWM4Ds_Mer2); 
%     fPWM_Mer2_test = Generate_PWM3D8_features(Q2_Mer_test,  PWM4Ds_Mer2);
%     
%     %% Classification  PWM features
%     [Mdl_m2LR,Accuracy_m2LR,ytrue,yfit_m2LR]=Classify_Multi_Class_Data(type_clf, fPWM_Mer2_train(:,1:5), y_train, fPWM_Mer2_test(:,1:5), y_test);
%        
% %% Di-Mers  + PWM1+2
% 
%     fPWM3D123_train=[fPWM3D12_train fPWM_Mer2_train];
%     fPWM3D123_test=[fPWM3D12_test  fPWM_Mer2_test];
% 
%      [Mdl_m2LR12,Accuracy_m2LR12,ytrue,yfit_m2LR12]=Classify_Multi_Class_Data(type_clf, fPWM3D123_train, y_train, fPWM3D123_test, y_test);

    end
    
    
end
        