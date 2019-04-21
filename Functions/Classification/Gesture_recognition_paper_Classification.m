    
%% ###############   Hand Gesture Detection using EMG 2019   ############################
% This script applies classification using Position Weight Matrices (PWM)

%% ###########################################################################
%  Author:
%  Abderrazak Chahid (abderrazak.chahid@gmail.com)
% Done: Feb,  2019
%
%% ###########################################################################

    
%% Initialization 
%     list_M=10;    
%     list_k=1.2;%[0.8:0.1:1.2];
    
if exist('cntm')~=1; cntm=1;cnt_inc=0;end  
if exist('mPWM_features')~=1; mPWM_features=0;end    
Acc_op=0;
%% PWM based features
    tic; tic;
    %% Get the positive and negative training samples to build PWM matrices
    PWM3Ds=Multi_PWM3D(Q_train,y_train);  
     
%% PWM1 features generation
    fPWM3D_train= Generate_PWM3D_features(Q_train, PWM3Ds);
    fPWM3D_test= Generate_PWM3D_features(Q_test, PWM3Ds);
   
    %% Perform the MultiLabels classification
    mPWM_type='MC_m1PWM_';mPWM_features=mPWM_features+1;

%     type_clf='SVM';
%     [Mdl_SVM,Accuracy_SVM,ytrue,yfit_SVM]=Classify_Multi_Class_Data(type_clf, fPWM3D_train, y_train, fPWM3D_test, y_test);
    
    [Mdl_LR1,Accuracy_LR1,ytrue,yfit_LR1]=Classify_Multi_Class_Data(type_clf, fPWM3D_train, y_train, fPWM3D_test, y_test);
    
    %% $$$$$     Get the higher accuracy model     $$$$$
    sz_fPWM=size(fPWM3D_train,2);
    exec_time=toc;
    Accuracy=Accuracy_LR1;
    Get_optimal_MC_mPWMModel_Accuracy
    
%% PWM2  features generation
    tic
    fPWM3D2_train= Generate_PWM3D2_features(Q_train, PWM3Ds);
    fPWM3D2_test= Generate_PWM3D2_features(Q_test, PWM3Ds);
   
    %% Perform the MultiLabels classification
    mPWM_type='MC_m1PWM2_';mPWM_features=mPWM_features+1;
%     type_clf='SVM';
%     [Mdl_SVM,Accuracy_SVM,ytrue,yfit_SVM]=Classify_Multi_Class_Data(type_clf, fPWM3D_train, y_train, fPWM3D_test, y_test);
    
    [Mdl_LR2,Accuracy_LR2,ytrue,yfit_LR2]=Classify_Multi_Class_Data(type_clf, fPWM3D2_train, y_train, fPWM3D2_test, y_test);
    
    %% $$$$$     Get the higher accuracy model     $$$$$
    sz_fPWM=size(fPWM3D2_train,2);
    exec_time=toc;
    Accuracy=Accuracy_LR2;
    Get_optimal_MC_mPWMModel_Accuracy
    
    
%% PWM 1+2  features generation
    mPWM_type='MC_m1PWM_m1PWM2_';mPWM_features=mPWM_features+1;
    
    fPWM3D12_train=[fPWM3D_train fPWM3D2_train];    %heatmap(fPWM3D2_train);
    fPWM3D12_test=[fPWM3D_test fPWM3D2_test];
    
    [Mdl_LR12,Accuracy_LR12,ytrue,yfit_LR12]=Classify_Multi_Class_Data(type_clf, fPWM3D12_train, y_train, fPWM3D12_test, y_test);
    
    %% $$$$$     Get the higher accuracy model     $$$$$
    sz_fPWM=size(fPWM3D12_train,2);
    exec_time=toc;
    Accuracy=Accuracy_LR12;
    Get_optimal_MC_mPWMModel_Accuracy
    
 
