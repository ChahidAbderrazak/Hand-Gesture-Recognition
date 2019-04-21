    
%% ###############   Hand Gesture Detection using EMG 2019   ############################
% This script applies classification using Position Weight Matrices (PWM)

%% ###########################################################################
%  Author:
%  Abderrazak Chahid (abderrazak.chahid@gmail.com)
% Done: Feb,  2019
%
%% ###########################################################################
% input parameters
m=3 ;      %% the kmers order

%% ###########################################################################
if exist('cntm')~=1; cntm=1;cnt_inc=0;end  
if exist('mPWM_features')~=1; mPWM_features=1;end    
Acc_op=0;

%% Build the nPWM matrices for different kMers
tic
mPWM_structure=Build_mPWMs_Structure(m, Q_train, y_train, Level_intervals);
PG_time=floor(toc);    
    %% Generate the Training features 
tic
fprintf('\n-->  Generate the Training features  ')
    [mPWM_feature_train]=Generate_mPWMs_features(mPWM_structure, Q_train);

    %% Generate the Testing features 
fprintf('\n-->  Generate the Testing features  ')
    [mPWM_feature_test]=Generate_mPWMs_features(mPWM_structure, Q_test);


%% ###########  Perform QuPWM-based  Feature selection      ###########################
fprintf('\n-->  Perform QuPWM-based  Feature selection  ')
% test
        name_features = fieldnames(mPWM_feature_train.C1)   %% list of QuPWM features  
        Selected_features=[2 6 9];                          %% Select amoung the defined features in <name_features>
        mPWM_features=mPWM_features+1;
        %% Select the Training and Testing features
        
        [fPWM_train,mPWM_type]=Get_Slected_PWM_features(mPWM_feature_train,name_features,Selected_features);
        [fPWM_test]=Get_Slected_PWM_features(mPWM_feature_test,name_features,Selected_features);

        save(strcat(path_Classification,'QuPWM_m',num2str(m),'-Subj',num2str(list_Subjets),'-',mPWM_type(2:end-1),'.mat'),'M','k','mPWM_type','m','X','y', 'fPWM_train','y_train','fPWM_test','y_test')
FG_time=floor(toc);
%% ###########  Perform the MultiLabels classification   ###########################
tic
fprintf('\n-->  Perform the MultiLabels classification  ')    
        [Mdl.LR,Mdl.Accuracy,Mdl.ytrue,Mdl.yfit]=Classify_Multi_Class_Data(type_clf, fPWM_train, y_train, fPWM_test, y_test);
        
Clf_time=floor(toc); 
    %% $$$$$     Get the higher accuracy model     $$$$$
    sz_fPWM=size(fPWM_train,2);
    exec_time=PG_time+FG_time+Clf_time;
    Timing=strcat('PG',num2str(PG_time),'_FG',num2str(FG_time),'_Clf',num2str(Clf_time));
    Accuracy=Mdl.Accuracy;
    Get_optimal_MC_mPWMModel_Accuracy
         