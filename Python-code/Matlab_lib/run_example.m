  
%% ##########   Hand Gesture Detection using EMG 2019  ########################
% This script detects hand gestures based on sEMG signals based on QuPWM methods

%% ###########################################################################
%  Author:
%  Abderrazak Chahid (abderrazak.chahid@gmail.com)
% Done: March,  2019
%
%% ###########################################################################

clear all;  close all ;format shortG;  addpath ./mPWM ;Include_function 
global y h filename  root_folder 


%% Cross Validation parameters
K=5;%5;
type_clf='LR';% 'SVM';%  
CV_type='KFold';% 'HoldOut';%   'LOO';%                                   strcat(num2str(K),'-Folds_CV');%   
Normalization=0;

%% Define the experiment 
list_Gesture=1:8;            %% The eight hand gestures classes
list_Trials_TR=1:2:9;        %%  Training samples are the odd trails 
list_Trials_TS=2:2:10;       %%  Testing samples are the even trails 
list_Subjets=1%[2:4] 
%% ### PWM-based parameters
m=3;     % kMers order
M=2*[3];%[3:5];%[4:7];%[5];%;
k=1.2;%[0.8:0.1:1.3];%
%% ----------------------------------------------------------------------------------
path_Classification='/QuPWM_ALL_feature/';             % Destination 
project_folder=pwd;project_folder = strsplit(project_folder,'\'); 
path_Classification=char(strcat('R:/chahida/Projects-Results/',char(project_folder(end)),path_Classification));


%% ###########################################################################
if exist('Comp_results_Table','var') == 0 , Comp_results_Table = table;  end                   % Table to save results
if exist(path_Classification)~=7, mkdir(path_Classification);end
suff=strcat('_Subj',num2str(list_Subjets),'_Norm',num2str(Normalization),'_',CV_type,'_',type_clf);
List_classes =num2str(list_Gesture);
Subjects=num2str(list_Subjets);

%% Random choice of Training and testing trials
% Trials=unique(Data.Trial);Ntrials=max(size(Trials));
% Trial_shuffle=randperm(Ntrials)
% 
% %         list_Trials_TR=Trial_shuffle(1:Ntrials/2);        %%  Training samples are the odd trails 
% %         list_Trials_TS=Trial_shuffle(1+Ntrials/2:end);       %%  Testing samples are the even trails 




Trials=strcat('TR;',num2str(list_Trials_TR),'-TS;',num2str(list_Trials_TS)   );

%% Random sampling the input  data
 Load_saved_data
%         [X,shuffle_index]=Shuffle_data(X);y=y(shuffle_index);

%% Split the data into training and testing
[X_train,y_train, X_test,y_test]=Split_Training_Testing_sets(Data,list_Subjets,list_Gesture,list_Trials_TR,list_Trials_TS);

%% Generate QuPWM features
[fPWM_train,fPWM_test,mPWM_structure]=mPWM_features_generation(M,k,m, X_train,y_train, X_test);
