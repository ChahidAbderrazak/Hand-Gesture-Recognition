 
%% ###############   Hand Gesture Detection using EMG 2019  ############################
% This script detects epileptic spikes bases onsignal processing methods

%% ###########################################################################
%  Author:
%  Abderrazak Chahid (abderrazak.chahid@gmail.com)
% Done: Dec,  2018
%
%% ###########################################################################

% clear all;  close all ;format shortG;  addpath ./Functions ;Include_function ;%log_html_file
% global y h filename  root_folder 
% 

%% load  MEG data  

%% Cross Validation parameters
global Normalization
K=5;%5;
type_clf= 'LR';% 'SVM';%
CV_type='KFold';%  'LOO';%                                   strcat(num2str(K),'-Folds_CV');%   


%% ###########################################################################
beta=0;EN_starplus=0;

if exist('Comp_results_Table','var') == 0 , Comp_results_Table = table;  end                   % Table to save results
if exist(path_Classification)~=7, mkdir(path_Classification);end

for Normalization=0%0:1;
   
    suff=strcat('_Norm',num2str(Normalization));
        %% Feature generation  & Classification
        Classification_Parameters
        
     %% Random sampling the input  data
%         [X,shuffle_index]=Shuffle_data(X);y=y(shuffle_index);


% % ### PWM-based Classification
    list_M=2*[6:10];
   %% PWM-based features
    tic
%        PWM2_Classification;
%        PWM2_Classification_Multi_Classes
       Gesture_recognition_paper_Classification

    Time_PWM2=toc

%     %% PWM8-based features 
%    list_M=2*[5:8];
%     tic
%         PWM8_Classification;
%     Time_PWM8=toc

                  
    
end
 
fprintf('\n################  The End ################\n\n')
