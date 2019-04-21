 
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


for Normalization=0%0:1;
   
%% ### PWM-based Classification  
   Hand_Gesture_Classification_using_kmer_mPWM

%        PWM2_Classification;
%        PWM2_Classification_Multi_Classes

end

%% Display resutls 
Comp_results_Table


fprintf('\n#######  Subject  %s  classification is done succesfully ######\n\n',noisy_file)
