 
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
% %% load  MEG data  
% %     Extracted_MEG_Samples
% %     Combine_patients_datasets
%     Load_saved_data

%% Cross Validation parameters
global Normalization
K=5;
type_clf= 'SVM';%'LR';% 
CV_type='KFold';%  'LOO';%                                   strcat(num2str(K),'-Folds_CV');%   


%% ###########################################################################
beta=0;EN_starplus=0
% Comp_results_Table = table;                     % Table to save results

for Normalization=0%0:1;
   
    
     %% Random sampling the input  data
    %     [X,shuffle_index]=Shuffle_data(X);y=y(shuffle_index);

        %% Display

        d_clf='--> Hand Gesture Detection using EMG 2019  :' ;
        d_data1=string(strcat('- CV type:',{''},CV_type,{''},', K=',num2str(K),',  Dataset: ',noisy_file ,', Electrodes configuration: ',num2str(Conf_Elctr),', Used Electrodes=',num2str(Electrode_list) ));
        d_data2=string(strcat('- Sampling:  L=',num2str(L_max),', Frame Step=',num2str(Frame_Step),', Th=',num2str(EN_b),', Norm=',num2str(Normalization)));

        fprintf('%s \n %s \n %s \n\n',d_clf,d_data1,d_data2);


    %% SCSA-based features
        SCSA_classification;

    %% PWM-based features
        PWM2_Classification;

    %% PWM8-based features
        PWM8_Classification;

                    
    
end
 
fprintf('\n################  The End ################\n\n')
