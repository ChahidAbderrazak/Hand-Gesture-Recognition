
%% ##########   Hand Gesture Detection using EMG 2019  ########################
% This script detects hand gestures based on sEMG signals based on QuPWM methods

%% ###########################################################################
%  Author:
%  Abderrazak Chahid (abderrazak.chahid@gmail.com)
% Done: Dec,  2018
%
%% ###########################################################################
clc; clear all;  close all ;format shortG;  addpath ./Functions ;Include_function ;%log_html_file
global y h filename  root_folder 

% Data sources
data_Source='./Input_data/Extracted_gesture_data/';                     % Source 
path_Classification='/QuPWM_ALL_feature/';             % Destination 
project_folder=pwd;project_folder = strsplit(project_folder,'\'); 
path_Classification=char(strcat('R:/chahida/Projects-Results/',char(project_folder(end)),path_Classification));

%% ###########################################################################
Comp_results_aLL = table;                     % Table to save results
if exist(path_Classification)~=7, mkdir(path_Classification);end
List_Data_files = dir(strcat(data_Source,'**/*.mat'));

for file_k=1%:size(List_Data_files,1)
    file_k
    filename=List_Data_files(file_k).name;
    data_Source=List_Data_files(file_k).folder;
    cname=strcat(data_Source,'\', filename);   
    load(cname); 

    %% Apply this script on the current data file
    Hand_Gesture_Classification_using_kmer_mPWM     

    %% Save partially Obtained results 
    Comp_results_aLL=[Comp_results_aLL;Comp_results_Table];

    save(strcat(path_Classification,'MC_',noisy_file,CV_type,'_',type_clf,'_On',string(datetime('now','Format','yyyy-MM-dd''T''HHmmss')),'.mat'),...
                 'perform_output','Comp_results_Table','noisy_file','data_Source','cname','perform_output')                                                                                                                    

end

%% Display resutls 
Comp_results_aLL

%% Save Obtained results on all the dataset

% excel sheet
writetable(Comp_results_aLL,strcat(path_Classification,'AllData_',num2str(size(List_Data_files,1)),pwm_param,noisy_file,suff,'_On',string(datetime('now','Format','yyyy-MM-dd''T''HHmmss')),'.xlsx'))

save(strcat(path_Classification,'AllData_',num2str(size(List_Data_files,1)),pwm_param,noisy_file,suff,'_On',string(datetime('now','Format','yyyy-MM-dd''T''HHmmss')),'.mat'),...
           'Comp_results_aLL','List_Data_files','data_Source','perform_output')                                                                                                                    

winopen(path_Classification);

fprintf('\n################  The End ################\n\n')
                                   
                                   
