
%% ###############   Hand Gesture Detection using EMG 2019 All DATA  ############################
% This script detects epileptic spikes bases on signal processing methods

%% ###########################################################################
%  Author:
%  Abderrazak Chahid (abderrazak.chahid@gmail.com)
% Done: Dec,  2018
%
%% ###########################################################################

clear all;  close all ;format shortG;  addpath ./Functions ;Include_function ;%log_html_file
global y h filename  root_folder 
Comp_results_aLL = table;                     % Table to save results

data_Source='./Input_data/Extracted_gesture_data/';                     % Source 
path_Classification='./Classification_results/Comparison/';             % Destination 


%% ###########################################################################
if exist(path_Classification)~=7, mkdir(path_Classification);end

List_Data_files = dir(strcat(data_Source,'**/*.mat'));

for file_k=1%:size(List_Data_files,1)
    file_k
    filename=List_Data_files(file_k).name;
    data_Source=List_Data_files(file_k).folder;
    cname=strcat(data_Source,'\', filename);   
    load(cname); 

    %% Apply this script on the current data file
%     for class_p=1:4
%                 
%         class_n=1+class_p;
%         class_p=1:4;
%         class_n=5:8;
        Main_EMG_Classification2019
     
%     end
     
%% Save partially Obtained results 
Comp_results_aLL=[Comp_results_aLL;Comp_results_Table];

save(strcat(path_Classification,'MC_',noisy_file,CV_type,'_',type_clf,'_On',string(datetime('now','Format','yyyy-MM-dd''T''HHmmss')),'.mat'),...
             'Comp_results_Table','noisy_file','data_Source','cname')                                                                                                                    
                                                                     
end


%% Save Obtained results on all the dataset

save(strcat(path_Classification,'MC_',num2str(size(List_Data_files,1)),'_Dataset_',CV_type,'_',type_clf,'_On',string(datetime('now','Format','yyyy-MM-dd''T''HHmmss')),'.mat'),...
           'Comp_results_aLL','List_Data_files','data_Source')                                                                                                                    

fprintf('\n################  The End ################\n\n')
                                   
                                   
