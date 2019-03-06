
%% #################  sEMG  data preparation KAUST    #########################
% This script prepare the data donloaded from : for matlab  usage. 
% Please make sure that you  have  have healthy and non-Heatlthy recodrs saved 
% in separate folder

%% ###########################################################################
%  Author:
%  Abderrazak Chahid (abderrazak.chahid@kasut.edu.sa)
% Done: Feb,  2019
%  
%% ###########################################################################

clear all; close all; addpath ./Functions
%% Load the Healthy sEMG 
fs=1000;                               % the sampling frequency
L_max=70;                              % Spikes frame size.  
Frame_Step=2;                          % sliding  frame step size
Results_path='../Extracted_gesture_data';

%% ############################   START HERE    ##############################
global sEMG Channels
mkdir(Results_path)
%% Select the Root folder
Root_folder = uigetdir; Root_folder=strcat(Root_folder,'\');

%% sEMG  records whith spikes
List_mat_files = dir(strcat(Root_folder ,'\*\*.mat'));

Number_trials=0;


% ######################## Read file in loop ########################
X=[];Gesture=[];
Trial_number=[];  ID_Subject=[]; 

for Subject_k=1:size(List_mat_files,1)
    
    subj= Subject_k;  
%     clearvars Xp Xn yp yn X Gesture  Gesture Trial_number Patient_Hlt SE_Hlt    ID_Subject ID_SE 

    %% Get the spikes from a Subject
    BD_Folder=List_mat_files(Subject_k).folder;  BD_Folder=strcat(BD_Folder,'\')
    Trial_k=List_mat_files(Subject_k).name;
    filename_mat=strcat(BD_Folder, Trial_k)  
      
       %% Get if this folder is for a healthy or Subject
        Slice_folders=regexp(BD_Folder,'\','split')
        sEMG_DB=string(Slice_folders(end-1))
        sEMG_DATASET=string(Slice_folders(end-2))

        %% Get the experiment infomation and details
%         [Subject_k, Gesture_k, Trial_k]=Extract_trial_info(Trial_k);
        
        %% Update the right cathegory 
        
        load(filename_mat)

        X=[X; data];
        Gesture=[Gesture; double(gesture)*ones(size(data,1),1)];
        ID_Subject=[ID_Subject; double(subject)*ones(size(data,1),1)];
        Trial_number=[Trial_number; double(trial)*ones(size(data,1),1)];
        
        Number_trials=Number_trials+1;

end
  
Gesture_type=unique(Gesture);
noisy_file=Get_the_Classes(Gesture_type);

% suff=strcat('_Nclss',num2str(max(size(Gesture))),'_L',num2str(L_max),'_FrStep',num2str(Frame_Step));
suff=strcat('_Nclss',num2str(max(size(Gesture_type))));
name_record=strcat('Nmtrials_',num2str(Number_trials),'_Size_',num2str(size(X,1)),'_',num2str(size(X,2)),'_Data_',noisy_file,suff);
path_record=strcat(Results_path,'/Subject',num2str(unique(ID_Subject)),'/',noisy_file(1:end-1));


%% Build the table data
colnames_results={'Gesture','Subject','Trial'};
data=[Gesture, ID_Subject, Trial_number];
Data= array2table(data, 'VariableNames',colnames_results);
Data.sEMG=X;
%% Save the classification data
   
if exist(path_record)~=7, mkdir(path_record);end
save(strcat(path_record,'/',name_record,'.mat'),'Data','Gesture_type','suff','Root_folder','noisy_file')
                                                                                                 
 

%% Plot the records
% 
% close all
% figr=43;figure(figr);
%     plot(X','r', 'LineWidth',2 ); hold on;% ylim([-1.5e-10 2E-10])
% %     plot(Xn','g', 'LineWidth',1 );  hold off
%     legend('Positive  Class  ')%, 'Negative Class  ')
% %     title(strcat('Subject:',noisy_file, {' '},', L=', num2str(L_max), ', Step=', num2str(Frame_Step) , {' '}, ', Number of Electrods: ',num2str(max(size(CHs)))))
%     xlabel('samples')
%     ylabel('time')
%     % ylim([-1.5e-10 2E-10])
%     set(gca,'fontsize',16)
%     
    %% save the figure of te used data
%     save_figure(path_record,figr,name_record) 
% close all
