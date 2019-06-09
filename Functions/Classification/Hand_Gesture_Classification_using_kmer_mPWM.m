 
%% ##########   Hand Gesture Detection using EMG 2019  ########################
% This script detects hand gestures based on sEMG signals based on QuPWM methods

%% ###########################################################################
%  Author:
%  Abderrazak Chahid (abderrazak.chahid@gmail.com)
% Done: March,  2019
%
%% ###########################################################################

% clear all;  close all ;format shortG;  addpath ./Functions ;Include_function 
% global y h filename  root_folder 
% 

%% Cross Validation parameters
K=5;%5;
type_clf='LR';% 'SVM';%  
CV_type='KFold';% 'HoldOut';%   'LOO';%                                   strcat(num2str(K),'-Folds_CV');%   
Normalization=0;

%% Define the experiment 
N_epoch=1
list_Gesture=1:8;            %% The eight hand gestures classes
list_Trials_TR=1:2:9;        %%  Training samples are the odd trails 
list_Trials_TS=2:2:10;       %%  Testing samples are the even trails 

for list_Subjets=1%[1:4] 
    
    for epoch=0%1:N_epoch
        
        %% Random choice of Training and testing trials
        Trials=unique(Data.Trial);Ntrials=max(size(Trials));
        Trial_shuffle=randperm(Ntrials);

%         list_Trials_TR=Trial_shuffle(1:Ntrials/2);        %%  Training samples are the odd trails 
%         list_Trials_TS=Trial_shuffle(1+Ntrials/2:end);       %%  Testing samples are the even trails 


        %% ### PWM-based parameters
        list_M=2*[3];%[3:5];%[4:7];%[5];%;
        list_k=1.2;%[0.8:0.1:1.3];%
        feature_type='MC_mPWM_';mPWM_features=0;

        %% ###########################################################################
        N_M=max(size(list_M));N_k=max(size(list_k)); 
        if exist('Comp_results_Table','var') == 0 , Comp_results_Table = table;  end                   % Table to save results
        if exist(path_Classification)~=7, mkdir(path_Classification);end
        suff=strcat('_Subj',num2str(list_Subjets),'_Norm',num2str(Normalization),'_',CV_type,'_',type_clf);
        List_classes =num2str(list_Gesture);
        Subjects=num2str(list_Subjets);
        Trials=strcat('TR;',num2str(list_Trials_TR),'-TS;',num2str(list_Trials_TS)   );
        %% Feature generation  & Classification
        Classification_Parameters

         %% Random sampling the input  data
        %         [X,shuffle_index]=Shuffle_data(X);y=y(shuffle_index);

        
        %% Split the data into training and testing
        [X_train,y_train, X_test,y_test]=Split_Training_Testing_sets(Data,list_Subjets,list_Gesture,list_Trials_TR,list_Trials_TS);

        %% Get the statistical properties of the data
        [mu,sigma]=Split_Multi_classes_samples(X,y);

        %% QuPWM-based features 

         for M= list_M                % Number of levels
            cnt = 1;
            k0=1.65/M;

            for k=k0*list_k

                M
                k
        %% get the normal Distribution N(mu, sigma)
            [Levels, Level_intervals]=Set_levels_Sigma(k,M,mu,sigma);

        %% get the  Optimal quatizer 
        % [Level_intervals,codebook] = lloyds(X_train(:),M);Levels=1:M;

        %% Quantization
            Q_train= mapping_levels(X_train,Level_intervals, Levels);
            Q_test= mapping_levels(X_test, Level_intervals, Levels);

        %% apply the QuPWM-based multi-labels classification    
        %     Gesture_recognition_paper_Classification
            Gesture_recognition_paper_Classification_optimized_mPWM

        %% Get the classification performance comparison     
            Update_classification_performance_tables

            cnt=cnt+1;

            end


            cntm=cntm+1;
            d=1;
         end

    end
    %% Save the PWM classification results
    pwm_param=strcat('_sigma1_k',num2str(N_k),'_M_',num2str(N_M),'NSubjs');
    feature_TAG=pwm_param;        % features TAG

    % excel sheet
    writetable(perform_output,strcat(path_Classification,feature_type,pwm_param,noisy_file,suff,'_Acc',num2str(Acc_op),'.xlsx'))

    % mat file sheet
    save(strcat(path_Classification,feature_type,pwm_param,noisy_file,suff,'_Acc',num2str(Acc_op),'.mat'),'pwm_param','feature_TAG','perform_output', 'noisy_file','*_all','list_*','*pd','mu','sigma',...
                                                                          'perform_output','Comp_results_Table','PWM_op_results','X','y','suff','filename')                                                                                                                    
    winopen(path_Classification);
    fprintf('\n#######  The experiment %s of  Subject  %s  classification is done succesfully ######\n\n', noisy_file,list_Subjets)

    
end
