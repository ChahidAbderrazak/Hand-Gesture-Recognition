 
%% ###############   Hand Gesture Detection using EMG 2019   ############################
% This script applies classification using Position Weight Matrices (PWM)

%% ###########################################################################
%  Author:
%  Abderrazak Chahid (abderrazak.chahid@gmail.com)
% Done: Jan,  2019
%
%% ###########################################################################
clearvars Output_results  Accuracy_av perform_output Accuracy_all Classifier
global Levels Level_intervals 
 
%% Classifier 
feature_type='PWM2_';

%% List of parameteres
%     list_M=2*[4:5];
%     list_k=[0.2 0.3];%[0.2:0.2:1.2]; 
    
%% #########################   Display   ################################
fprintf('\n --> Run CV  classification using : ');
d_data0=string(strcat(feature_type(1:end-1),'-based Features'));
fprintf(' %s\n ',d_data0);


%% Get the statistical properties of the data
[Data_pd,mu,sigma]=Split_Multi_classes_samples(X,y);


%% Script Starts
    cntm=1;Acc_op=0;cnt_inc=0;
    
for M= list_M                % Number of levels

    cnt = 1;
    list_k=1.5/M;
    N_M=max(size(list_M));N_k=max(size(list_k)); 

    for k=list_k

%% get the notmal Distribution N(mu, sigma)
        [Levels, Level_intervals]=Set_levels_Sigma(k,M,mu,sigma);
        d=1;


%% Cross-Validation
        subject=1;
        tic
%         [Accuracy,sz_fPWM]=Classify_LeaveOut_PWM(X,y,type_clf);starplus='StarPlus';
        [sz_fPWM, Accuracy,Avg_sensitivity,Avg_specificity,Avg_precision,Avg_gmean,Avg_f1score,Avg_AUC]=PWM_Data_CrossValidation(X, y,CV_type, K,type_clf);

        exec_time=toc;
        
   %% get the LOO performance
        Accuracy_all(cntm,cnt,subject)=Accuracy
        resolution=k*sigma;

        cnt_inc=cnt_inc+1;Output_results(cnt_inc,:)=[M,mu,sigma,k,resolution,sz_fPWM,Accuracy,exec_time];
        Classifier{cnt_inc}=type_clf;
        
        % Get the best accuracy 
        if Acc_op<Accuracy
            PWM_op_results=[M,mu,sigma,k,resolution,sz_fPWM,Accuracy,exec_time];
            Acc_op=Accuracy;
            
            % {'Vector_Size','Accuracy','Sensitivity','Specificity','Precision','Gmean','F1score'};
            CV_results_op=[sz_fPWM, Accuracy,Avg_sensitivity,Avg_specificity,Avg_precision,Avg_gmean,Avg_f1score,Avg_AUC];
            
            %{'Dataset','Gestures','size','Method','parameters','CV','K','Classifier'}
            CV_config_op={noisy_file,strcat('P=',num2str(class_p),', N=',num2str(class_n)), num2str(size(X,1)),feature_type(1:end-1), strcat('M=',num2str(M),', k=',num2str(k)),CV_type,num2str(K),type_clf };



        end
        
        
        cnt=cnt+1;

    end
    
%     Accuracy_av(cntm,:)=mean(Accuracy_all(:,:,cntm));
    cntm=cntm+1;
    d=1;
end


%% Plot the Accuracy VS levels 
    figr=30;Plot_Levels_VS_Accuracy_avg(list_k,list_M,Accuracy_all,figr)
     
%% Add the  results to a table
    colnames={'M','mu','sigma','k','resolution','Vector_Size','Accuracy','Time'};
    perform_output= array2table(Output_results, 'VariableNames',colnames);
    perform_output.Classifier=Classifier';

%% Save the PWM classification results
    pwm_param=strcat('_sigma1_k',num2str(N_k),'_M_',num2str(N_M));
    feature_TAG=pwm_param;        % features TAG
    
    save(strcat('./Classification_results/',feature_type,noisy_file,pwm_param,suff,'_norm',num2str(Normalization),'_',CV_type,'_',type_clf,'_Acc',num2str(Acc_op),'.mat'),'pwm_param','feature_TAG','perform_output', 'noisy_file','*_all','list_*','*pd','mu','sigma',...
                                                                          'PWM_op_results','X','y','suff','filename')                                                                                                                    


%% Get the best results of PWM2
   colnames_results={'Vector_Size','Accuracy','Sensitivity','Specificity','Precision','Gmean','F1score','AUC'};
   Comp_performance_Table= array2table(CV_results_op, 'VariableNames',colnames_results);
    
  
   colnames_results={'Dataset','Gestures','size','Method','parameters','CV','K','Classifier'};
   Comp_config_Table= array2table(CV_config_op, 'VariableNames',colnames_results);
    
   % Add the optimal parameters
   Comp_results_Table=[Comp_results_Table; [Comp_config_Table ,Comp_performance_Table]];
   
% % {'Vector_Size','Accuracy','Sensitivity','Specificity','Precision','Gmean','F1score'};
% CV_results_op=[sz_fPWM, Accuracy,Avg_sensitivity,Avg_specificity,Avg_precision,Avg_gmean,Avg_f1score];
% 
% %{'Dataset','Configuration','size','L','step','Method','parameters','CV','K','Classifier'}
% CV_config_op={noisy_file,num2str(Conf_Elctr), num2str(size(X,1)),num2str(L_max),num2str(Frame_Step),feature_type(1:end-1), strcat('M=',num2str(M),', k=',num2str(k)),CV_type,num2str(K),type_clf };
 

