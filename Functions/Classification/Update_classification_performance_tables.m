
%% Add the  results to a table
    colnames={'Type_mPWM_Feature','M','mu','sigma','k','resolution','Vector_Size','Accuracy','Total_Time_s'};
    perform_output= array2table(Output_results, 'VariableNames',colnames);
    perform_output.Classifier=Classifier';
    perform_output.Feature=Feature_type';
    perform_output

%% Get the best results of PWM2
   colnames_results={'Type_mPWM_Feature','Vector_Size','Accuracy','Sensitivity','Specificity','Precision','Gmean','F1score','AUC'};
   Comp_performance_Table= array2table(CV_results_op, 'VariableNames',colnames_results);
    
   colnames_results={'epoch','Dataset','Subjects','Trials','Gestures','Dataset_size','Method','parameters','CV','K','Classifier','Timing'};
   Comp_config_Table= array2table(CV_config_op, 'VariableNames',colnames_results);
    
   % Add the optimal parameters
   Comp_results_Table=[Comp_results_Table; [Comp_config_Table ,Comp_performance_Table]];


%% Display resutls 
Comp_results_Table





% % {'Vector_Size','Accuracy','Sensitivity','Specificity','Precision','Gmean','F1score'};
% CV_results_op=[mPWM_features,sz_fPWM, Accuracy,Avg_sensitivity,Avg_specificity,Avg_precision,Avg_gmean,Avg_f1score,Avg_AUC];
% 
% %{'Dataset','Gestures','size','Method','parameters','CV','K','Classifier'}
% CV_config_op={noisy_file,strcat('P=',num2str(class_p),', N=',num2str(class_n)), num2str(size(X,1)),feature_type(1:end-1), strcat('M=',num2str(M),', k=',num2str(k)),CV_type,num2str(K),type_clf };
