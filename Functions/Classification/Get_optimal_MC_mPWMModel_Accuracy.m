    if exist('cntm')~=1; cntm=1;cnt_inc=0;end  
    if exist('cnt')~=1; cnt=1;end  

    if exist('mPWM_features')~=1; mPWM_features=1;end  
    
% Get the best accuracy 
        
        Accuracy_all(cntm,cnt,mPWM_features)=Accuracy;
        resolution=k*sigma;

        cnt_inc=cnt_inc+1;Output_results(cnt_inc,:)=[mPWM_features, M,mu,sigma,k,resolution,sz_fPWM,Accuracy, exec_time];
        Classifier{cnt_inc}=type_clf;
        Feature_type{cnt_inc}=mPWM_type(1:end-1);
Timing
        
        %%  compute parameters 
        Avg_sensitivity=-1; Avg_specificity=-1; Avg_precision=-1; Avg_gmean=-1; Avg_f1score=-1; Avg_AUC=-1;
%         Sensitivity=-1; Specificity=-1; Precision=-1; Gmean=-1; F1score=-1; AUC=-1;

        if Acc_op<Accuracy
            PWM_op_results=[mPWM_features, M,mu,sigma,k,resolution,sz_fPWM,Accuracy,exec_time];
            Acc_op=Accuracy;
            
            % {'Vector_Size','Accuracy','Sensitivity','Specificity','Precision','Gmean','F1score'};
            CV_results_op=[mPWM_features,sz_fPWM, Accuracy,Avg_sensitivity,Avg_specificity,Avg_precision,Avg_gmean,Avg_f1score,Avg_AUC];
            
            %{'Dataset','Gestures','size','Method','parameters','CV','K','Classifier'}
            CV_config_op={noisy_file, Subjects,Trials,List_classes, num2str(size(X,1)),mPWM_type(1:end-1), strcat('M=',num2str(M),', k=',num2str(k)),CV_type,num2str(K),type_clf, Timing };

        end
    

                
        