% %     fPWM_train=[]; fPWM_test=[]; mPWM_type='';
% %     for m=1:mPWM_structure.m
% %         for C=unique([y_train;y_test])'
% %             eval(strcat('fPWM_train=[fPWM_train mPWM_feature_train.C',num2str(C),'.m',num2str(m),'fPWM];'));
% %             eval(strcat('fPWM_test=[fPWM_test mPWM_feature_test.C',num2str(C),'.m',num2str(m),'fPWM];'));
% %             mPWM_type=strcat(mPWM_type,',m',num2str(m),'fPWM');
% %         end
% %     end



name_features = fieldnames(mPWM_feature_train.C1);
   
mPWM_features=0;
Z=max(size(name_features));
combined_features='';
%% list the existing feaetures
Existing_features = 1:Z;
%% ###########  Try all PWM-based features  combinaisons with  ###########################

for sz_combinaison=3%1:Z

    new_combinaison = nchoosek(Existing_features,uint16(sz_combinaison));

    %% Build the features matrix for classification 
    for comb=1:size(new_combinaison,1)

        mPWM_features=comb;
        Selected_features=new_combinaison(comb,:); %Existing_features; % 

        [fPWM_train,mPWM_type]=Get_Slected_PWM_features(mPWM_feature_train,name_features,Selected_features);
        [fPWM_test,mPWM_type]=Get_Slected_PWM_features(mPWM_feature_test,name_features,Selected_features);

        
        
       
        
%% ###########  Perform the MultiLabels classification   ###########################
      tic
         eval(['[Mdl.LR',num2str(mPWM_features),',Accuracy_LR(mPWM_features),ytrue,yfit_LR1]=Classify_Multi_Class_Data(type_clf, fPWM_train, y_train, fPWM_test, y_test);'])
     
    %% $$$$$     Get the higher accuracy model     $$$$$
    sz_fPWM=size(fPWM_train,2);
    exec_time=toc;
    Accuracy=Accuracy_LR(mPWM_features);
    Get_optimal_MC_mPWMModel_Accuracy
        
    end
    
end



% %  close all
% % 
% % PWMs=Build_mPWMs_Structure(y_train, Q_train,m);
% % 
% % 
% % % % % heatmap(fPWM3D2_train(1:20,:));
% % % % classes = ismember(y,[1:4]);
% % % % parallelcoords(X(classes,:), 'group',y(classes),'standardize','on', 'labels',varNames)
% % %            
% % % for comb=1:7
% % %     PWM=PWM3Ds(:,:,comb);
% % %     PWM2=PWM3Ds(:,:,comb+1);
% % % 
% % %     heatmap(abs(PWM-PWM2));
% % %     N_PWM(comb,:)=vecnorm(PWM);
% % %     
% % % end
% % % 
% % % 
% % % 
% % % 
% % % % A=magic(4);
% % % % A=A(:,1:2)
% % % % SK_features=sum(A'.^2)'
% % % % 
% % % % 
% % % % Levels=10
% % % % 
% % % % %% Run the K-Cross Validation
% % % % [accuracy,Avg_accuracy, sensitivity, specificity, precision, gmean, f1score0]=K_Fold_CrossValidation_for_PWM(X, y, K, type_clf);
% % % % 
% % % % 
% % % % % % type_clf='SVM';% 'LR';% 
% % % % % % K=5;                     % K-folds CV 
% % % % % % 
% % % % % % X=SFP_SCSA_h1;
% % % % % % 
% % % % % % %% Cross valiadtion of the row data 
% % % % % % [accuracy,Avg_Accuracy,Avg_sensitivity,Avg_specificity,Avg_precision,Avg_gmean,Avg_f1score]=K_Fold_CrossValidation(X, y, K, type_clf);
% % % % % % Load_saved_data
% % % % % % X=Scale_down_to_unit(X);
% % % % % % 
% % % % % % N=size(X,1);
% % % % % % [Xp,Xn]=Split_classes_Data(X,y) ;
% % % % % % filename2 = strrep(filename,'_','__');
% % % % % % figure;  
% % % % % % histogram(Xp); hold on
% % % % % % histogram(Xn); hold on
% % % % % % title(strcat('Samples values distribution for N=',num2str(N),' , data:',filename2))
% % % % % %  
% % % % % % 
% % % % % % % % Run the K-Cross Validation
% % % % % % % % Raw data
% % % % % % % [accuracy,Avg_accuracy, sensitivity, specificity, precision, gmean, f1score0]=K_Fold_CrossValidation(X, y, K, type_clf);
% % % % % % % 
% % % % % % % % PWM features
% % % % % % % [accuracy,Avg_accuracy, sensitivity, specificity, precision, gmean, f1score0]=K_Fold_CrossValidation_for_PWM(X, y, K, type_clf);
% % % % % % % 
% % % % 
