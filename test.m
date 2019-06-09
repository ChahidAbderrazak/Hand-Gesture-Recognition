clc; clear all;  
close all ;format shortG;  addpath ./Functions ;Include_function ;%log_html_file

load('example_QuPWM_features.mat')

type_clf='LR';
Selected_features_op=Scan_optimal_QuPWM_features(type_clf, mPWM_feature_train, y_train, mPWM_feature_test, y_test)

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
