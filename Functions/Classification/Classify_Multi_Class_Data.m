function [Mdl,accuracy,ytrue,yfit,score]= Classify_Multi_Class_Data(type_clf, X_train, y_train, X_test, y_test)

fprintf('\n-->  Data Classification using %d Classes ', max(size(unique(y_test))));

switch type_clf
    case 'LR' 
        [Mdl,accuracy,ytrue,yfit,score]= Multi_Class_LR_classifier(X_train, y_train, X_test, y_test);

    case 'SVM'
            [Mdl,accuracy,ytrue,yfit,score]= Multi_Class_SVM_classifier(X_train, y_train, X_test, y_test);
 
      otherwise
        
        warning(strcat('The chosen classifier:',type_clf,' is not available.'));
       
end
  

accuracy
   


function [Mdl,accuracy0,sensitivity0,specificity0,precision0,gmean0,f1score0,AUC,ytrue,yfit,score]=LR_classifier(X_train, y_train, X_test, y_test)
    Combine_TR=[X_train, y_train];
    Combine_TS=[X_test, y_test];

% %%% ####################################### 
[M,N]=size(Combine_TR);
training_set= array2table(NO_T(Combine_TR));
training_set.class = Combine_TR(:,end);

 
[M_TS,N_TS]=size(Combine_TS);
testing_set = array2table(NO_T(Combine_TS));
testing_set.class = Combine_TS(:,end);

%% Model training
Mdl= fitglm(training_set,'linear','Distribution','binomial','link', 'logit');


%% Model_testing 

% yfit=trainedClassifier.predictFcn(testing_set);
yfit0 = Mdl.predict(testing_set);score=yfit0;
yfit0=yfit0-min(yfit0);yfit0=yfit0/max(yfit0);
yfit=double(yfit0>0.5);

%% Compute the accuracy
[accuracy0,sensitivity0,specificity0,precision0,gmean0,f1score0]=prediction_performance(testing_set.class, yfit);

ytrue=Combine_TS(:,end);


%% Compute the ROC curve
[X,Y,T,AUC] = perfcurve(y_test ,score,1);
% Plot the ROC curve.
figure(201);
plot(X,Y)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification by Logistic Regression')
legend(strcat('AUC=',num2str(AUC)))
set(gca,'fontsize',16)
grid on


function [Mdl_Multi_Class,accuracy0,y_test,yfit,scores]=Multi_Class_tTree_classifier(X_train, y_train, X_test, y_test)

tTree = templateTree('surrogate','on');
tEnsemble = templateEnsemble('GentleBoost',1000,tTree);
% pool = parpool; % Invoke workers
options = statset('UseParallel',true);
Mdl_Multi_Class = fitcecoc(X_train,y_train,'Coding','onevsall','Learners',tEnsemble,...
                'FitPosterior',1,'Options',options);
            
[yfit,scores] = predict(Mdl_Multi_Class,X_test);
[accuracy0]=Multi_Class_prediction_performance(y_test , yfit);

function [Mdl_Multi_Class,accuracy0,y_test,yfit,scores]=Multi_Class_LR_classifier(X_train, y_train, X_test, y_test)
tic
t = templateLinear('Learner','logistic');
Name_Classes=unique(y_train)';
%% Train an ECOC multiclass model using the default options.
Mdl_Multi_Class = fitcecoc(X_train,y_train,'Learners',t,'FitPosterior',1,'ClassNames',Name_Classes,'Verbose',0);
time_TR=toc;

tic;
[yfit,scores] = predict(Mdl_Multi_Class,X_test);
if size(yfit,1)~=size(y_test,1)
    yfit=yfit';
end
time_TS=toc;
[accuracy0]=Multi_Class_prediction_performance(y_test , yfit);

% %% Compute the ROC curve
% score=scores(:,2);
% [X,Y,T,AUC] = perfcurve(y_test ,scores,1);
% % Plot the ROC curve.
% figure(201);
% plot(X,Y)
% xlabel('False positive rate') 
% ylabel('True positive rate')
% title('ROC for Classification by SVM')
% legend(strcat('AUC=',num2str(AUC)))
% set(gca,'fontsize',16)
% grid on



function [CompactSVMModel,accuracy0,sensitivity0,specificity0,precision0,gmean0,f1score0,AUC,y_test,yfit,score]= SVM_classifier(X_train, y_train, X_test, y_test)
CVSVMModel = fitcsvm(X_train,y_train,'Holdout',0.1);
CompactSVMModel = CVSVMModel.Trained{1}; % Extract trained, compact classifier
[yfit,scores] = predict(CompactSVMModel,X_test);
[accuracy0,sensitivity0,specificity0,precision0,gmean0,f1score0]=prediction_performance(y_test , yfit);

%% Compute the ROC curve
score=scores(:,2);
[X,Y,T,AUC] = perfcurve(y_test ,score,1);
% Plot the ROC curve.
figure(201);
plot(X,Y)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification by SVM')
legend(strcat('AUC=',num2str(AUC)))
set(gca,'fontsize',16)
grid on


function [Mdl_Multi_Class,accuracy0,y_test,yfit,scores]= Multi_Class_SVM_classifier(X_train, y_train, X_test, y_test)

t = templateSVM('Standardize',1,'KernelFunction','gaussian');
% t = templateSVM('KernelFunction','rbf');%,'Alpha',1,'KernelScale',2);

Name_Classes=unique(y_train)';
%% Train an ECOC multiclass model using the default options.

Mdl_Multi_Class = fitcecoc(X_train,y_train,'Learners',t,'FitPosterior',0,'ClassNames',Name_Classes,'Verbose',0);

[yfit,scores] = predict(Mdl_Multi_Class,X_test);
[accuracy0]=Multi_Class_prediction_performance(y_test , yfit);

% %% Compute the ROC curve
% score=scores(:,2);
% [X,Y,T,AUC] = perfcurve(y_test ,scores,1);
% % Plot the ROC curve.
% figure(201);
% plot(X,Y)
% xlabel('False positive rate') 
% ylabel('True positive rate')
% title('ROC for Classification by SVM')
% legend(strcat('AUC=',num2str(AUC)))
% set(gca,'fontsize',16)
% grid on



function A=NO_T(B)
[M,N]=size(B);
Mh=M/2;
A=B(:,1:end-1); 
