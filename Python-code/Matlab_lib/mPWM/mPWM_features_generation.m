function [fPWM_train,fPWM_test,mPWM_structure]=mPWM_features_generation(M,k,m, X_train,y_train, X_test)

%% Get the statistical properties of the data
[mu,sigma]=Split_Multi_classes_samples(X_train,y_train);


k0=1.65/M;
%% get the normal Distribution N(mu, sigma)
    [Levels, Level_intervals]=Set_levels_Sigma(k*k0,M,mu,sigma);

%% get the  Optimal quatizer 
% [Level_intervals,codebook] = lloyds(X_train(:),M);Levels=1:M;

%% Quantization
    Q_train= mapping_levels(X_train,Level_intervals, Levels);
    Q_test= mapping_levels(X_test, Level_intervals, Levels);
    
%% Build the nPWM matrices for different kMers
tic
mPWM_structure=Build_mPWMs_Structure(m, Q_train, y_train, Level_intervals);
PG_time=floor(toc);    
    %% Generate the Training features 
tic
fprintf('\n-->  Generate the Training features  ')
    [mPWM_feature_train]=Generate_mPWMs_features(mPWM_structure, Q_train);

    %% Generate the Testing features 
fprintf('\n-->  Generate the Testing features  ')
    [mPWM_feature_test]=Generate_mPWMs_features(mPWM_structure, Q_test);


%% ###########  Perform QuPWM-based  Feature selection      ###########################
fprintf('\n-->  Perform QuPWM-based  Feature selection  ')

name_features = fieldnames(mPWM_feature_train.C1)   %% list of QuPWM features  
Selected_features=[2 6 9];                          %% Select amoung the defined features in <name_features>
%% Select the Training and Testing features

[fPWM_train,mPWM_type]=Get_Slected_PWM_features(mPWM_feature_train,name_features,Selected_features);
[fPWM_test]=Get_Slected_PWM_features(mPWM_feature_test,name_features,Selected_features);

end
