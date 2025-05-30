function [fPWM_train, fPWM_test]=QuPWMs_features(m, X_train, y_train, X_test,k,M,mu,sigma )

fprintf('\n-->  Generate QuPWM features of order %d ',m)
fprintf('\n-->  Quantization ')

%% get the normal Distribution N(mu, sigma)

        [Levels, Level_intervals]=Set_levels_Sigma(k,M,mu,sigma);

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
% test
        name_features = fieldnames(mPWM_feature_train.C1);   %% list of QuPWM features 
        name_features=name_features(find(cellfun(@isempty, strfind(name_features,'_size')))) % remove the size related attribures
        Selected_features=[2 6 9];  1:max(size(name_features));%                        %% Select amoung the defined features in <name_features>
        mPWM_features=mPWM_features+1;
        %% Select the Training and Testing features
        
        [fPWM_train,mPWM_type,fPWM_sizes]=Get_Slected_PWM_features(mPWM_feature_train,name_features,Selected_features);
        [fPWM_test]=Get_Slected_PWM_features(mPWM_feature_test,name_features,Selected_features);
        FG_time=floor(toc);