
function  Selected_features_op=Scan_optimal_QuPWM_features(type_clf,mPWM_feature_train, y_train, mPWM_feature_test, y_test)

    name_features = fieldnames(mPWM_feature_train.C1);   %% list of QuPWM features 
    name_features=name_features(find(cellfun(@isempty, strfind(name_features,'_size')))); % remove the size related attribures

        
        
    mPWM_features=0; cnt0=0; Acc_max=0;
    Z=max(size(name_features));
    %% list the existing feaetures
    Existing_features = 1:Z;
    %% ###########  Try all PWM-based features  combinaisons with  ###########################

    for sz_combinaison=1:Z
        
        mPWM_features=mPWM_features+1;
        new_combinaison = nchoosek(Existing_features,uint16(sz_combinaison));

        %% Build the features matrix for classification 
        for comb=1:size(new_combinaison,1)

            mPWM_features=comb;
            Selected_features=new_combinaison(comb,:); %Existing_features; % 

            [fPWM_train,mPWM_type, kMes_features]=Get_Slected_PWM_features(mPWM_feature_train, name_features, Selected_features);
            [fPWM_test]=Get_Slected_PWM_features(mPWM_feature_test, name_features, Selected_features);
            
            %% ###########  Perform the MultiLabels classification   ###########################
            eval(['[Mdl.LR',num2str(mPWM_features),',Accuracy_LR(mPWM_features),ytrue,yfit_LR1]=Classify_Multi_Class_Data(type_clf, fPWM_train, y_train, fPWM_test, y_test);'])

            %% get the performance
            cnt0=cnt0+1; X=mPWM_feature_train;
            accuracy=Accuracy_LR(mPWM_features);
            feature_name{cnt0}=mPWM_type;
            V_com(Selected_features)=1; Hex_com(cnt0)=bi2de(V_com);
            Output_results(cnt0,:)=[sz_combinaison, Hex_com(cnt0),size(X,2), accuracy];
            
            Classifier{cnt0}=type_clf;
            
            fprintf('Acc= %f   -   Acc_max=%f \n', accuracy, Acc_max)
            
            
            if Acc_max<accuracy  
                Combinaison_op=mPWM_type;
                Selected_features_op=Selected_features
                Hex_com_op=Hex_com(cnt0);
                Acc_max=accuracy;
                Output_results
            end
        end
    end
end