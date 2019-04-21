function [fPWM,mPWM_type]=Get_Slected_PWM_features(mPWM_feature,name_features,Selected_features)  

       combined_features='';Combine_features_names='';

       fPWM=[];  
    
    for z=Selected_features
      combined_features=strcat(combined_features,'-',name_features{z});
      feature_n=char(name_features(z));
      Combine_features_names=strcat(Combine_features_names,',',feature_n);


        for C=string(fieldnames(mPWM_feature)')
   
            eval(strcat('fPWM=[fPWM mPWM_feature.',C,'.',feature_n,'];'));
        end
    end

    mPWM_type=strcat('[',combined_features(2:end),']');
        
end