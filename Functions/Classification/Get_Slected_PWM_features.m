function [fPWM,mPWM_type,kMes_features]=Get_Slected_PWM_features(mPWM_feature,name_features,Selected_features)  
%         list_all_features = fieldnames(mPWM_feature.C1);   %% list of QuPWM features 
%         list_all_features=list_all_features(find(cellfun(@isempty, strfind(list_all_features,'_size'))))
%         name_features=list_all_features;
        
       combined_features='';Combine_features_names='';
       fPWM=[]; fPWM_sizes=0; 
       cnt=1;kMes_features(cnt)=1;
            
       if Selected_features==-1 ; Selected_features=1:max(size(name_features)); end

    for z=Selected_features
      combined_features=strcat(combined_features,'-',name_features{z});
      
      feature_n=char(name_features(z));
      Combine_features_names=strcat(Combine_features_names,',',feature_n);
      eval(['fPWM_sizes(z+1)=mPWM_feature.C1.',name_features{z},'_size;']);

        for C=string(fieldnames(mPWM_feature)')
            eval(strcat('fPWM=[fPWM mPWM_feature.',C,'.',feature_n,'];'));
        end
        
        cnt=cnt+1;kMes_features(cnt)=size(fPWM,2)+1;
    end

    mPWM_type=strcat('[',combined_features(2:end),']');
        
end