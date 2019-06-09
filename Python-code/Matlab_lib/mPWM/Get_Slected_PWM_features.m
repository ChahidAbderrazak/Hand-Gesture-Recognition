function [fPWM, mPWM_type, kMes_features]=Get_Slected_PWM_features(mPWM_feature,name_features,Selected_features)  


       combined_features='';Combine_features_names='';
       fPWM=[];
       cnt=1;kMes_features(cnt)=1;
       
       if Selected_features==-1
           
            name_features = fieldnames(mPWM_feature.C1);   %% list of QuPWM features 
            name_features=name_features(find(cellfun(@isempty, strfind(name_features,'_size')))) % remove the size related attribures
            Selected_features=1:max(size(name_features));% 
            fprintf('\n\t-->    QuPWM-based selection : ALL FEATURES  ')
       else
           
           fprintf('\n\t-->    QuPWM-based selection : %d  ', Selected_features)

       end
    
       
       
    for z=Selected_features
      combined_features=strcat(combined_features,'-',name_features{z});
      feature_n=char(name_features(z));
      Combine_features_names=strcat(Combine_features_names,',',feature_n);


        for C=string(fieldnames(mPWM_feature)')
   
            eval(strcat('fPWM=[fPWM mPWM_feature.',C,'.',feature_n,'];'));
        end
    end

    mPWM_type=strcat('[',combined_features(2:end),']');
    cnt=cnt+1;kMes_features(cnt)=size(fPWM,2)+1;
        
end