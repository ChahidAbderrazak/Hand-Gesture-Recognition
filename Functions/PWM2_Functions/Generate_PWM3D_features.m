 

function fPWM3D_features= Generate_PWM3D_features(Q_input, PWM3Ds)
 fPWM3D_features=[];
 
 for k=1:size(PWM3Ds,3)
     
    PWM3Ds_k=PWM3Ds(:,:,k);
    fPWM= zeros(size(Q_input,1), size(Q_input,2)); %f1 is the first feature of PWM

    % replace the integer values by its probability from VPM
    for i=1:size(Q_input,1)
        for j=1:size(Q_input,2)
            PWM_idx=Q_input(i,j);
            fPWM(i,j)= PWM3Ds_k(PWM_idx,j);
        end
    end
    % sum all the probabilities to get the PWM
    fPWM3D=sum(fPWM,2);
    fPWM3D_features=[fPWM3D_features fPWM3D];
    
 end

d=1;
end