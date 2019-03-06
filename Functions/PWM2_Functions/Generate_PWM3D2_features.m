function fPWM3D2_features= Generate_PWM3D2_features(Q_input, PWM3Ds)
 fPWM3D2_features=[];
[Na,Nb,NClass]=size(PWM3Ds);
for L=1:Na
     for k=1:NClass
        PWM3Ds_k=zeros([Na,Nb]);
        PWM3Ds_k(L,:)=PWM3Ds(L,:,k);
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
        fPWM3D2_features=[fPWM3D2_features fPWM3D];

     end
     
end


d=1;
