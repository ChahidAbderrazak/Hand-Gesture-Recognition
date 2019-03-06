%% this function projects the binary patternn or the positive and negative data stored in 
% ACGT_pattern [pos_data; neg_data] to reduced dimension features 2*Nb_Layers

function [PWM3D]=General_PWM3Ds_matrices(Q1_train,y_train)

global Ratio_PWM
Ratio_PWM=1;

[Ma, Na,Nb_Layers]=size(Q1_train);
Mpwm=floor(Ma*Ratio_PWM);
slice_PWM=randi( Ma, [1 Mpwm]);

for k=1:Nb_Layers
    
    if k==Nb_Layers
        target=1;
    else
        target=0;
     end
    
    s=(k-1)*2 +1: k*2+target;
   
    featuresK_pos=Q1_train(slice_PWM,:,k);
    featuresK_neg=neg_pattern(slice_PWM,:,k);
    [PWKp(k,:), PWKn(k,:)]=GPWM_matrix_generator(featuresK_pos,featuresK_neg);
end


end
 