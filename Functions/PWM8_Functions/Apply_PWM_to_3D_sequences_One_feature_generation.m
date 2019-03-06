%% the Input_sequence,neg_pattern should be binary patterns

function [fPWM]=Apply_PWM_to_3D_sequences_One_feature_generation(Input_sequence, PWM)
%% PWM Matrix normalization option 
[Mp, Np,Nb_Layers]=size(Input_sequence);  
 [Na, Nb,NClasses,NMers]=size(PWM);
norm_rows=1;
PWM=normalize_PWD_matrix(Mp,norm_rows, PWM);

%% Start feature generation 

fPWM=[];
for k=1:Nb_Layers

    s=(k-1)*2 +1: k*2;
    patternK_pos=Input_sequence(:,:,k);
    fPWM_new=GPWM_feature_generator(patternK_pos,PWM);  
    fPWM=[fPWM fPWM_new];
end


d=1;
function [features_PWK]=GPWM_feature_generator(patternK_pos,PWM)
[M,N]=size(PWM);
if M==2
    features_PWK=patternK_pos*PWM(1,:)';

elseif N>2
    features_PWK=patternK_pos*PWM';
end

d=1;
