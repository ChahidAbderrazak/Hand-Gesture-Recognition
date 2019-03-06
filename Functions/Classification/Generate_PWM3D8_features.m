
function fPWM= Generate_PWM3D8_features(Q_mer, PWM4D8)
global Levels

[Na, Nb,NClasses,NMers]=size(PWM4D8);
fPWM=[];

for kmer=1:3%NMers
    for C=1:NClasses
        PWM3D=PWM4D8(:,:,C,kmer);
        Qk_mer=Q_mer(:,:,kmer);
        fPWM_new=Apply_PWM_to_3D_sequences_One_feature_generation(Qk_mer, PWM3D);
        fPWM=[fPWM fPWM_new];
        d=1;
    end
end
end


function PWM=Get_PWM_classk(PWM4D8_Mer1)


        d=1;

end
