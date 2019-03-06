
function PWM4Ds=Multi_PWM4D_mers(Qk_Mer,y_train)

[Na, Nb,NMers]=size(Qk_Mer);

for Kmer=1:NMers

    Q_input=Qk_Mer(:,:,Kmer);
    %% Build the PWM matrices
    PWM4Ds(:,:,:,Kmer)=Multi_PWM3D(Q_input,y_train);
end

d=1;