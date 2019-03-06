function  PWMs=Multi_PWM3D(Q_sequences,y)
global Levels
Classes=unique(y)';
cnt=1;
for C=Classes

    Idx=find(y==C);
    Q_input=Q_sequences(Idx,:);
    %% Build the PWM matrices
    PWMs(:,:,cnt)= Generate_PWM_matrix(Q_input, Levels);
    cnt=cnt+1;
end

  
    
end

  