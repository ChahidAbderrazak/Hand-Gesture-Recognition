function PWM_matrix= Generate_PWM_matrix(Q_train, Levels)
Nl=max(size(Levels));
q=unique(Q_train)';
Nq=max(size(q));


if Nl~= Nq
%     Levels
%     q
%     fprintf('The qi in Levels in %d are different then the levels existing in  Q_train %d \n',Levels, q)
%     dome
end
PWM_matrix= zeros(Nq, size(Q_train,2));

for k=1:Nq%Nl
    for i=1:size(Q_train, 2)
        PWM_matrix(k,i)= sum(Q_train(:, i) == q(k))/size(Q_train,1);
    end
end


end