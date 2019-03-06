function [X_train,y_train, X_test,y_test]=Split_Training_Testing_sets(X_table,list_Subjets,list_Gesture,list_Trials_TR,list_Trials_TS )

%% Positive class
Idxp=Get_the_Class_Samples(list_Subjets,list_Gesture,list_Trials_TR,X_table);
X_train=X_table.sEMG(Idxp,:);   y_train=X_table.Gesture(Idxp,:);


%% Negative  class
Idxn=Get_the_Class_Samples(list_Subjets,list_Gesture,list_Trials_TS,X_table);
X_test=X_table.sEMG(Idxn,:);    y_test=X_table.Gesture(Idxn,:);

d=1;
end


function Idx=Get_the_Class_Samples(list_Subjets,list_Gesture,list_Trials,X_table)

Idx=[];
for S=list_Subjets
    idxS=find(X_table.Subject==S);
    
    if min(size(idxS))>0

        for T=list_Trials

            idxT=find(X_table.Trial(idxS)==T);

            if min(size(idxT))>0

                for C=list_Gesture

                   idxC=find(X_table.Gesture(idxS(idxT))==C);

                   if min(size(idxC))>0
                       Idx_new=idxS(idxT(idxC));
                       Idx=[Idx;Idx_new];
                   end

                end



            end
            
            d=1;

        end

    end
end

Idx=unique(Idx);
data=X_table(Idx,:);
d=1;

end