function [X,y]=Define_the_Multi_Classes_data(X_table,list_Subjets,list_Trials,class_list)

%% Positive class
Idx=Get_the_Class_Samples(list_Subjets,list_Trials,class_list,X_table);
X=X_table.sEMG(Idx,:); y=X_table.Gesture(Idx,:);

end



function Idx=Get_the_Class_Samples(list_Subjets,list_Trials,class_list,X_table)

Idx=[];
for S=list_Subjets
    idxS=find(X_table.Subject==S);
    
    if min(size(idxS))>0

        for T=list_Trials

            idxT=find(X_table.Trial(idxS)==T);

            if min(size(idxT))>0

                for C=class_list

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