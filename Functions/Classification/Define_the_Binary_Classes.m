function [Xp,yp, Xn, yn]=Define_the_Binary_Classes(X_table, class_p, class_n,list_Subjets,list_Trials)

%% Positive class
Idxp=Get_the_Class_Samples(list_Subjets,list_Trials,class_p,X_table);
Xp=X_table.sEMG(Idxp,:); Np=size(Xp,1); yp=ones(Np,1);


%% Negative  class
Idxn=Get_the_Class_Samples(list_Subjets,list_Trials,class_n,X_table);
Xn=X_table.sEMG(Idxn,:); Nn=size(Xn,1); yn=zeros(Nn,1);

d=1;
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