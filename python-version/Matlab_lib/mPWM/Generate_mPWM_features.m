%% Generates  the different mPWMS features using the PWM matrices of <mPWM_structure>
function  [mPWM_features]=Generate_mPWM_features(mPWM_structure, Q_sequences)  
            
fprintf('\n-->  Generate  mPWM-based  features ')

Q_Levels=unique(Q_sequences)';

mfPWM=[];   fPWM2=[];   fPWM=[];

save('log.mat')
if norm(floor(Q_Levels)- floor(mPWM_structure.Levels))==0
    cnt=1;
 
        for C=mPWM_structure.Classes %% loops over the classes features

             fprintf('\n\t-->  Generate  mPWM-based  features up to m%dPWM_C%d \t',mPWM_structure.m,C)
             tic
             [PWM_features_k]= Generate_mPWM_features_per_Class(mPWM_structure,Q_sequences, C);
             toc
            %% Concatenate features
            eval(['mPWM_features.C',num2str(C),'=PWM_features_k ;']);
         end
else
    
        
        mPWM_structure.Levels
        fprintf('The number of  Levels in %d are different then the levels existing in  Q_sequences %d \n',Levels, mPWM_structure.Levels)
        dome
        
end
    

end



function [PWM_features]= Generate_mPWM_features_per_Class(mPWM_structure, Q_input,C)
%% Generate the mPWM features
[M, N]=size(Q_input);
             
%% normalize the PWM matrix
% [mPWM_C]=normalize_array_rowwise(mPWM_C);
% % [mPWM_C]=normalize_array_colmnwise(mPWM_C);

%% Get teh PWM for the Class <C>

save('log2.mat')
fprintf('w0');
eval(['mPWM_C=mPWM_structure.C',num2str(C),';']);
fprintf('w0');
for k=1:mPWM_structure.m
    fprintf('w%d',k);
eval(['motifs',num2str(k),'=mPWM_structure.C',num2str(C),'.motif',num2str(k),' ;']);
eval(['Mm',num2str(k),'=max(size(motifs',num2str(k),')) ;']);
eval(['m',num2str(k),'PWM=mPWM_structure.C',num2str(C),'.m',num2str(k),'PWM ;']);
% eval(['m',num2str(k),'PWM=normalize_array_colmnwise(mPWM_structure.C',num2str(C),'.m',num2str(k),'PWM) ;']);

%% feature matrix
eval(['m',num2str(k),'fPWM=zeros(size(Q_input,1), size(Q_input,2)-k+1);']);
eval(['p',num2str(k),'fPWM=zeros(size(Q_input,1), Mm',num2str(k),');']);

end
                
         
for sample=1:M
    %          sample
    Q_sample=[Q_input(sample,:) -1*ones(1,mPWM_structure.m)];

    for n=1:N

    %             n
       motif_n=Q_sample(n:n+mPWM_structure.m-1); 
       m1_idx=find(motif_n(1)==motifs1);
       
       %% Get the mPWM weights
       m1fPWM(sample,n)= m1PWM(m1_idx,n);
       p1fPWM(sample,m1_idx)= p1fPWM(sample,m1_idx) + m1PWM(m1_idx,n);

       if mPWM_structure.m>1 & n<=N-mPWM_structure.m+1
           
%% loop the deferent motifs
           for k=2:mPWM_structure.m
               
               switch k
                   
                   case 2  %% di-mers
                       
                           Idx_start=(m1_idx-1)*Mm1;
                           m2_idx=find(motif_n(2)==motifs2(Idx_start+1:Idx_start+Mm1,2));
                           m2_idx=m2_idx+Idx_start;
                           motif_found=motifs2(m2_idx,:);
                           motif_n;

                           if norm(motif_n(1:2)-motif_found)~=0
                              fprintf('error motifs found is wrong please check the code\n');
                               ed=a;
                           else

                               %% Get the mPWM weights
                                m2fPWM(sample,n)= m2PWM(m2_idx,n);
                                p2fPWM(sample,m2_idx)= p2fPWM(sample,m2_idx) + m2PWM(m2_idx,n);
                                d=1;
                           end
                
                   case 3  %% Tri-mers
                       
                           Idx_start=(m2_idx-1)*Mm1;
                           m3_idx=find(motif_n(3)==motifs3(Idx_start+1:Idx_start+Mm1,3));
                           m3_idx=m3_idx+Idx_start;
                           motif_found=motifs3(m3_idx,:);
                           motif_n;

                           if norm(motif_n(1:3)-motif_found)~=0
                              fprintf('error motifs found is wrong please check the code\n');
                               ed=a;
                           else

                               %% Get the mPWM weights
                                m3fPWM(sample,n)= m3PWM(m3_idx,n);
                                p3fPWM(sample,m3_idx)= p3fPWM(sample,m3_idx) + m3PWM(m3_idx,n);
                                d=1;
                           end
               end
               
%% ######################################################################################################### 





           end
       end
       
       
    end
end



%% Save the mPEMS features seperatly 
for k=1:mPWM_structure.m
eval(['PWM_features.m',num2str(k),'fPWM=m',num2str(k),'fPWM;']);
eval(['PWM_features.p',num2str(k),'fPWM=p',num2str(k),'fPWM;']);
eval(['PWM_features.m',num2str(k),'fPWM1=sum(m',num2str(k),'fPWM,2);']);

end
     
end



function [PWM_features,  mfPWM, fPWM2]= Generate_mPWM8_features_per_Class(mPWM_structure, Q_input,C)

%% Generate the mPWM features
[M, N]=size(Q_input);
             
%% normalize the PWM matrix
% [mPWM_C]=normalize_array_rowwise(mPWM_C);
% % [mPWM_C]=normalize_array_colmnwise(mPWM_C);

%% Get teh PWM for the Class <C>
eval(['mPWM_C=mPWM_structure.C',num2str(C),';']);

for k=1:mPWM_structure.m
eval(['motifs',num2str(k),'=mPWM_structure.C',num2str(C),'.motif',num2str(k),' ;']);
eval(['Mm',num2str(k),'=max(size(motifs',num2str(k),')) ;']);
eval(['m',num2str(k),'PWM=normalize_array_rowwise(mPWM_structure.C',num2str(C),'.m',num2str(k),'PWM) ;']);

%% feature matrix
eval(['m',num2str(k),'fPWM=zeros(size(Q_input,1), size(Q_input,2)-k+1,Mm',num2str(k),');']);


end
                
         
for sample=1:M
    %          sample
    Q_sample=[Q_input(sample,:) -1*ones(1,mPWM_structure.m)];

    for n=1:N

    %             n
       motif_n=Q_sample(n:n+mPWM_structure.m-1); 
       m1_idx=find(motif_n(1)==motifs1);
       
       %% Get the mPWM weights
       m1fPWM(sample,n)= m1PWM(m1_idx,n);


       if n<=N-mPWM_structure.m+1
           Idx_start=(m1_idx-1)*Mm1;
           m2_idx=find(motif_n(2)==motifs2(Idx_start+1:Idx_start+Mm1,2));
           m2_idx=m2_idx+Idx_start;
           motif_found=motifs2(m2_idx,:);
           motif_n;

           if norm(motif_n-motif_found)~=0
               d=1
           end
           %% Get the mPWM weights
           m2fPWM(sample,n,m2_idx)= m2PWM(m2_idx,n);

 

       end
    end
end

 
%% Convert the 3D feature matrix to 2D feature matrix
m1fPWM_flat = reshape(m1fPWM,[],size(m1fPWM,2)*size(m1fPWM,3),1);
m2fPWM_flat = reshape(m2fPWM,[],size(m2fPWM,2)*size(m2fPWM,3),1);

%% Concatenate feature matrix
mfPWM=[m1fPWM_flat m2fPWM_flat];
fPWM2=[ sum(m1fPWM_flat,2) sum(m2fPWM_flat,2)];



%% Save the mPEMS features seperatly 
for k=1:mPWM_structure.m
eval(['PWM_features=m',num2str(k),'fPWM;']);
end
     

end

    
            
function [mfPWM, fPWM2]= Generate_m1PWM_features(Q_input, mPWM_C)
[M, N]=size(Q_input);
 for s=1:M
    for n=1:N-(m-1)
        PWM_idx=Q_input(s,n);
        mfPWM(s,n)= mPWM_C(PWM_idx,n);
    end
 end
 
% sum all the probabilities to get the PWM
fPWM2=sum(mfPWM,2);
d=1;
end

            


 
function [output]= normalize_array_colmnwise(input)
[m,n]=size(input);
output = zeros(m,n);
for i=1:m
output(i,:)= input(i,:)/sum(input(i,:));
end
end

function [output]=normalize_array_rowwise(input)
[m,n]=size(input);
output = zeros(m,n);
for i=1:n
output(:,i)= input(:,i)/sum(input(:,i));
end
end

function idx_out=find_motif(X,motif)
m=size(X,2);
cnt=1;

idx=find(X(1)==motif(:,1));


% Motifs_indeces=
idx_out(cnt)=idx(1);
for k=2:m
    idx2=find(X(k)==motif(:,k));
    idx=intersect(idx,idx2);
end

d=1;

end
