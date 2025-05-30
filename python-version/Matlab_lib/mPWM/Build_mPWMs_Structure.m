%% Build the different mPWMS with respoect to all classes y using sequences quantized using 
% the Level_intervals

function  mPWM_structure=Build_mPWMs_Structure(m, Q_sequences, y, Level_intervals)  
%% The previously used Quantization 
Levels=1:max(size(Level_intervals))+1;%min(min(Q_sequences));

 %% BUILD mPWM Matrices
Nl=max(size(Levels));
q=unique(Q_sequences)';
Nq=max(size(q));
       
Classes=unique(y)';
if size(Classes,2)==1; Classes=Classes';end
if size(Levels,2)==1; Levels=Levels';end
if size(Level_intervals,2)==1; Level_intervals=Level_intervals';end

 

%% Initialize the mPWM_structure output
eval(['mPWM_structure.m=',num2str(m),';'])
mPWM_structure.Classes=Classes;
mPWM_structure.Levels=Levels;
mPWM_structure.Level_intervals=Level_intervals;


  
if Nl== Nq

    cnt=1;
    
    
    for C=Classes

        Idx=find(y==C) ;
        
        Q_input_Class=Q_sequences(Idx,:);

        %% Build the different kmers
        fprintf('\n-->  Building mPWM matrices for the class %d ',C)

        [mPWM]=Extract_kMiers(Q_input_Class,Levels,m);
                   
        eval(['mPWM_structure.C',num2str(C),'=mPWM ;'])

        cnt=cnt+1;
    end

  
else
    
        Levels
        q
        fprintf('The qi in Levels in %d are different then the levels existing in  Q_sequences %d \n',Levels, q)
        dome
        
end
    
    
end

function [mPWM_structure]=Extract_kMiers(Input_sequence,Levels,m)

    Input_sequence = double(Input_sequence); 
    [M, N]=size(Input_sequence);
    N_levels=size(Levels,2);
    % Assign to each level a letter
    Seq_letter=char([65:90 97:122  char(194:194+N_levels-52) ]); N_letters=size(Seq_letter,2); %   or Seq_letter='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'    
    % start the scanning
    if m>1
        Input_sequence=[Input_sequence 0*Input_sequence(:,end-(m-2):end)-1];
    end

    eval(['mPWM_structure.m=',num2str(m),';'])

     for k=1:m
        eval(['mPWM_structure.motif',num2str(k),'= permn(Levels,k);'])
        eval([' Nmotif=max(size(mPWM_structure.motif',num2str(k),'));']);

        %% Create empthy mPWM matrices for each kMers
        eval(['mPWM_structure.m',num2str(k),'PWM= zeros(Nmotif,N);'])

     end


    for k=1:m

        eval([' Nmotif=max(size(mPWM_structure.motif',num2str(k),'));']);
        PWM_matrix= zeros(Nmotif, N);
        eval(['motifs=mPWM_structure.motif',num2str(k),';'])

        for sample=1:M
            for n=1:N
               motif_n=Input_sequence(sample,n:n+k-1);   
               idx_motif=find_motif(motif_n,motifs);

               % Update the appropiate PWM_matrix
                PWM_matrix(idx_motif,n)= PWM_matrix(idx_motif,n)+1;

            end
        end

        %% normalize the PWM matrix
        [PWM_matrix]=normalize_array_rowwise(PWM_matrix);

        eval(['mPWM_structure.m',num2str(k),'PWM=PWM_matrix ;'])

        %% Display msg
         fprintf('\n\t-->  Building m%dPWM matrices ',k)

    end

    d=1;
end


function idx=find_motif(X,motif)
    m=size(X,2);
    idx=find(X(1)==motif(:,1));
    for k=2:m
        idx2=find(X(k)==motif(:,k));
        idx=intersect(idx,idx2);
    end

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