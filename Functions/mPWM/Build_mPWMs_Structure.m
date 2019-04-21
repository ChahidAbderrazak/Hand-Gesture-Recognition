%% Build the different mPWMS with respoect to all classes y using sequences quantized using 
% the Level_intervals

function  mPWM_structure=Build_mPWMs_Structure(m, Q_sequences, y, Level_intervals)  
%% The previously used Quantization 
Levels=1:max(size(Level_intervals))+1;

%% BUILD mPWM Matrices
Nl=max(size(Levels));
q=unique(Q_sequences)';
Nq=max(size(q));
       
Classes=unique(y)';
   
%% Initialize the mPWM_structure output
eval(['mPWM_structure.m=',num2str(m),';'])
mPWM_structure.Classes=Classes;
mPWM_structure.Levels=Levels;
mPWM_structure.Level_intervals=Level_intervals;

if Nl== Nq

    cnt=1;
    
    for C=Classes

        Idx=find(y==C);
        Q_input_Class=Q_sequences(Idx,:);

        %% Build the different kmers
        fprintf('\n-->  Building mPWM matrices for the class %d ',C)

        [mPWM]=Extract_kMiers(Q_input_Class,Levels,m);
                   
        eval(['mPWM_structure.C',num2str(C),'=mPWM ;'])

        cnt=cnt+1;
    end

  
else
    
    %     Levels
    %     q
    %     fprintf('The qi in Levels in %d are different then the levels existing in  Q_sequences %d \n',Levels, q)
    %     dome
        
end
    
    
end

