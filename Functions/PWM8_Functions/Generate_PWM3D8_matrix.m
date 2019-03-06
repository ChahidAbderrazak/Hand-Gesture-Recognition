
% function [PWM4Ds_Mer1, PWM4Ds_Mer2]=Generate_PWM3D8_matrix(Q_train,y_train)
function [PWM4Ds_Mer1]=Generate_PWM3D8_matrix(Q_train,y_train)

global Levels

%      PWM4Ds=Multi_PWM4D_mers(Q_train,y_train);

    %% Mono-Mers  Position Weight Matrix-BASED FEATURES
    [Q1_Mer,name_Mer1] = Extract_Miers1(Q_train,Levels); 
    
     % Geneate the coresponding PWMs for eack 1-mer pattern
     PWM4Ds_Mer1=Multi_PWM4D_mers(Q1_Mer,y_train);[Na, Nb,NClasses,NMers]=size(PWM4Ds_Mer1);
   
%     %% Di-Mers  Position Weight Matrix-BASED FEATURES
%     [Q2_Mer, name_Mer2] = Extract_Miers2(Q_train,Levels);
%     
%      % Geneate the coresponding PWMs for eack 1-mer pattern
%      PWM4Ds_Mer2=Multi_PWM4D_mers(Q2_Mer,y_train);    

end



