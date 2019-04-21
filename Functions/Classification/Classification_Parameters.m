
%% ####################   Classification paramter ############################
% This script sets the classification paramter to classify X,y data

%% ###########################################################################
%  Author:
%  Abderrazak Chahid (abderrazak.chahid@gmail.com)
% Done: Jan,  2019
%
%% ###########################################################################

% K=5;
% CV_type='KFold';%'LOO';%                                     strcat(num2str(K),'-Folds_CV');%   
% type_clf='SVM';%'LR';%                        
% Normalization=0;
% EN_starplus=1;

%% ###########################################################################

%% Define the binary classification
% 
% list_Subjets=1;
% list_Trials=1;
% class_list=1:8;
% [X,y]=Define_the_Multi_Classes_data(Data,list_Subjets,list_Trials,class_list);

%% Define the Multi-Class classification
% class_p=1:4;class_n=5:8;
%  class_p=1;class_n=2;
% [Xp,yp, Xn, yn]=Define_the_Binary_Classes(Data, class_p, class_n,list_Subjets,list_Trials);
% Nn=size(Xn,1);  Np=size(Xp,1);  Ndata=min(Nn,Np);
% X=[Xp(1:Ndata,:);Xn(1:Ndata,:)];
% y=[yp(1:Ndata,:);yn(1:Ndata,:) ];

%% Multi clasees
X=Data.sEMG;
y=Data.Gesture;
idx=find(y<=8);
y=y(idx);
X=X(idx,:);


%% Normalization 
% X0=X;
% X=X*10^12;

if Normalization==1
    X=Scale_down_to_unit(X);
else
    
end

 %% Random sampling the input  data
%     [X,shuffle_index]=Shuffle_data(X);y=y(shuffle_index);

Data_info = strsplit(data_Source,'\');
%% Display

d_clf='--> Hand Gesture Detection using EMG 2019  :' ;
d_data1=string(strcat('- Classification: CV- ',{''},CV_type,{''},', K=',num2str(K) ));
d_data_info=string(strcat('- Dataset:  Trials: ',filename ,',  Subject: ',Data_info(end-1), ',  Classes: ',Data_info(end)  ));

% d_data2=string(strcat('- Sampling:  L=',num2str(L_max),', Frame Step=',num2str(Frame_Step),', Norm=',num2str(Normalization)));

fprintf('\n######################################################\n ');
fprintf(' %s \n %s \n %s ',d_clf,d_data1,d_data_info);
fprintf('\n######################################################\n \n\n');




















%   %% Splitting the data 80/20 (training/testing)  data
%       [Seq_pos,Seq_neg,yp,yn]=Split_Features_Pos_Neg(X,y);
% 
%     [Mp, Np] = size(Seq_pos); [Mn, Nn] = size(Seq_neg); Mmin = min(Mp,Mn); 
%     TR = floor(0.8*Mmin); % TR represents the size of the trainign data
%     TR_X_pos = Seq_pos(1:TR,:);  TR_y_pos = yp(1:TR);     
%     TR_X_neg = Seq_neg(1:TR,:);  TR_y_neg = yn(1:TR);   
%     
%     TS_X_pos = Seq_pos(TR+1:Mmin,:);   TS_y_pos = yp(TR+1:Mmin);  
%     TS_X_neg = Seq_neg(TR+1:Mmin,:);   TS_y_neg = yn(TR+1:Mmin); 
%     
%     
%     %% Get the training and testing data
%     Xtrain= [TR_X_pos; TR_X_neg];        ytrain=[TR_y_pos;TR_y_neg];
%     Xtest= [TS_X_pos; TS_X_neg];        ytest=[TS_y_pos;TS_y_neg];

    
    
    
    