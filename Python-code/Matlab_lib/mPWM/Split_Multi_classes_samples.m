function [mu,sigma]=Split_Multi_classes_samples(X_MC,y_MC)


Classes=unique(y_MC)';
for k=1:max(size(Classes))
    idx=find(y_MC==Classes(k));
    X=X_MC(idx,:);
    y=y_MC(idx);
    
%% Get the statistical properties of the Dataset
    X_pd = fitdist(X(:),'Normal');
    
%% GEt the noral distibution for each subject
    sigma0(k)=X_pd.sigma;         mu0(k)=X_pd.mu;                  
        
end  

%% 

%% Build the table data
colnames={'mu','sigma'};
data=[mu0', sigma0'];
Data_pd= array2table(data, 'VariableNames',colnames);

mu=mean(mu0); 
sigma=min(sigma0); %sigma=mean(sigma0); 

d=1;
    