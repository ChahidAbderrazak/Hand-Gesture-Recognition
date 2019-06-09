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
        
    
    
    %% Automated
    nbins=500;
    if k==1
        [Count(k,:),centers] = hist(X(:),nbins);
        


    else
        [Count(k,:)]= hist(X(:),centers);
    
    end
%     
%     lgnd{k}=strcat('Class',num2str(k));
%      
%     figure(85);
%         plot(centers,Count(k,:));
%         legend(lgnd)
%         hold on

    %% disctruubiton per positin 
%         for po=1:size(X,2)
%             
%             [pos_distr(:,po),centers] = hist(X(:,po),centers);  
%            
%         end
%         
%         Count_pos(:,:,k)=pos_distr;
%          
%          
%         figure(86);subplot(4,2,k);plot(centers,pos_distr)
%         xlim([-0.1,0.1])
%         
%         figure(87); plot(centers,mean(pos_distr'));xlim([-0.1,0.1]);hold on
%         
        

            
    
    
end  

%% 

%% Build the table data
colnames={'mu','sigma'};
data=[mu0', sigma0'];

mu=mean(mu0); 
sigma=min(sigma0);% sigma=mean(sigma0); 
d=1;
    
end

function [Levels, Level_intervals]=Automated_quantization(M, X,Y)
    nbins=100;
    Levels= 1:M;

  %% Get blaced dataset
    classes=unique(Y);
    indp=find(Y==classes(1)); series1=X(indp,:); series1=series1(:);
    indn=find(Y~=classes(1)); series2=X(indn,:); series2=series2(:); 
    [Count1,centers] = hist(series1,nbins);
    [Count2] = hist(series2,centers);
    diff_count=abs(Count1-Count2);
    
    sigma=min([std(series1),std(series2)]);    mu=mean([mean(series1),mean(series2)]);

    %% Metod 1
    Level_intervals=linspace(mu-3*sigma, mu+3*sigma, M-1);
    
    %% method 2  (toe b improved later if needed)
%     Level_intervals= Auto_split(M,centers,diff_count);
    
   d=1;
    
%     figure
%     plot(centers,Count1);
%     hold on
%     plot(centers,Count2);
%     hold on
%     plot(centers,diff_count);
%   
%     hold off
%     legend('First Series','Second Series') % add legend
%     
%      
%     figure;histogram(series1,nbins);hold on;histogram(series2,nbins)
%     legend('First Series','Second Series') % add legend
% 
%     
% 
%     figure
%     plot(centers,DataSum);
%     hold on
%     plot(centers,Count2);
%     hold on
%     plot(centers,abs(DataSum-Count2));
%     hold off
%     legend('First Series','Second Series') % add legend
    
end

function Level_intervals= Auto_split(M,centers,diff_count)


     Dmin=min(diff_count);     Dmax=max(diff_count);
     mu=mean(diff_count);
     Emin=mu-0.8*(mu-Dmin);
     Emax=mu+0.8*(Dmax-mu);
     
     idx=find(Emin<=diff_count);% & diff_count<=Emax);

%      plot(centers(idx),diff_count(idx))


%% get the most varying area
     diff_count=diff_count(idx);
     centers=centers(idx);
     Area_total=sum(diff_count);

     
     for tol=0.8:0.05:0.95
        Step_area=tol*Area_total/(M);

        area=0;
        cnt=0;
        for k=1:size(centers,2)

            area=area+diff_count(k);

            if area>Step_area 
               cnt=cnt+1;
                Level_intervals0(cnt)=centers(k);
                area_cnt(cnt)=area; area=0;

            end
        end

        if size(Level_intervals0,2)==M
            break;
        end
     end
     
     d=1;
     
     for k=1:size(Level_intervals0,2)-1
         
         Level_intervals(k)=mean(Level_intervals0(k:k+1));
         
     end
     
     
     d=1;
end