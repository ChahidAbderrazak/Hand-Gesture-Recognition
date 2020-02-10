% function [accuracy,sensitivity,specificity,precision,gmean,f1score]=Multi_Class_prediction_performance(ytrue, yfit)
 function [accuracy]=Multi_Class_prediction_performance(ytrue, yfit)

% if size(ytrue,1)>=1
 
    C=confusionmat(ytrue, yfit);
%     sensitivity = C(2,2)/(C(2,1)+C(2,2))*100;
%     specificity = C(1,1)/(C(1,1)+C(1,2))*100;
%     precision = (C(2,2)/(C(2,2)+C(1,2)))*100;
    accuracy= sum(diag(C))/sum(sum(C))*100;

    
% else
%     
%     if ytrue==yfit
% 
%         accuracy= 100;
%         
%     else
%         accuracy= 0;
%     end
% 
% end

 d=1;
    