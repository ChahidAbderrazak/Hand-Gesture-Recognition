PCA_size=5; 

%% the datset before splitting 
X=rand(100,22);

%% Apply PCA for feature reduction 
% Now get the principal components.
coeff = pca(X);

% Take the coefficients and transform the input data into a PCA-data.
X_pca = X * coeff(:,1:PCA_size);

%% Classification using X_pca