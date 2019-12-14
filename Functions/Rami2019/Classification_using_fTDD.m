
% [X_train,y_train, X_test,y_test]

x=X_train(1,:)';
steps=0;
winsize=size(x,1)
wininc=1;
fTDD = getfTDDfeat(x,steps,winsize,wininc);

d=1;
