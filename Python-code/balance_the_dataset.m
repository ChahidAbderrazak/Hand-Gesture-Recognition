Negative_sample_ratio_TS=1;
idxp=find(y==1);    Np=max(size(idxp));
idxn=find(y==0);    Nn=max(size(idxn));
N=Negative_sample_ratio_TS*min(Nn,Np);
s = RandStream('mlfg6331_64'); Rndm_idx=randsample(s,Nn,Nn,false);
idxn2=idxn(Rndm_idx(1:N));
idx=[idxp ;idxn2];
X2=X(idx,:);
y2=y(idx,:);
