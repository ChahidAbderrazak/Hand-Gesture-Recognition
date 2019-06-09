function [Levels, Level_intervals]=Set_levels_Sigma(k,M,mu,sigma)

Levels= 1:M;
% N=M-1;
% VECTOR=[-floor(N/2): floor(N/2)];
% Level_intervals= mu+k*sigma*VECTOR; 
Level_intervals=linspace(mu-3*sigma, mu+3*sigma, M-1);

d=1;