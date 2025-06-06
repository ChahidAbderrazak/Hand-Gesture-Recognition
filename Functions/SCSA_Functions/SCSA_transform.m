
%%
    %**********************************************************************
    %                           SCSA  Function                            *
    %**********************************************************************

 % Author : Professor Taous_Meriem Laleg . EMAN Group KAUST 
 % taousmeriem.laleg@kaust.edu.sa 
 
 %% Description
 % Script compute the SCSA algorithm  on a 1D signal of size N  with gm=0.5
 % Done: Jun,  2013

%% input parapeters
% y   :  Noisy  signal
% fs : sampling frequency 
% h  : SCSA parameter
% gm  : SCSA parameter
 
%% output parapeters
% yscsa   :  denoised  signal
% Nh : Number of Eigen values chosen
% SC_h : The decomposed Matrix of the Schrödinger problem


function [h,Ys,Nh,Eigen_Spectrum, ProbaDNA,Sum_prod_basis]= SCSA_transform(Y,h0,fs,gm)

[N M]=size(Y);
stop=0;
Eigen_Spectrum=zeros([N M]);
parfor i=1:N
    y=Y(i,:);
    [h1, yscsa,Nh0,Eigen_Spectrum0, psinnor]= SCSA1D_vector(y, fs,h0,gm);
%     figure; plot(y,'r'); hold on ; plot(yscsa,'b'); hold off ; 
    Nh(i)=Nh0;h(i)=h1;
    Ys(i,:)=yscsa;
%     Proba=psinnor.^2'
    Proba=normalize_matrix(1,(psinnor.^2)');
    Sum_prod_basis(i,:)= Eigen_Spectrum0'*psinnor';
    I=Eigen_Spectrum0*0+1;
    ProbaDNA(i,:)=I'*Proba;
    Eigen_Spectrum00=Eigen_Spectrum0(1:Nh0)';
%     Eigen_Spectrum(i,1:Nh0)=[Eigen_Spectrum00];
     Eigen_Spectrum(i,:)=Eigen_Spectrum00;

    
end


Eigen_Spectrum( :, all(~Eigen_Spectrum,1) ) = [];




d=1;

function [h, yscsa,Nh,Neg_lamda,psinnor]= SCSA1D_vector(y,fs,h,gm)

Lcl = (1/(2*sqrt(pi)))*(gamma(gm+1)/gamma(gm+(3/2)));
N=max(size(y));
%% remove the negative part
if min(y)<0
    y_scsa = y - min(y);
else
   y_scsa = y; 
end

%% Build Delta metrix for the SC_hSA
feh = 2*pi/N;
D=delta(N,fs,feh);

%% start the SC_hSA
Y = diag(y_scsa);
SC_h = -h*h*D-Y; % The Schrodinger operaor

% = = = = = = Begin : The eigenvalues and eigenfunctions
[psi,lamda] = eig(SC_h); % All eigenvalues and associated eigenfunction of the schrodinger operator
% Choosing  eigenvalues
All_lamda = diag(lamda);
ind = find(All_lamda<0);


%  negative eigenvalues
Neg_lamda=0*y';
Neg_lamda(ind) = All_lamda(ind);
kappa = diag((abs(Neg_lamda)).^gm); 
Nh = size(kappa,1); %%#ok<NASGU> % number of negative eigenvalues

psinnor=0*y;

if Nh~=0
    
% Associated eigenfunction and normalization

% psin = psi(:,ind(:,1)); % The associated eigenfunction of the negarive eigenvalues
psin=psi;
I = simp(psin.^2,fs); % Normalization of the eigenfunction 
psinnor = psin./sqrt(I);  % The L^2 normalized eigenfunction 


%yscsa =4*h*sum((psinnor.^2)*kappa,2); % The 1D SC_hSA formula
yscsa1 =((h/Lcl)*sum((psinnor.^2)*kappa,2)).^(2/(1+2*gm));
else
  yscsa1=0*y;
  yscsa1=yscsa1-10*abs(max(y));
  Neg_lamda=0;

  disp('There are no negative eigenvalues. Please change the SCSA parameters: h, gm ')
end


if size(y_scsa) ~= size(yscsa1)
yscsa1 = yscsa1';
end
 
 %% add the removed negative part
 
 if min(y)<0
   yscsa = yscsa1 + min(y);
 else
     yscsa = yscsa1;
 end
  
% lgnd=string(J(1:k))
% figure;
% plot(psin(:,1:k))
% legend(lgnd)
% set(gca,'fontsize',16)
% 




    %**********************************************************************
    %*********              Numerical integration                 *********
    %**********************************************************************

    % Author: Taous Meriem Laleg

    function y = simp(f,dx);
    %  This function returns the numerical integration of a function f
    %  using the Simpson method

    n=length(f);
    I(1)=1/3*f(1)*dx;
    I(2)=1/3*(f(1)+f(2))*dx;

    for i=3:n
        if(mod(i,2)==0)
            I(i)=I(i-1)+(1/3*f(i)+1/3*f(i-1))*dx;
        else
            I(i)=I(i-1)+(1/3*f(i)+f(i-1))*dx;
        end
    end
    y=I(n);
    

    %**********************************************************************
    %*********             Delata Metrix discretization           *********
    %**********************************************************************
    
    
%Author: Zineb Kaisserli

function [Dx]=delta(n,fex,feh)
    ex = kron([(n-1):-1:1],ones(n,1));
    if mod(n,2)==0
        dx = -pi^2/(3*feh^2)-(1/6)*ones(n,1);
        test_bx = -(-1).^ex*(0.5)./(sin(ex*feh*0.5).^2);
        test_tx =  -(-1).^(-ex)*(0.5)./(sin((-ex)*feh*0.5).^2);
    else
        dx = -pi^2/(3*feh^2)-(1/12)*ones(n,1);
        test_bx = -0.5*((-1).^ex).*cot(ex*feh*0.5)./(sin(ex*feh*0.5));
        test_tx = -0.5*((-1).^(-ex)).*cot((-ex)*feh*0.5)./(sin((-ex)*feh*0.5));
    end
    Ex = full(spdiags([test_bx dx test_tx],[-(n-1):0 (n-1):-1:1],n,n));
    
    Dx=(feh/fex)^2*Ex;
    
    
    
    
function A_N=normalize_matrix(row, A)

[K,R]=size(A);
A_N=A./sum(A')';
% if row==1
%     s_norm=sum(A');
%     for i=1:K
%         if i> K
%            d=1; 
%         end
%     A_N(i,:)=A(i,:)/s_norm(i);
%     end
% else
%     
% end
d=1;