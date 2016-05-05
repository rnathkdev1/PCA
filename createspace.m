function [Psy,Phi,S,V,ui,Omega,reconstX,eigenval,numclass]=createspace(Gamma,class)
%% Calculating the new space
numclass=length(unique(class));

Psy=mean(Gamma,2);
Psy=repmat(Psy,[1, size(Gamma,2)]);

Phi=(Gamma-Psy);
S=Phi'*Phi;

[V,eigenval]=eig(S);%Each column is an eigenvector where in increasing order of eigenvalue
eigenval=unique(eigenval);
eigenval(eigenval==0)=[];
ui=Phi*V; % Each column is the eigenvector of The Big One. (NXM)

%% Reconstruction of the dataset

Omega=ui'*Phi; % ak is the column of Omega
reconstX=Psy+ui*V';

end
