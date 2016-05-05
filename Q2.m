clc; clear;
%% Creating dataset of Q2
Gamma=[-2 1 2 -3 4 1 0 3 0 2 1 1 2 3 -2 -3 2 1 0;
    1 2 -4 2 -4 2 5 2 2 1 -3 0 0 1 -2 1 1 -3 -2;
    1 -3 2 1 0 -3 -5 -1 3 3 -2 -3 -2 -1 1 0 5 4 2;
    3 -1 0 2 2 -5 -4 -1 2 -1 3 4 4 2 1 2 -2 1 -1]';
%% Calculating the new space
Psy=mean(Gamma,2);
Psy=repmat(Psy,[1, size(Gamma,2)]);

Phi=(Gamma-Psy);
S=Phi'*Phi;
disp('The inner product matrix is given by');
S
[V,eigenval]=eig(S);

E=V; %Each column is an eigenvector where in increasing order of eigenvalue
ui=Phi*V; % Each column is the eigenvector of The Big One.

%% Reconstruction of the dataset
Omega=ui'*Phi; % ak is the column of Omega
reconstX=Psy+ui*V';

%% Reconstruction using 3 dimensions

ui(:,1)=[];
mse=sum(sum((eigenval)));
fprintf('The minimum MSE representation is given by \n');
E(:,1)=[];
reconstX_3=Psy+ui*E';
reconstX_3
% Calculating the error
error_3=reconstX_3-Gamma;
rmserror_3=rms(error_3);
disp('RMS error for each point in the 3D reconstruction is given by');
rmserror_3'
%% Reconstruction of the dataset using 2 dimensions
ui(:,1)=[];
E(:,1)=[];

reconstX_2=Psy+ui*E';
error_2=reconstX_2-Gamma;
rmserror_2=rms(error_2);
disp('RMS error for each point in the 2D reconstruction is given by');
rmserror_2'
%% Creating the new data vector given in the question
Y=[1 3 0 3 -2 2 4 1 3 0 -2 0 1 1 -3 0 1 -2 -3];

dist_3D=pdist2(Y,reconstX_3')'
dist_4D=pdist2(Y,Gamma')'
