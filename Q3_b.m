clc; clear;
load('data.mat');

testX=X(:,testimages);
testlabels=Y(testimages);

trainX=X(:,trainimages);
trainlabels=Y(trainimages);

classified=zeros(1,length(testimages));

[Psy_train,Phi_train,S_train,V_train,ui_train,Omega_train,reconstX_train,eigenval_train,numclass]=createspace(trainX,trainlabels);

% Calculating the error
error1=reconstX_train-trainX;
rmserror1=rms(error1');
toterror1=rms(rmserror1);

%% Nearest Neighbor Algorithm
% Regenerating the face space for test cases

%% Testing images (this case without PCA reduction)
distances=pdist2(testX',reconstX_train');
[~, Index]=min(distances,[],2);
answer=trainlabels(Index);
accuracy_all=sum(answer==testlabels)

fprintf('The accuracy of classification is %d on %d without PCA reduction\n',accuracy_all, length(testlabels));

%% Reconstruction and testing in reduced dimension space


%% Performing the PCA by providing a 90% variability
initvar=trace(S_train);
% finvar=sum(eigenval_train);

% Estimating the number of eigenvalues whose sum is below 10% variability
limit=0.1*initvar;

B=cumsum(eigenval_train);
I=find(B<limit,1,'last');

%% Reconstructing with the remaining dimensions

ui_train(:,1:I)=[];
V_train(:,1:I)=[];
reconstX_PCA=Psy_train+ui_train*V_train';

distances=pdist2(testX',reconstX_PCA');
[~, Index]=min(distances,[],2);
answer=trainlabels(Index);
accuracy_all=sum(answer==testlabels)
fprintf('The accuracy of classification is %d on %d with PCA reduction\n',accuracy_all, length(testlabels));