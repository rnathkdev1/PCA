%% Creating the dataset in Q1
X=[2.5 2.4; 0.5 0.7;2.2 2.9; 1.9 2.2; 3.1 3.0; 2.3 2.7; 2.0 1.6; 1.0 1.1; 1.5 1.6; 1.1 0.9];

%% Calculating the new space
meanX=mean(X);
meanX=repmat(meanX,[length(X) 1]);

D=(X-meanX)';
S=D*D';

[V,~]=eig(S);

e1=V(:,1)'; %Row vector
e2=V(:,2)'; %Row vector
E=[e1;e2]; %Each row is an eigenvector where in increasing order of eigenvalue
ak=E*D; %First row is a1 and second is a2, the new coefficients for the eigenspace

%% Reconstruction of the dataset

ak=ak'; %For ease of matrix manipulation
reconstX=meanX+ak*E;

plote1=[unique(meanX)'; unique(meanX)'+e1];
plote2=[unique(meanX)'; unique(meanX)'+e2];
%% Plotting the dataset and new eigenspace
figure(1)
scatter(X(:,1),X(:,2));
hold on;
plot(plote1(:,1),plote1(:,2),'red');
plot(plote2(:,1),plote2(:,2),'red');
xlabel('Original X');
ylabel('Original Y');
title('Eigenvector plot')

%% Plotting in the new set of coordinates
figure(2)
scatter(ak(:,2),ak(:,1));
ylabel('Lowest Eigenvector')
xlabel('Highest eigenvector');
title('Eigenspace');


%% Calculating the optimal One dimensional representation of data
ak2=zeros(length(ak),1);
ak2(:,2)=ak(:,2);
reconstX_=meanX+ak2*E;

figure(3)
scatter(reconstX_(:,1),reconstX_(:,2));
xlabel('Original X');
ylabel('Original Y');
title('Compressed Values');

%% Displaying required answers.
disp('Principal Components are \n');
e1
e2
disp('The new a1 and a2 for each data point are')
ak
disp('Range of data is 3.4534');