clc; clear;

load('data.mat');
testX=X(:,testimages);
trainX=X(:,trainimages);
classified=zeros(1,length(testimages));
%% Calculating the new space
trainX=trainX';
meanX=mean(trainX);
meanX=repmat(meanX,[size(trainX,1) 1]);

D=(trainX-meanX);
S=D*D';

[V,eigval]=eig(S);
eigval=unique(eigval);
eigval(eigval==0)=[];
E=V'; %Each row is an eigenvector where in increasing order of eigenvalue
ak=E*D; %First row is a1 and second is a2, the new coefficients for the eigenspace

%% Reconstruction of the dataset

ak=ak'; %For ease of matrix manipulation
reconstX_all=meanX+(ak*E)';

%% Performing the PCA by providing a 90% variability
initvar=trace(S);

% Estimating the number of eigenvalues whose sum is below 10% variability
limit=0.1*initvar;

B=cumsum(eigval);
I=find(B<limit,1,'last');

%% Reconstructing with the remaining dimensions
ak_reconst=ak(:,I+1:end);
E_reconst=E(I+1:end,:);
eig_reconst=eigval(I+1:end);
reconstX_PCA=meanX+(ak_reconst*E_reconst)';

% Calculating the error
error=reconstX_PCA-trainX;
rmserror=rms(error');
toterror=rms(rmserror);

%% Displaying the 5 eigenfaces of the first image:

for i=size(ak_reconst,2)-5:size(ak_reconst,2)
    figure(size(ak_reconst,2)-i+1)
    imshow(reshape(ak_reconst(:,i),231,195));
    title('Eigenface');
end

fprintf('The new number of dimensions is %d\n',size(ak_reconst,2));

initvar=repmat(initvar,[size(ak_reconst,2), 1]);
percentvar=(eig_reconst./initvar)*100;
figure(6)
bar(fliplr(percentvar'));
title('Bar Graph representing the contribution of eigenvalue');
ylabel('Percentage Contribution');

%% Reconstructing the images using 10, 20, 30, 40 and 50 eigenfaces

%Using 10 eigenfaces
reconstX_10=meanX+(ak(:,end-10:end)*E(end-10:end,:))';
% Using 20 eigenfaces
reconstX_20=meanX+(ak(:,end-20:end)*E(end-20:end,:))';
% Using 30 eigenfaces
reconstX_30=meanX+(ak(:,end-30:end)*E(end-30:end,:))';
% Using 40 eigenfaces
reconstX_40=meanX+(ak(:,end-40:end)*E(end-40:end,:))';
% Using 50 eigenfaces
reconstX_50=meanX+(ak(:,end-50:end)*E(end-50:end,:))';

figure(7)
subplot(1,2,1)
imshow(reshape(reconstX_10(1,:),231,195))
title('Image reconstruction using 10 eigenfaces')
subplot(1,2,2)
imshow(reshape(trainX(1,:),231,195));
title('Original Image')

figure(8)
subplot(1,2,1)
imshow(reshape(reconstX_20(1,:),231,195))
title('Image reconstruction using 20 eigenfaces')
subplot(1,2,2)
imshow(reshape(trainX(1,:),231,195));
title('Original Image')

figure(9)
subplot(1,2,1)
imshow(reshape(reconstX_30(1,:),231,195))
title('Image reconstruction using 30 eigenfaces')
subplot(1,2,2)
imshow(reshape(trainX(1,:),231,195));
title('Original Image')

figure(10)
subplot(1,2,1)
imshow(reshape(reconstX_40(1,:),231,195))
title('Image reconstruction using 40 eigenfaces')
subplot(1,2,2)
imshow(reshape(trainX(1,:),231,195));
title('Original Image')

figure(11)
subplot(1,2,1)
imshow(reshape(reconstX_50(1,:),231,195))
title('Image reconstruction using 50 eigenfaces')
subplot(1,2,2)
imshow(reshape(trainX(1,:),231,195));
title('Original Image')

%% Nearest Neighbor Algorithm
% Regenerating the face space for test cases

%% Calculating the new space for test images
testX=testX';
meanX_test=mean(testX);
meanX_test=repmat(meanX_test,[size(testX,1) 1]);

D_test=(testX-meanX_test);
S_test=D_test*D_test';

[V_test,eigval_test]=eig(S_test);
eigval_test=unique(eigval_test);
eigval_test(eigval_test==0)=[];
E_test=V_test'; %Each row is an eigenvector where in increasing order of eigenvalue
ak_test=E_test*D_test; %First row is a1 and second is a2, the new coefficients for the eigenspace

%% Reconstruction and testing of the dataset in full dimension space

ak_test=ak_test'; %For ease of matrix manipulation
reconstX_test_all=meanX_test+(ak_test*E_test)';

%Calculating the distance threshold
theta_c=0.5*max(max((pdist2(ak',ak'))));
Epsilon=sqrt(sum(((testX-reconstX_test_all).*(testX-reconstX_test_all)),2));

% Testing
Epsilon_k=pdist2(ak_test',ak');
% the i,j th element of D contains the distance between the ith entry of
% test data and Jth entry of the reconst traindata.
[Epsilon_kmin, I]=min(Epsilon_k,[],2);
[Epsilon_kmax, ~]=max(Epsilon_k,[],2);

logicalNotFace=Epsilon>=theta_c;
logicalUnknownFace=and((Epsilon<theta_c), (Epsilon_kmax>=theta_c));
logicalKnownFace=and((Epsilon<theta_c), (Epsilon_kmin<=theta_c));


classified(logicalNotFace)=0;
classified(logicalUnknownFace)=-1;


%% Reconstruction and testing in reduced dimension space
ak_test_reconst=ak_test(:,I+1:end);
E_test_reconst=E_test(I+1:end,:);
eig_test_reconst=eigval_test(I+1:end);
reconstX_test_PCA=meanX_test+(ak_test_reconst*E_test_reconst)';

