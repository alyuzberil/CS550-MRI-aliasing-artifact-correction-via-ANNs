clear all;
close all;
% cd Images
% untar('IXI-T1.tar');
% gunzip('IXI002-Guys-0828-T1.nii.gz');
% V1 = niftiread('IXI002-Guys-0828-T1.nii');


load('ixi-t1_final.mat');
load('map.mat');

image = imdata(504,:,:);
image = squeeze(image);

im = zeros(size(image,1),size(image,2),8);
for i=1:8
    im(:,:,i) = image.*map(:,:,i);
end
% figure;
% montage(abs(im),'displayRange',[]);
% load('coilImages.mat');
% im = coilImages(100,:,:,:);

imageTemp = zeros(size(image,1),size(image,2));
for i=1:1:8
    imageTemp = imageTemp + conj(im(:,:,i)).*(im(:,:,i));
end
imageTemp = sqrt(imageTemp);
imageTemp(isnan(imageTemp))=0;
image = imageTemp;

Rx = 1;
Ry = 3;
calibx = 256;
caliby = 60;
[imu, Mu, autocalibration] = undersample(im, Rx, Ry, calibx, caliby);
im = imu;

% Parameters
b1x = 5;
b1y = 2;
N1 = 32;

b2x = 1;
b2y = 1;
N2 = 8;

b3x = 3;
b3y = 2;
nout = Rx - 1;

% normalization = 1;
% if normalization == 1
%     minRefImage = min(min(image));
%     minImage = min(min(min(real(im))));
%     image = image - minRefImage;
%     im = im - minImage;
%     image = image/max(max(real(image)));
%     im = im/max(max(max(real(im))));
% end

normalization = 1;
if normalization == 1
    image(real(image)<0)= 0;
    im(real(im)<0) = 0;
    image = image/max(max(real(image)));
    im = im/max(max(max(real(im))));
end

kspace = fft2c(im);
% normalize = 1/max(kspace(:));
% if normalization == 1
%     kspace = kspace*normalize;
%     autocalibration = autocalibration*normalize;
% end

[Xdim,Ydim,numCoils] = size(kspace);

kspaceORG = kspace;
kx = transpose(uint32((1:Xdim+1)));
ky = uint32((1:Ydim+1));

[autocalibrationX,autocalibrationY,autocalibrationZ] = size(autocalibration);
autocalibration_re = zeros(autocalibrationX,autocalibrationY,2*autocalibrationZ);
autocalibration_re(:,:,1:numCoils) = real(autocalibration);
autocalibration_re(:,:,numCoils+1:2*numCoils) = imag(autocalibration);

accRate = max(Ry,Rx);
numCoilChannels = 2*autocalibrationZ;
autocalibrationZ = numCoilChannels;

w1_allchannels = zeros(b1x, b1y, numCoilChannels, Ydim, numCoilChannels);
w2_allchannels = zeros(b2x, b2y, Ydim,N2,numCoilChannels);
w3_allchannels = zeros(b3x, b3y, N2,accRate - 1, numCoilChannels);  

recAreaXstart = ceil(b1x/2) + floor(b2x/2) + floor(b3x/2)-1+1; 
recAreaXend = autocalibrationX - recAreaXstart + 1;


autocalibration = autocalibration_re;

recAreaYstart = ((ceil(b1y/2)-1) + (ceil(b2y/2)-1) + (ceil(b3y/2)-1)) * accRate + 2;     
recAreaYend = autocalibrationY  - ((floor(b1y/2) + floor(b2y/2) + floor(b3y/2))) * accRate + 1;

recAreaX = recAreaXend - recAreaXstart + 1;
recAreaY = recAreaYend - recAreaYstart + 1;
recAreaZ = accRate - 1;

sizeW1 = [b1x, b1y, numCoilChannels, N1];
sizeW2 = [b2x, b2y, N1, N2];
sizeW3 = [b3x, b3y, N2, accRate - 1];

recAreaArr = zeros(autocalibrationZ,recAreaX,recAreaY,recAreaZ);
for i=1:autocalibrationZ
    recArea = zeros(recAreaX,recAreaY,recAreaZ);
    
    for j=1:accRate-1
        recAreaYstart = ((ceil(b1y/2)-1) + (ceil(b2y/2)-1) + (ceil(b3y/2)-1)) * accRate + j + 1;
        recAreaYend = autocalibrationY  - ((floor(b1y/2) + (floor(b2y/2)) + floor(b3y/2))) * accRate + j;
        recArea(:,:,j) = autocalibration(recAreaXstart:recAreaXend, recAreaYstart:recAreaYend,i);
    end
    recAreaArr(i,:,:,:) = recArea;
%     [W_conv1,W_conv2,W_conv3,error]=learning(autocalibration,recArea,accRate,sizeW1,sizeW2,sizeW3);
%     w1_allchannels(:,:,:,:,ind_c) = W_conv1;
%     w2_allchannels(:,:,:,:,ind_c) = W_conv2;
%     w3_allchannels(:,:,:,:,ind_c) = W_conv3;                              
%     print('Norm of Error = ',error);     
end
save('RAKI_PART1.mat');


function [w1,w2,w3,error] = learning(ACS,target,acc_rate,sizeW1,sizeW2,sizeW3)
ACS = squeeze(ACS);
target = squeeze(target);
W_conv1 = initializer(sizeW1);
h_conv1 = ReLU(myconv2d(ACS, W_conv1,[1 acc_rate]));

W_conv2 = initializer(sizeW2);
h_conv2 = ReLU(myconv2d(h_conv1, W_conv2,[1 acc_rate]));

W_conv3 = initializer(sizeW3);
h_conv3 = myconv2d(h_conv2, W_conv3, [],[1 acc_rate]);
errorMat = target - h_conv3;
targetTemp = target(:);
disp(norm(targetTemp )^2);
errorMat = errorMat(:);
error = norm(errorMat)^2;
disp(error);

errorArr = zeros(10^(7),1);
errorArr(1) = error;
for i=2:length(errorArr)
    dy = (-target + h_conv3).*h_conv3;
    [dh2, dw3, db] = vl_nnconv(h_conv2, W_conv3,[], dy, 'Dilate',[1 acc_rate]);
    
    h_conv1 = vl_nnrelu(vl_nnconv(ACS, W_conv1, [],'Dilate',[1 acc_rate]));
    l_conv2 = vl_nnconv(h_conv1, W_conv2, [],'Dilate',[1 acc_rate]);
    dh2l2 = l_conv2;
    dh2l2(dh2l2<=0) = 0;
    dh2l2(isnan(dh2l2)) = 1;
    dh2l2(dh2l2>0) = 1;
    dl2 = dh2.*dh2l2;
    [dh1, dw2, db] = vl_nnconv(h_conv1, W_conv2,[], dl2, 'Dilate',[1 acc_rate]);
    
    l_conv1 = vl_nnconv(ACS, W_conv1, [],'Dilate',[1 acc_rate]);
    dh1l1 = l_conv1;
    dh1l1(dh1l1<=0) = 0;
    dh1l1(isnan(dh1l1)) = 1;
    dh1l1(dh1l1>0) = 1;
    dl1 = dh1.*dh1l1;
    [dACS, dw1, db] = vl_nnconv(ACS, W_conv1,[], dl1, 'Dilate',[1 acc_rate]);
    
    W_conv1 = W_conv1 - learningRate*dw1;
    W_conv2 = W_conv2 - learningRate*dw2;
    W_conv3 = W_conv3 - learningRate*dw3;
    
    h_conv1 = ReLU(myconv2d(ACS, W_conv1,[1 acc_rate]));
    h_conv2 = ReLU(myconv2d(h_conv1, W_conv2,[1 acc_rate]));
    h_conv3 = myconv2d(h_conv2, W_conv3, [],[1 acc_rate]);
    errorMat = target - h_conv3;
    errorMat = errorMat(:);
    error = norm(errorMat)^2;
    errorArr(i) = error;
end

w1 = W_conv1;
w2 = W_conv1;
w3 = W_conv1;

end

function [initial] = initializer(size)
mu = 0;
sigma = 0.01;
initial = normrnd(mu,sigma,size);
end


function [imu, Mu, autocalibration] = undersample(im, Rx, Ry, calibx, caliby)
Mu = zeros(size(im,1),size(im,2),size(im,3));
imu = zeros(size(im,1),size(im,2),size(im,3));
autocalibration = zeros(calibx,caliby,size(im,3));
d = fft2c(im);
centerx1 = size(im,1)/2-(calibx/2-1);
centerx2 = size(im,1)/2+calibx/2;
centery1 = size(im,2)/2-(caliby/2-1);
centery2 = size(im,2)/2+caliby/2;
for i=1:size(im,3)
    temp = zeros(size(im,1),size(im,2));
    for j=1:Rx:size(im,1)
        for l=1:Ry:size(im,2)
            temp(size(im,1)+1-j,size(im,2)+1-l)=d(size(im,1)+1-j,size(im,2)+1-l,i);
        end
    end
    autocalibration(:,:,i) = d(centerx1:centerx2,centery1:centery2,i);
    temp(centerx1:centerx2,centery1:centery2) = d(centerx1:centerx2,centery1:centery2,i);
    Mu(:,:,i) = temp;
    imu(:,:,i) = ifft2c(temp);
end
end


function d = fft2c(im)
 d = fftshift(fft2(ifftshift(im)));
end
 
 function im = ifft2c(d)
 im = fftshift(ifft2(ifftshift(d)));
 end