clear all;
close all;
load('RAKI_PART1.mat');
load('weights.mat');

kspaceUnd = kspace;
[kspaceUndX,kspaceUndY,kspaceUndZ] = size(kspaceUnd);

kspaceUndRec = zeros(kspaceUndX,kspaceUndY,kspaceUndZ*2);
kspaceUndRec(:,:,1:kspaceUndZ) = real(kspaceUnd);
kspaceUndRec(:,:,kspaceUndZ+1:kspaceUndZ*2) = imag(kspaceUnd);
kspaceUndRec = reshape(kspaceUndRec,[1,kspaceUndX,kspaceUndY,kspaceUndZ*2]);
kspaceRecon = kspaceUndRec;

for i=1:numCoilChannels
    % disp("Coil #"+(i));
    
    w1 = double(w1_allchannels(:,:,:,:,i));
    w2 = double(w2_allchannels(:,:,:,:,i));
    w3 = double(w3_allchannels(:,:,:,:,i));
    
    res = cnn_3layer(kspaceUndRec,w1,w2,w3,accRate);
    recAreaXendall = kspaceUndX - recAreaXstart + 1;
    
    for j=1:accRate-1
        recAreaYstart = ((ceil(b1y/2)-1) + ((ceil(b2y/2)-1)) + (ceil(b3y/2)-1)) * accRate + j + 1;
        recAreaYendall = kspaceUndY - ((floor(b1y/2)) + (floor(b2y/2)) + floor(b3y/2)) * accRate + j;
        kspaceRecon(1,recAreaXstart:recAreaXendall,recAreaYstart:accRate:recAreaYendall,i) = res(1,:,1:accRate:end,j);     
    end
end

kspaceRecon = squeeze(kspaceRecon);
kspaceReconComplex = kspaceRecon(:,:,1:numCoilChannels/2) + kspaceRecon(:,:,(numCoilChannels/2)+1:numCoilChannels)*1j;
kspaceReconComplex = enforceConstraint(kspaceReconComplex,kspaceORG,Rx,Ry,calibx,caliby);
kspaceReconFinal = kspaceReconComplex; 
im = ifft2c(kspaceReconFinal);
imOrg = ifft2c(kspaceORG);

mSOS = zeros(size(kspace,1),size(kspace,2));
for i=1:numCoilChannels/2
    mSOS = mSOS + conj(im(:,:,i)).*(im(:,:,i));
end
mSOS = sqrt(mSOS);
mSOS(isnan(mSOS))=0;

mSOSOrg = zeros(size(kspace,1),size(kspace,2));
for i=1:1:numCoilChannels/2
    mSOSOrg = mSOSOrg + conj(imOrg(:,:,i)).*(imOrg(:,:,i));
end
mSOSOrg = sqrt(mSOSOrg);
mSOSOrg(isnan(mSOSOrg))=0;

% normalization = 1;
% if normalization == 1
%     minRefImage = min(min(image));
%     minUndImage = min(min(mSOSOrg));
%     minRecImage = min(min(mSOS));
%     image = image - minRefImage;
%     mSOSOrg = mSOSOrg - minUndImage;
%     mSOS = mSOS - minRecImage;
%     image = image/max(max(real(image)));
%     mSOSOrg = mSOSOrg/max(max(real(mSOSOrg)));
%     mSOS = mSOS/max(max(real(mSOS)));
% end

normalization = 1;
if normalization == 1
%     image(real(image)<0)= 0;
%     mSOSOrg(real(mSOSOrg)<0)= 0;
%     image(real(image)<0)= 0;
    image = image/max(max(real(image)));
    mSOSOrg = mSOSOrg/max(max(real(mSOSOrg)));
    mSOS = mSOS/max(max(real(mSOS)));
end

peaksnr = psnr(abs(mSOS)/max(abs(mSOS(:))),abs(image)/max(abs(image(:))));
ssimval = ssim(abs(mSOS)/max(abs(mSOS(:))),abs(image)/max(abs(image(:))));
disp(peaksnr+"  "+ssimval);


% peaksnr = psnr(abs(mSOSOrg)/max(abs(mSOSOrg(:))),abs(image)/max(abs(image(:))));
% ssimval = ssim(abs(mSOSOrg)/max(abs(mSOSOrg(:))),abs(image)/max(abs(image(:))));
% disp(peaksnr);
% disp(ssimval);
load('undersampledImage.mat','mSOSUnd');
mSOSOrg = mSOSUnd;
mSOSOrg = 1.4*mSOSOrg/max(max(real(mSOSOrg)));
image = 1.4*image/max(max(real(image)));
mSOS = 1.4*mSOS/max(max(real(mSOS)));
errorImage = 2*(image - mSOS);
% figure; 
% subplot(1,4,1); imshow(image,[0 max(max(image))/1.4]); title("Reference");
% subplot(1,4,2); imshow(mSOSOrg,[0 max(max(mSOSOrg))/1.4]); title("Undersampled");
% subplot(1,4,3); imshow(mSOS,[0 max(max(mSOS))/1.4]); title("Reconstructed");
% subplot(1,4,4); imshow(errorImage,[0 max(max(errorImage))*2]); title("Error");
% saveas(gcf,"rakiSample.png");
% saveas(gcf,"rakiSample.eps",'epsc');

% figure;
% imshow(mSOS,[min(min(mSOS)) max(max(mSOS))/1.4]); title("RAKI");
% saveas(gcf,"rakiRecon.png");
% saveas(gcf,"rakiRecon.eps",'epsc');

images = zeros(256,1024);
images(:,1:256) = mSOSOrg;
images(:,257:512) = mSOS;
images(:,513:768) = image;
images(:,769:1024)= errorImage;
imwrite(abs(images),'rakiSample.png');
imwrite(abs(mSOS),'rakiRecon.png');

function [enforced] = enforceConstraint(kspaceRec,kspaceORG,Rx,Ry,calibx,caliby)
enforced = kspaceRec;
for i=1:size(kspaceORG,3)
    for j=1:Rx:size(kspaceORG,1)
        for l=1:Ry:size(kspaceORG,2)
            enforced(size(kspaceORG,1)+1-j,size(kspaceORG,2)+1-l,i) = kspaceORG(size(kspaceORG,1)+1-j,size(kspaceORG,2)+1-l,i);
        end
    end
end
centerx1 = size(kspaceORG,1)/2-(calibx/2-1);
centerx2 = size(kspaceORG,1)/2+calibx/2;
centery1 = size(kspaceORG,2)/2-(caliby/2-1);
centery2 = size(kspaceORG,2)/2+caliby/2;
enforced(centerx1:centerx2,centery1:centery2,:) = kspaceORG(centerx1:centerx2,centery1:centery2,:);
end

function [res] = cnn_3layer(kspaceUndRec,w1,w2,w3,accRate)
kspace = squeeze(kspaceUndRec);
h_conv1 = vl_nnrelu(vl_nnconv(kspace, w1, [],'Dilate',[1 accRate]));
h_conv2 = vl_nnrelu(vl_nnconv(h_conv1, w2, [],'Dilate',[1 accRate]));
h_conv3 = vl_nnconv(h_conv2, w3, [],'Dilate',[1 accRate]);
res = reshape(h_conv3,[1,size(h_conv3,1),size(h_conv3,2),size(h_conv3,3)]);
end

% function [res] = cnn_3layer(kspaceUndRec,w1,w2,w3,accRate)
% kspace = squeeze(kspaceUndRec);
% h_conv1 = ReLU(myconv2d(kspace, w1,[1,accRate]));
% h_conv2 = ReLU(myconv2d(h_conv1, w2,[1 accRate]));
% h_conv3 = myconv2d(h_conv2, w3,[1,accRate]);
% res = reshape(h_conv3,[1,size(h_conv3,1),size(h_conv3,2),size(h_conv3,3)]);
% end

function d = fft2c(im)
 d = fftshift(fft2(ifftshift(im)));
end
 
 function im = ifft2c(d)
 im = fftshift(ifft2(ifftshift(d)));
 end
 
function [output] = myconv2d(x,w,dilation)
accRateX = dilation(1);
accRateY = dilation(2);
numItX = size(x,1) - ((size(w,1)-1)*accRateX);
numItY = size(x,2) - ((size(w,2)-1)*accRateY);
output = zeros(numItX,numItY,size(w,4));
for k=1:size(w,4)
    for i=1:numItX
        for j=1:numItY
            xkernelend = i+((size(w,1)-1)*accRateX);
            ykernelend = j+((size(w,2)-1)*accRateY);
            kernelX = x(i:accRateX:xkernelend,j:accRateY:ykernelend,:);
            output(i,j,k) = output(i,j,k) + sum(kernelX.*w(:,:,:,k),'all');
        end
    end
end
end

function [X] = ReLU(X)
X(X<=0) = 0;
end