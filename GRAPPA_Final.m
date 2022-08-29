clear all;
close all;

load('ixi-t1_final.mat');
load('map.mat');

psnrArr = zeros(81,1);
ssimArr = zeros(81,1);
for ii=1:81
image = imdata(500+ii,:,:);
image = squeeze(image);

im = zeros(size(image,1),size(image,2),8);
for i=1:8
    im(:,:,i) = image.*map(:,:,i);
end
% figure;
% montage(abs(im),'displayRange',[]);


imageTemp = zeros(size(image,1),size(image,2));
for i=1:1:8
    imageTemp = imageTemp + conj(im(:,:,i)).*(im(:,:,i));
end
image = imageTemp;
 
% load('multicoil-data.mat');

mSOS = zeros(size(im,1),size(im,2));
for i=1:8
    mSOS = mSOS + conj(im(:,:,i)).*(im(:,:,i));
end
mSOS = sqrt(mSOS);
mSOS(isnan(mSOS))=0;
calibx = 60;
caliby = 60;
Rx = 1;
Ry = 3;
lambda = 0;
[undersampled_images, undersampled_kspace, autocalibration] = undersample(im, Rx, Ry, calibx, caliby);
[reconst_images, Mr] = grappa(undersampled_kspace, autocalibration, Ry, lambda, calibx, caliby);
mSOSUnd = zeros(size(im,1),size(im,2));
for i=1:8
    mSOSUnd = mSOSUnd + conj(undersampled_images(:,:,i)).*(undersampled_images(:,:,i));
end
mSOSUnd = sqrt(mSOSUnd);
mSOSUnd(isnan(mSOSUnd))=0;

mSOSr = zeros(size(im,1),size(im,2));
for i=1:8
    mSOSr = mSOSr + conj(reconst_images(:,:,i)).*(reconst_images(:,:,i));
end
mSOSr = sqrt(mSOSr);
mSOSr(isnan(mSOSr))=0;
error = abs(abs(mSOSr)/max(abs(mSOSr(:)))-abs(mSOS)/max(abs(mSOS(:))));
peaksnrSOS = psnr(abs(mSOSr)/max(abs(mSOSr(:))),abs(mSOS)/max(abs(mSOS(:))));
ssimvalSOS = ssim(abs(mSOSr)/max(abs(mSOSr(:))),abs(mSOS)/max(abs(mSOS(:))));
disp("Psnr: "+peaksnrSOS+"   Ssim: "+ssimvalSOS);
psnrArr(ii) = peaksnrSOS;
ssimArr(ii) = ssimvalSOS;

% figure;
% subplot(2,2,1); imshow(abs(mSOS),[]);
% title("Referance Image");
% subplot(2,2,2); imshow(abs(mSOSUnd),[]);
% title("Undersampled Image");
% subplot(2,2,3); imshow(abs(mSOSr),[]);
% title("Magnitude Image");
% subplot(2,2,4); imshow(abs(error));
% title("Error Image");
end
grappaPSNR = psnrArr;
grappaSSIM = ssimArr;
% save('grappaResults.mat','grappaPSNR','grappaSSIM');


function [undersampled_images, undersampled_kspace, autocalibration] = undersample(im, Rx, Ry, calibx, caliby)
undersampled_kspace = zeros(size(im,1),size(im,2),size(im,3));
undersampled_images = zeros(size(im,1),size(im,2),size(im,3));
autocalibration = zeros(calibx,caliby,size(im,3));
kspace = fft2c(im);
centerx1 = size(im,1)/2-(calibx/2-1);
centerx2 = size(im,1)/2+calibx/2;
centery1 = size(im,2)/2-(caliby/2-1);
centery2 = size(im,2)/2+caliby/2;
for i=1:size(im,3)
    temp = zeros(size(im,1),size(im,2));
    for j=1:Rx:size(im,1)
        for l=1:Ry:size(im,2)
            temp(size(im,1)+1-j,size(im,2)+1-l)=kspace(size(im,1)+1-j,size(im,2)+1-l,i);
        end
    end
    autocalibration(:,:,i) = kspace(centerx1:centerx2,centery1:centery2,i);
    temp(centerx1:centerx2,centery1:centery2) = kspace(centerx1:centerx2,centery1:centery2,i);
    undersampled_kspace(:,:,i) = temp;
    undersampled_images(:,:,i) = ifft2c(temp);
end
end




function kernel =calibrate(autocalibration,Ry,lambda, calibx, caliby)
if Ry==2
    kernel = zeros(48,8);
    dataPoint = zeros((calibx-2)*(caliby-2),8);
    data_kernel = zeros((calibx-2)*(caliby-2),48);
    for l=1:8
        calib_kspacet = autocalibration(:,:,l);
        c = 1;
        for i=2:calibx-1
            for j=2:caliby-1
                % Each neighboring points is added to Ma one by one
                dataPoint(c,l) = calib_kspacet(i,j);
                data_kernel1 = calib_kspacet(i-1,j-1);
                data_kernel2 = calib_kspacet(i,j-1);
                data_kernel3 = calib_kspacet(i+1,j-1);
                data_kernel4 = calib_kspacet(i-1,j+1);
                data_kernel5 = calib_kspacet(i,j+1);
                data_kernel6 = calib_kspacet(i+1,j+1);
                data_kernel(c,1+6*(l-1):6*l) = [data_kernel1,data_kernel2,data_kernel3,data_kernel4,data_kernel5,data_kernel6];
                c = c+1;
            end
        end
    end
    for l=1:8
        % After generating Ma matrix and Mk vector, kernel can be
        % constructed
        kernel(:,l) = pinv(data_kernel'*data_kernel+lambda*eye(48))*data_kernel'*dataPoint(:,l);
    end
    
elseif Ry==3
    kernelUp = zeros(64,8);
    kernelDown = zeros(64,8);
    dataPoint_Up = zeros((calibx-3)*(caliby-3),8);
    dataPoint_Down = zeros((calibx-3)*(caliby-3),8);
    data_kernel = zeros((calibx-3)*(caliby-3),64);
    for l=1:8
        calib_kspacet = autocalibration(:,:,l);
        c = 1;
        for i=2:calibx-2
            for j=2:caliby-2
                dataPoint_Up(c,l) = calib_kspacet(i,j);
                dataPoint_Down(c,l) = calib_kspacet(i,j+1);
                data_kernel1 = calib_kspacet(i-1,j-1);
                data_kernel2 = calib_kspacet(i,j-1);
                data_kernel3 = calib_kspacet(i+1,j-1);
                data_kernel4 = calib_kspacet(i+2,j-1);
                data_kernel5 = calib_kspacet(i-1,j+2);
                data_kernel6 = calib_kspacet(i,j+2);
                data_kernel7 = calib_kspacet(i+1,j+2);
                data_kernel8 = calib_kspacet(i+2,j+2);
                data_kernel(c,1+8*(l-1):8*l) = [data_kernel1,data_kernel2,data_kernel3,data_kernel4,data_kernel5,data_kernel6,data_kernel7,data_kernel8];
                c = c+1;
            end
        end
    end
    for l=1:8
        % After generating Ma matrix and Mk vector, kernel can be
        % constructed
        kernelUp(:,l) = pinv(data_kernel'*data_kernel+lambda*eye(64))*data_kernel'*dataPoint_Up(:,l);
        kernelDown(:,l) = pinv(data_kernel'*data_kernel+lambda*eye(64))*data_kernel'*dataPoint_Down(:,l);
    end
    kernel = zeros(64,8,2);
    kernel(:,:,1) = kernelUp;
    kernel(:,:,2) = kernelDown;
    
    
elseif Ry==4
    kernelMid = zeros(80,8);
    kernelUp = zeros(80,8);
    kernelDown = zeros(80,8);
    dataPoint_Mid = zeros((calibx-4)*(caliby-4),8);
    dataPoint_Up = zeros((calibx-4)*(caliby-4),8);
    dataPoint_Down = zeros((calibx-4)*(caliby-4),8);
    data_kernel = zeros((calibx-4)*(caliby-4),80);
    for l=1:8
        calib_kspacet = autocalibration(:,:,l);
        c = 1;
        for i=3:calibx-2
            for j=3:caliby-2
                dataPoint_Mid(c,l) = calib_kspacet(i,j);
                dataPoint_Up(c,l) = calib_kspacet(i,j-1);
                dataPoint_Down(c,l) = calib_kspacet(i,j+1);
                data_kernel1 = calib_kspacet(i-2,j-2);
                data_kernel2 = calib_kspacet(i-1,j-2);
                data_kernel3 = calib_kspacet(i,j-2);
                data_kernel4 = calib_kspacet(i+1,j-2);
                data_kernel5 = calib_kspacet(i+2,j-2);
                data_kernel6 = calib_kspacet(i-2,j+2);
                data_kernel7 = calib_kspacet(i-1,j+2);
                data_kernel8 = calib_kspacet(i,j+2);
                data_kernel9 = calib_kspacet(i+1,j+2);
                data_kernel10 = calib_kspacet(i+2,j+2);
                data_kernel(c,1+10*(l-1):10*l) = [data_kernel1,data_kernel2,data_kernel3,data_kernel4,data_kernel5,data_kernel6,data_kernel7,data_kernel8,data_kernel9,data_kernel10];
                c = c+1;
            end
        end
    end
    for l=1:8
        % After generating Ma matrix and Mk vector, kernel can be
        % constructed
        kernelMid(:,l) = pinv(data_kernel'*data_kernel+lambda*eye(80))*data_kernel'*dataPoint_Mid(:,l);
        kernelUp(:,l) = pinv(data_kernel'*data_kernel+lambda*eye(80))*data_kernel'*dataPoint_Up(:,l);
        kernelDown(:,l) = pinv(data_kernel'*data_kernel+lambda*eye(80))*data_kernel'*dataPoint_Down(:,l);
    end
    kernel = zeros(80,8,3);
    kernel(:,:,1) = kernelUp;
    kernel(:,:,2) = kernelMid;
    kernel(:,:,3) = kernelDown;
end
end


function [reconstructedImages, reconstructed_kspace] = grappa(undersampled_kspace, calib_kspace, Ry, lambda, calibx, caliby)
if Ry==2
    reconstructed_kspace = undersampled_kspace;
    kernel =calibrate(calib_kspace,Ry,lambda, calibx, caliby);
    reconstructedImages = zeros(size(undersampled_kspace,1),size(undersampled_kspace,2),size(undersampled_kspace,3));
    for k=1:8
        a = kernel(:,k);
        for j=3:2:size(undersampled_kspace,2)-1
            for i=2:size(undersampled_kspace,1)-1
                kernel_vect = zeros(48,1);
                for l=0:7
                    kernel_vect(6*l+1) = undersampled_kspace(i-1,j-1,l+1);
                    kernel_vect(6*l+2) = undersampled_kspace(i,j-1,l+1);
                    kernel_vect(6*l+3) = undersampled_kspace(i+1,j-1,l+1);
                    kernel_vect(6*l+4) = undersampled_kspace(i-1,j+1,l+1);
                    kernel_vect(6*l+5) = undersampled_kspace(i,j+1,l+1);
                    kernel_vect(6*l+6) = undersampled_kspace(i+1,j+1,l+1);
                end
                reconstructed_kspace(i,j,k)=transpose(a)*kernel_vect;
            end
        end
        centerx1 = size(undersampled_kspace,1)/2-(calibx/2-1);
        centerx2 = size(undersampled_kspace,1)/2+calibx/2;
        centery1 = size(undersampled_kspace,2)/2-(caliby/2-1);
        centery2 = size(undersampled_kspace,2)/2+caliby/2;
        reconstructed_kspace(centerx1:centerx2,centery1:centery2,:) = undersampled_kspace(centerx1:centerx2,centery1:centery2,:);
        reconstructedImages(:,:,k) = ifft2c(reconstructed_kspace(:,:,k));
    end
    
elseif Ry==3
    reconstructed_kspace = undersampled_kspace;
    kernel =calibrate(calib_kspace,Ry,lambda, calibx, caliby);
    reconstructedImages = zeros(size(undersampled_kspace,1),size(undersampled_kspace,2),size(undersampled_kspace,3));
    for k=1:8
        kernelUp = kernel(:,k,1);
        kernelDown = kernel(:,k,2);
        for j=3:3:size(undersampled_kspace,2)-2
            for i=2:size(undersampled_kspace,1)-2
                kernel_vect = zeros(64,1);
                for l=0:7
                    kernel_vect(8*l+1) = undersampled_kspace(i-1,j-1,l+1);
                    kernel_vect(8*l+2) = undersampled_kspace(i,j-1,l+1);
                    kernel_vect(8*l+3) = undersampled_kspace(i+1,j-1,l+1);
                    kernel_vect(8*l+4) = undersampled_kspace(i+2,j-1,l+1);
                    kernel_vect(8*l+5) = undersampled_kspace(i-1,j+2,l+1);
                    kernel_vect(8*l+6) = undersampled_kspace(i,j+2,l+1);
                    kernel_vect(8*l+7) = undersampled_kspace(i+1,j+2,l+1);
                    kernel_vect(8*l+8) = undersampled_kspace(i+2,j+2,l+1);
                end
                reconstructed_kspace(i,j,k)=transpose(kernelUp)*kernel_vect;
                reconstructed_kspace(i,j+1,k)=transpose(kernelDown)*kernel_vect;
            end
        end
        centerx1 = size(undersampled_kspace,1)/2-(calibx/2-1);
        centerx2 = size(undersampled_kspace,1)/2+calibx/2;
        centery1 = size(undersampled_kspace,2)/2-(caliby/2-1);
        centery2 = size(undersampled_kspace,2)/2+caliby/2;
        reconstructed_kspace(centerx1:centerx2,centery1:centery2,:) = undersampled_kspace(centerx1:centerx2,centery1:centery2,:);
        reconstructedImages(:,:,k) = ifft2c(reconstructed_kspace(:,:,k));
    end
      
elseif Ry==4
    reconstructed_kspace = undersampled_kspace;
    kernel =calibrate(calib_kspace,Ry,lambda, calibx, caliby);
    reconstructedImages = zeros(size(undersampled_kspace,1),size(undersampled_kspace,2),size(undersampled_kspace,3));
    for k=1:8
        kernelUp = kernel(:,k,1);
        kernelMid = kernel(:,k,2);
        kernelDown = kernel(:,k,3);
        for j=6:4:size(undersampled_kspace,2)-2
            for i=3:size(undersampled_kspace,1)-2
                kernel_vect = zeros(80,1);
                for l=0:7
                    kernel_vect(10*l+1) = undersampled_kspace(i-2,j-2,l+1);
                    kernel_vect(10*l+2) = undersampled_kspace(i-1,j-2,l+1);
                    kernel_vect(10*l+3) = undersampled_kspace(i,j-2,l+1);
                    kernel_vect(10*l+4) = undersampled_kspace(i+1,j-2,l+1);
                    kernel_vect(10*l+5) = undersampled_kspace(i+2,j-2,l+1);
                    kernel_vect(10*l+6) = undersampled_kspace(i-2,j+2,l+1);
                    kernel_vect(10*l+7) = undersampled_kspace(i-1,j+2,l+1);
                    kernel_vect(10*l+8) = undersampled_kspace(i,j+2,l+1);
                    kernel_vect(10*l+9) = undersampled_kspace(i+1,j+2,l+1);
                    kernel_vect(10*l+10) = undersampled_kspace(i+2,j+2,l+1);
                end
                reconstructed_kspace(i,j-1,k)=transpose(kernelUp)*kernel_vect;
                reconstructed_kspace(i,j,k)=transpose(kernelMid)*kernel_vect;
                reconstructed_kspace(i,j+1,k)=transpose(kernelDown)*kernel_vect;
            end
        end
        centerx1 = size(undersampled_kspace,1)/2-(calibx/2-1);
        centerx2 = size(undersampled_kspace,1)/2+calibx/2;
        centery1 = size(undersampled_kspace,2)/2-(caliby/2-1);
        centery2 = size(undersampled_kspace,2)/2+caliby/2;
        reconstructed_kspace(centerx1:centerx2,centery1:centery2,:) = undersampled_kspace(centerx1:centerx2,centery1:centery2,:);
        reconstructedImages(:,:,k) = ifft2c(reconstructed_kspace(:,:,k));
    end
end

end

function d = fft2c(im)
 d = fftshift(fft2(ifftshift(im)));
end
 
 function im = ifft2c(d)
 im = fftshift(ifft2(ifftshift(d)));
 end