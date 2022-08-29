%% Modify the Images
clear all;
close all;
load('ixi-t1_new.mat');
imdata2 = zeros(size(imdata,1),256,256);
for i=1:size(imdata,1)
    image = imdata(i,:,:);
    image = squeeze(image);
    image = imresize(image,[256 192]);
    image2 = zeros(256,256);
    image2(:,33:224) = image;
    imdata2(i,:,:) = image2;
end
imdata = imdata2;
save('ixi-t1_final.mat','imdata');

deneme = imdata2(150,:,:);
deneme = squeeze(deneme);
figure;
imshow(deneme,[]);

%% Generate Coil Map
map = zeros(256,256,8);
X = [1:256];
Y = [1:256];
centers = zeros(8,2);
centers(1,:) = [32,64];
centers(2,:) = [96,64];
centers(3,:) = [160,64];
centers(4,:) = [224,64];
centers(5,:) = [32,192];
centers(6,:) = [96,192];
centers(7,:) = [160,192];
centers(8,:) = [224,192];
for k=1:8
    for i=1:256
        for j=1:256
            x = [X(i),Y(j)];
            map(i,j,k) = 1/(exp(norm(x-centers(k,:))/(128/1.21)));
        end
    end
end
save('map.mat','map');
load('ixi-t1_final.mat');
% figure;
% subplot(2,4,1); imshow(map(:,:,1),[]);
% subplot(2,4,2); imshow(map(:,:,2),[]);
% subplot(2,4,3); imshow(map(:,:,3),[]);
% subplot(2,4,4); imshow(map(:,:,4),[]);
% subplot(2,4,5); imshow(map(:,:,5),[]);
% subplot(2,4,6); imshow(map(:,:,6),[]);
% subplot(2,4,7); imshow(map(:,:,7),[]);
% subplot(2,4,8); imshow(map(:,:,8),[]);
% sgtitle("Magnitude of the coil sensitivities");
% saveas(gcf,"coilmaps.png");
% saveas(gcf,"coilmaps.eps",'epsc');

images = zeros(512,1024);
images(1:256,1:256) = map(:,:,1);
images(1:256,257:512) = map(:,:,2);
images(1:256,513:768) = map(:,:,3);
images(1:256,769:1024)= map(:,:,4);
images(257:end,1:256) = map(:,:,5);
images(257:end,257:512) = map(:,:,6);
images(257:end,513:768) = map(:,:,7);
images(257:end,769:1024)= map(:,:,8);
imwrite(abs(images),'coilMapping.png');

image = imdata(504,:,:);
image = squeeze(image);
figure; imshow(image,[]);

im = zeros(size(image,1),size(image,2),8);
for i=1:8
    im(:,:,i) = image.*map(:,:,i);
    im(:,:,i) = im(:,:,i)/max(max(im(:,:,i)));
end
figure;
montage(abs(im),'displayRange',[]);

images = zeros(512,1024);
images(1:256,1:256) = im(:,:,1);
images(1:256,257:512) = im(:,:,2);
images(1:256,513:768) = im(:,:,3);
images(1:256,769:1024)= im(:,:,4);
images(257:end,1:256) = im(:,:,5);
images(257:end,257:512) = im(:,:,6);
images(257:end,513:768) = im(:,:,7);
images(257:end,769:1024)= im(:,:,8);
imwrite(abs(images),'coilImage.png');

% figure;
% montage(abs(map1),'displayRange',[]);
% title("Map 1: Magnitude of the coil sensitivities");

return;

%% Generate Coil Images
load('ixi-t1_final.mat');
load('map.mat');
coilImages = zeros(size(imdata,1),size(imdata,2),size(imdata,3),8);
for j=1:size(imdata,1)
    j
    image = imdata(j,:,:);
    image = squeeze(image);
    
    im = zeros(size(image,1),size(image,2),8);
    for i=1:8
        im(:,:,i) = image.*map(:,:,i);
    end
    coilImages(j,:,:,:) = im;
end


load('map.mat');
figure;
montage(abs(map),'displayRange',[]);
title("Map: Magnitude of the coil sensitivities");
