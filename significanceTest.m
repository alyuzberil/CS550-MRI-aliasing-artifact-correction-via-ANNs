clear all;
close all;
load('grappaResults.mat');
load('rakiResults.mat');
load('results_gan.mat');

psnrGAN = transpose(psnr);
ssimGAN = transpose(ssim);
load('results_unet.mat');
psnrUNET= transpose(psnr);
ssimUNET = transpose(ssim);

%% Mean and Standard deviation
psnr = psnrUNET;
ssim = ssimUNET;

disp("meanPSNR: "+mean(psnr));
disp("stdPSNR: "+std(psnr));
disp("meanSSIM: "+mean(ssim));
disp("stdSSIM: "+std(ssim));

%% Significance Test
A = ssimUNET;
B = ssimGAN;

% A = [2.5,3.5,2.9,2.1,6.9,2.4,4.9,6.6,2.0,2.0,5.8,7.5];
% B = [4.0,5.6,3.2,1.9,9.5,2.3,6.7,6.0,3.5,4.0,8.1,19.9];

diff = B - A;
sign = zeros(length(diff),1);
sign(diff>0) = 1;
sign(diff<0) = -1;
diff1 = abs(diff);
[difference,Index] = sort(diff1,'ascend');
sign = sign(Index);
difference(difference==0) = [];
sign(sign==0) = [];
WPos = 0;
WNeg = 0;
for i=1:length(sign)
    if sign(i)<0
        WNeg = WNeg + i;
    elseif sign(i)>0
        WPos = WPos + i;
    end
end
[num,numEqual] = mode(difference);
if numEqual==1
    numEqual = 0;
end
mean = length(sign)*(length(sign)+1)/4;
deviation = sqrt((length(sign)*(length(sign)+1)*(2*length(sign)+1)/24)-(numEqual^3-numEqual)/48);
W = min(WPos,WNeg);
z = (W-mean)/deviation;

mu = 0;
sigma = 1;
pd = makedist('Normal','mu',mu,'sigma',sigma);
p_value = cdf(pd,z);
p_value = 1-2*p_value;
disp("P_value: "+p_value);




