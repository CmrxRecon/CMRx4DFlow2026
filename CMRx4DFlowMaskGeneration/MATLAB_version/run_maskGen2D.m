% Copyright (c) [07/2024] Hao Li, Fudan University (h_li@fudan.edu.cn). All rights reserved.

clc;clear;%close all;

totalPoints = 384; % Total number of random points per mask
patternNum = 24; % Number of masks to generate

maskSize = [128,24]; % Mask size
centerRadiusX = ceil(maskSize(1)*0.05); % Long axis of the fully sampled ellipse
centerRadiusY = ceil(maskSize(2)*0.15); % Short axis of the fully sampled ellipse
sigmaX = maskSize(1) / 5; % Control the width of the Gaussian distribution in X direction
sigmaY = maskSize(2) / 5; % Control the width of the Gaussian distribution in Y direction

% Two key parameters:
minDistFactor = 0; % Factor to scale minDist based on distance from the center
prob = 1; % Used to control the probability of point repetition across different masks

masks = fun_maskGen2D(maskSize, centerRadiusX, centerRadiusY, totalPoints, patternNum, sigmaX, sigmaY, minDistFactor, prob);

% figure; imshow3D(masks); title('All masks');

combinedNum = 6;%round(patternNum/5); % Randomly choose a few patterns
selectedIndices = randperm(patternNum, combinedNum);
combinedMask = sum(masks(:, :, selectedIndices), 3);
figure; imshow(permute(combinedMask>0,[2,1]), []); % colormap jet;
title(sprintf('Combined mask from %d patterns',combinedNum));


combinedMask = sum(masks, 3);
figure; imshow(permute(combinedMask>0,[2,1]), [])
% figure; imshow(combinedMask,[]); colormap jet;
% title('All Points Distribution');

% calculate undersampled factor
underFactor = (maskSize(1)*maskSize(2)*patternNum)/sum(masks(:)>0);
disp("Undersampled Factor: "+underFactor)
pattern.mask = permute(masks,[2,1,3]);

all_phs = [];
for i = 1:size(masks,3)    
    all_phs = cat(2,all_phs,permute(masks(:,:,i),[2,1]));
end
figure,imshow(all_phs,[])