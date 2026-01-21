function masks = fun_maskGen2D(maskSize, centerRadiusX, centerRadiusY, totalPoints, patternNum, sigmaX, sigmaY, minDistFactor, repDecayFactor)
% Description: This function generates multiple random sampling masks with Gaussian weighted probability and dynamically changing minimum distance constraint.
%              The merged mask has a distribution that is as uniform as possible with the fewest duplicate points.
%              The generated masks have varying densities from the center to the periphery, controlled by Gaussian distributions along both axes.
%
% Usage:
%   masks = fun_maskGen2D(maskSize, centerRadiusX, centerRadiusY, totalPoints, patternNum, sigmaX, sigmaY, minDistFactor, repDecayFactor)
%
% Inputs:
%   - maskSize: [width, height] - Size of the mask.
%   - centerRadiusX: Long axis of the fully sampled ellipse.
%   - centerRadiusY: Short axis of the fully sampled ellipse.
%   - totalPoints: Total number of random points per mask.
%   - patternNum: Number of masks to generate.
%   - sigmaX: Controls the width of the Gaussian distribution in the X direction.
%   - sigmaY: Controls the width of the Gaussian distribution in the Y direction.
%   - minDistFactor: Factor to scale minDist based on distance from the center.
%   - repDecayFactor: Fixed probability factor to control point repetition across masks (range: 0~1).
%
% Outputs:
%   - masks: 3D array containing the generated masks.
%
% Example:
%   maskSize = [64, 256]; % Mask size
%   centerRadiusX = 6; % Long axis of the fully sampled ellipse
%   centerRadiusY = 16; % Short axis of the fully sampled ellipse
%   totalPoints = 1000; % Total number of random points per mask
%   patternNum = 100; % Number of masks to generate
%   sigmaX = maskSize(1) / 4; % Control the width of the Gaussian distribution in X direction
%   sigmaY = maskSize(2) / 4; % Control the width of the Gaussian distribution in Y direction
%   minDistFactor = 3; % Factor to scale minDist based on distance from the center
%   repDecayFactor = 0.5; % Fixed probability factor to control point repetition across masks
%
%   masks = fun_maskGen2D(maskSize, centerRadiusX, centerRadiusY, totalPoints, patternNum, sigmaX, sigmaY, minDistFactor, repDecayFactor);
%
% Copyright (c) [07/2024] Hao Li, Fudan University (h_li@fudan.edu.cn). All rights reserved.
%
% 2024.7.26 HL Modification.
% 1. 点与点之间的距离改为符合高斯分布，而不是固定距离minDist越小，离kspace中心越近，minDist越小。其分布与create_gaussian_weight_matrix产生的权重分布大体一致（具体分布仍可优化）。
% 2. 手动设置mask间采集点的重复概率衰减系数repDecayFactor（范围：0~1），即每采集到一个位置，该位置的权重*repDecayFactor，以降低被再次选中的概率。
% 3. 修正了若干错误，之前版本重复点的回避机制有问题。


width = maskSize(1);
height = maskSize(2);

masks = zeros(height, width, patternNum);

% Create initial weight matrix
initialWeight = create_gaussian_weight_matrix(maskSize, sigmaX, sigmaY);

weight = initialWeight;

for p = 1:patternNum

    mask = zeros(height, width);

    % Ensure the center region is fully sampled
    [X, Y] = meshgrid(1:width, 1:height);
    centerEllipse = ((X - width / 2) / centerRadiusX).^2 + ((Y - height / 2) / centerRadiusY).^2 <= 1;
    mask(centerEllipse) = 1;
    % weight(centerEllipse) = weight(centerEllipse) * repDecayFactor; % Reduce probability for the center region

    % Random sampling with weighted probability
    points = random_sampling(maskSize, totalPoints - sum(centerEllipse(:)), weight, minDistFactor, initialWeight);

    % Set the sampled points in the mask
    for i = 1:size(points, 1)
        x = points(i, 1);
        y = points(i, 2);
        if x > 0 && x <= width && y > 0 && y <= height && mask(y, x) == 0
            mask(y, x) = 1;
            weight(y, x) = weight(y, x) * repDecayFactor; % Reduce probability for this point
        end
    end

    % Ensure the mask has the required number of points
    currentPoints = sum(mask(:));
    while currentPoints < totalPoints
        additionalPoints = random_sampling(maskSize, totalPoints - currentPoints, weight, minDistFactor, initialWeight);
        for i = 1:size(additionalPoints, 1)
            x = additionalPoints(i, 1);
            y = additionalPoints(i, 2);
            if x > 0 && x <= width && y > 0 && y <= height && mask(y, x) == 0
                mask(y, x) = 1;
                weight(y, x) = weight(y, x) * repDecayFactor; % Reduce probability for this point
            end
        end
        currentPoints = sum(mask(:));
    end

    % Store the mask
    masks(:, :, p) = mask;
end
end

function weight = create_gaussian_weight_matrix(maskSize, sigmaX, sigmaY)
% Create a Gaussian weight matrix where the weight decreases with distance from center
width = maskSize(1);
height = maskSize(2);
centerX = width / 2;
centerY = height / 2;

% Create a grid of distances from the center
[X, Y] = meshgrid(1:width, 1:height);
distancesX = (X - centerX).^2 / (2 * sigmaX^2);
distancesY = (Y - centerY).^2 / (2 * sigmaY^2);

% Create Gaussian weight matrix
weight = exp(-(distancesX + distancesY));
end

function points = random_sampling(maskSize, totalPoints, weight, minDistFactor, initialWeight)
% Random sampling algorithm with weighted probability and dynamically changing minimum distance constraint
width = maskSize(1);
height = maskSize(2);

% Normalize weight to sum to 1
weight = weight / sum(weight(:));

% Create cumulative distribution function (CDF) for weighted sampling
cdf = cumsum(weight(:));

% Generate random points based on the CDF
points = zeros(totalPoints, 2);
count = 0;
while count < totalPoints
    r = rand();
    index = find(cdf >= r, 1, 'first');
    [y, x] = ind2sub([height, width], index);
    newPoint = [x, y];

    % Calculate minDist based on initialWeight
    gaussianWeight = initialWeight(y, x);
    factor= (1-gaussianWeight)/2 + 0.5;         % TODO: can be optimized
    minDist = minDistFactor * factor;

    % Check if the new point is at least minDist away from existing points
    if count == 0 || all(sqrt(sum((points(1:count, :) - newPoint).^2, 2)) >= minDist)
        count = count + 1;
        points(count, :) = newPoint;
    end
end
end
