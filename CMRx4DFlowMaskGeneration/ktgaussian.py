# Copyright (c) [07/2024] Hao Li, Fudan University (h_li@fudan.edu.cn). All rights reserved.
# Translated to Python
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def create_gaussian_weight_matrix(mask_size, sigma_x, sigma_y):
    width, height = mask_size
    x = np.linspace(1 - (width + 1) / 2, width - (width + 1) / 2, width)
    y = np.linspace(1 - (height + 1) / 2, height - (height + 1) / 2, height)
    X, Y = np.meshgrid(x, y)
    weight = np.exp(-(X**2 / (2 * sigma_x**2) + Y**2 / (2 * sigma_y**2)))
    return weight

def random_sampling_optimized(mask_size, total_points, weight, min_dist_lookup, existing_mask):
    width, height = mask_size
    sampled_points = []
    
    current_weight = weight * (1 - existing_mask)
    if np.sum(current_weight) <= 0:
        return np.array([])

    flat_weight = current_weight.ravel()
    prob = flat_weight / flat_weight.sum()
    
    forbidden_mask = np.zeros((height, width), dtype=bool)
    batch_size = max(total_points * 2, 1000)
    indices = np.random.choice(width * height, size=batch_size, p=prob)
    
    count = 0
    idx_ptr = 0
    Y_grid, X_grid = np.ogrid[:height, :width]

    while count < total_points and idx_ptr < batch_size:
        idx = indices[idx_ptr]
        idx_ptr += 1
        
        y, x = divmod(idx, width)
        
        if forbidden_mask[y, x] or existing_mask[y, x]:
            continue
            
        sampled_points.append([x + 1, y + 1])
        count += 1
        
        d = min_dist_lookup[y, x]
        y_min, y_max = max(0, int(y - d)), min(height, int(y + d + 1))
        x_min, x_max = max(0, int(x - d)), min(width, int(x + d + 1))
        
        region_y = Y_grid[y_min:y_max, 0]
        region_x = X_grid[0, x_min:x_max]
        
        dist_sq = (region_y[:, np.newaxis] - y)**2 + (region_x - x)**2
        forbidden_mask[y_min:y_max, x_min:x_max] |= (dist_sq < d**2)
        
        if idx_ptr >= batch_size and count < total_points:
            current_weight = weight * (1 - existing_mask) * (1 - forbidden_mask)
            sw = current_weight.sum()
            if sw <= 0: break
            prob = current_weight.ravel() / sw
            indices = np.random.choice(width * height, size=batch_size, p=prob)
            idx_ptr = 0

    return np.array(sampled_points)

def fun_mask_gen_2d(
    mask_size,
    center_radius_x,
    center_radius_y,
    total_points,
    pattern_num,
    sigma_x,
    sigma_y,
    min_dist_factor,
    rep_decay_factor,
):
    """
    Generate kt-Poisson-like undersampling masks for (PE, SPE) k-space.

    Parameters
    ----------
    mask_size : (int, int)
        (PE, SPE) grid size.
    center_radius_x, center_radius_y : float
        Radius of the fully-sampled central ellipse along PE (x) and SPE (y).
    total_points : int
        Target number of sampled points (including the central ellipse) per mask.
    pattern_num : int
        Number of masks to generate (typically Nt).
    sigma_x, sigma_y : float
        Standard deviations of the Gaussian sampling density along PE and SPE.
    min_dist_factor : float
        Scales a per-location exclusion radius; 0 disables minimum-distance constraint.
    rep_decay_factor : float
        Multiplicative decay applied to the sampling weight at selected locations across patterns;
        1 disables repetition control, <1 reduces repeated sampling.

    Returns
    -------
    masks : np.ndarray, shape (SPE, PE, pattern_num), dtype float32
        Binary masks, stored as (SPE, PE, pattern_index).
    """
    width, height = mask_size
    masks = np.zeros((height, width, pattern_num), dtype=np.float32)
    
    initial_weight = create_gaussian_weight_matrix(mask_size, sigma_x, sigma_y)
    weight = initial_weight.copy()
    min_dist_lookup = min_dist_factor * ((1.0 - initial_weight) / 2.0 + 0.5)
    
    if (center_radius_x <= 0.5) or (center_radius_y <= 0.5):
        center_ellipse = np.zeros((height, width), dtype=bool)
        cy = int(round((height - 1) / 2))
        cx = int(round((width  - 1) / 2))
        center_ellipse[cy, cx] = True
    else:
        center_ellipse = (
            ((X - (width - 1) / 2) / center_radius_x) ** 2
            + ((Y - (height - 1) / 2) / center_radius_y) ** 2
            <= 1
        )
    num_center_points = np.sum(center_ellipse)

    for p in (range(pattern_num)):
        mask = np.zeros((height, width), dtype=np.float32)
        mask[center_ellipse] = 1
        
        needed = total_points - int(num_center_points)
        if needed > 0:
            points = random_sampling_optimized(mask_size, needed, weight, min_dist_lookup, mask)
            if len(points) > 0:
                xs = points[:, 0].astype(int) - 1
                ys = points[:, 1].astype(int) - 1
                mask[ys, xs] = 1
                weight[ys, xs] *= rep_decay_factor
        
        curr_total = np.sum(mask)
        if curr_total < total_points:
            extra_points = random_sampling_optimized(mask_size, total_points - int(curr_total), 
                                                   weight, min_dist_lookup, mask)
            if len(extra_points) > 0:
                xs = extra_points[:, 0].astype(int) - 1
                ys = extra_points[:, 1].astype(int) - 1
                mask[ys, xs] = 1
                weight[ys, xs] *= rep_decay_factor

        masks[:, :, p] = mask
    
    return masks
