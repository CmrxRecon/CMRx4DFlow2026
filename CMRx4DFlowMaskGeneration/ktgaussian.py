# Copyright (c) 07/2024 Hao Li, Fudan University (h_li@fudan.edu.cn).
# All rights reserved.
# Translated to Python

import numpy as np


def create_gaussian_weight_matrix(mask_size, sigma_x, sigma_y):
    """
    Create a 2D anisotropic Gaussian weight (sampling density) over the k-space grid.

    Parameters
    ----------
    mask_size : (int, int)
        (width, height) = (PE, SPE) grid size.
    sigma_x, sigma_y : float
        Standard deviations of the Gaussian along x (PE) and y (SPE).

    Returns
    -------
    weight : np.ndarray, shape (height, width)
        Gaussian weights (higher near center).
    """
    width, height = mask_size
    x = np.linspace(1 - (width + 1) / 2, width - (width + 1) / 2, width)
    y = np.linspace(1 - (height + 1) / 2, height - (height + 1) / 2, height)
    X, Y = np.meshgrid(x, y)
    weight = np.exp(-(X**2 / (2 * sigma_x**2) + Y**2 / (2 * sigma_y**2)))
    return weight


def random_sampling_optimized(mask_size, total_points, weight, min_dist_lookup, existing_mask):
    """
    Sample points from a weighted 2D grid while enforcing a local minimum-distance constraint.

    This function:
    - draws candidate indices according to the provided weight map (excluding existing_mask),
    - accepts candidates that are not forbidden,
    - marks a disk (radius d) around accepted points as forbidden, where d varies per location.

    Parameters
    ----------
    mask_size : (int, int)
        (width, height) grid size.
    total_points : int
        Number of points to sample.
    weight : np.ndarray, shape (height, width)
        Nonnegative sampling weights.
    min_dist_lookup : np.ndarray, shape (height, width)
        Per-location exclusion radius (in pixels). Set to 0 to disable distance constraints.
    existing_mask : np.ndarray, shape (height, width)
        Binary mask of already-selected locations (1 = occupied).

    Returns
    -------
    sampled_points : np.ndarray, shape (N, 2)
        Sampled points as 1-based coordinates [[x, y], ...], where N <= total_points.
        Returns an empty array if no valid sampling locations exist.
    """
    width, height = mask_size
    sampled_points = []

    # Exclude already-selected points from the probability mass.
    current_weight = weight * (1 - existing_mask)
    if np.sum(current_weight) <= 0:
        return np.array([])

    flat_weight = current_weight.ravel()
    prob = flat_weight / flat_weight.sum()

    forbidden_mask = np.zeros((height, width), dtype=bool)

    # Draw more candidates than needed to compensate for rejections.
    batch_size = max(total_points * 2, 1000)
    indices = np.random.choice(width * height, size=batch_size, p=prob)

    count = 0
    idx_ptr = 0
    Y_grid, X_grid = np.ogrid[:height, :width]

    while count < total_points and idx_ptr < batch_size:
        idx = indices[idx_ptr]
        idx_ptr += 1

        y, x = divmod(idx, width)

        # Skip if forbidden by distance constraint or already selected.
        if forbidden_mask[y, x] or existing_mask[y, x]:
            continue

        # Store as 1-based coordinates to match the original implementation.
        sampled_points.append([x + 1, y + 1])
        count += 1

        # Mark neighborhood within radius d as forbidden.
        d = float(min_dist_lookup[y, x])
        if d > 0:
            y_min, y_max = max(0, int(y - d)), min(height, int(y + d + 1))
            x_min, x_max = max(0, int(x - d)), min(width, int(x + d + 1))

            region_y = Y_grid[y_min:y_max, 0]
            region_x = X_grid[0, x_min:x_max]

            dist_sq = (region_y[:, np.newaxis] - y) ** 2 + (region_x - x) ** 2
            forbidden_mask[y_min:y_max, x_min:x_max] |= (dist_sq < d**2)

        # If we run out of candidates, resample from the remaining valid region.
        if idx_ptr >= batch_size and count < total_points:
            current_weight = weight * (1 - existing_mask) * (1 - forbidden_mask)
            sw = current_weight.sum()
            if sw <= 0:
                break
            prob = current_weight.ravel() / sw
            indices = np.random.choice(width * height, size=batch_size, p=prob)
            idx_ptr = 0

    return np.array(sampled_points)


def fun_mask_gen_2d(
    mask_size,
    total_points,
    pattern_num,
    sigma_x,
    sigma_y,
    min_dist_factor=3,
    rep_decay_factor=0.5,
    center_radius_x=0.5,
    center_radius_y=0.5,
):
    """
    Generate kt-Gaussian-like undersampling masks on a 2D (PE, SPE) k-space grid.

    The method:
    1) Defines a Gaussian sampling density (higher near the center).
    2) Forces a fully-sampled central region (ellipse). If the radii are small, falls back to a
       single center point.
    3) For each pattern, samples additional points with a spatial minimum-distance constraint.
    4) Optionally reduces repeated sampling across patterns by decaying weights at selected points.

    Parameters
    ----------
    mask_size : (int, int)
        (width, height) = (PE, SPE) grid size.
    total_points : int
        Target number of sampled points per mask (including the fully-sampled center).
    pattern_num : int
        Number of masks (patterns) to generate (often equals Nt).
    sigma_x, sigma_y : float
        Gaussian standard deviations along PE (x) and SPE (y).
    min_dist_factor : float, default=3
        Scales the per-location exclusion radius derived from the Gaussian weight map.
        Set to 0 to disable the minimum-distance constraint.
    rep_decay_factor : float, default=0.5
        Multiplicative decay applied to weights at selected locations across patterns.
        - 1.0 disables repetition control.
        - < 1.0 discourages repeatedly sampling the same locations.
    center_radius_x, center_radius_y : float, default=0.5
        Radii of the fully-sampled central ellipse along x and y, in pixel units of the grid
        coordinate system used below.

    Returns
    -------
    masks : np.ndarray, shape (height, width, pattern_num), dtype float32
        Binary masks stored as (SPE, PE, pattern_index).

    Notes
    -----
    If you enable the ellipse branch, you must define X and Y (grid coordinates) before using them,
    e.g. via `np.meshgrid`. Otherwise this function will raise a NameError.
    """
    width, height = mask_size
    masks = np.zeros((height, width, pattern_num), dtype=np.float32)

    initial_weight = create_gaussian_weight_matrix(mask_size, sigma_x, sigma_y)
    weight = initial_weight.copy()

    # Larger exclusion radii in low-weight (outer) regions, smaller near the center.
    min_dist_lookup = min_dist_factor * ((1.0 - initial_weight) / 2.0 + 0.5)

    # Define the fully-sampled center region.
    if (center_radius_x <= 0.5) or (center_radius_y <= 0.5):
        # Fallback: only sample the exact center point.
        center_ellipse = np.zeros((height, width), dtype=bool)
        cy = int(round((height - 1) / 2))
        cx = int(round((width - 1) / 2))
        center_ellipse[cy, cx] = True
    else:
        # Fully-sampled central ellipse.
        center_ellipse = (
            ((width - (width - 1) / 2) / center_radius_x) ** 2
            + ((height - (height - 1) / 2) / center_radius_y) ** 2
            <= 1
        )

    num_center_points = int(np.sum(center_ellipse))

    for p in range(pattern_num):
        mask = np.zeros((height, width), dtype=np.float32)

        # Always include the fully-sampled center region.
        mask[center_ellipse] = 1

        # First pass: attempt to reach target count.
        needed = total_points - num_center_points
        if needed > 0:
            points = random_sampling_optimized(mask_size, needed, weight, min_dist_lookup, mask)
            if len(points) > 0:
                xs = points[:, 0].astype(int) - 1
                ys = points[:, 1].astype(int) - 1
                mask[ys, xs] = 1
                weight[ys, xs] *= rep_decay_factor

        # Second pass: if still short, try to fill remaining points.
        curr_total = int(np.sum(mask))
        if curr_total < total_points:
            extra = total_points - curr_total
            extra_points = random_sampling_optimized(mask_size, extra, weight, min_dist_lookup, mask)
            if len(extra_points) > 0:
                xs = extra_points[:, 0].astype(int) - 1
                ys = extra_points[:, 1].astype(int) - 1
                mask[ys, xs] = 1
                weight[ys, xs] *= rep_decay_factor

        masks[:, :, p] = mask

    return masks