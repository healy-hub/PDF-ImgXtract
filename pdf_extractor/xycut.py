from typing import List, Tuple
import numpy as np

def projection_by_bboxes(boxes: np.ndarray, axis: int) -> np.ndarray:
    """
    Get projection profile by bounding boxes.
    Args:
        boxes: [N, 4] (x1, y1, x2, y2)
        axis: 0 for X-axis (vertical projection), 1 for Y-axis (horizontal projection)
    """
    assert axis in [0, 1]
    if boxes.size == 0:
        return np.zeros(0, dtype=int)
        
    length = np.max(boxes[:, axis::2])
    res = np.zeros(length, dtype=int)
    for start, end in boxes[:, axis::2]:
        start = max(0, int(start))
        end = min(length, int(end))
        if start < end:
            res[start:end] += 1
    return res

def split_projection_profile(arr_values: np.ndarray, min_value: float, min_gap: float):
    """
    Split projection profile.
    Returns (start_indices, end_indices) arrays.
    """
    arr_index = np.where(arr_values > min_value)[0]
    if not len(arr_index):
        return None

    arr_diff = arr_index[1:] - arr_index[0:-1]
    arr_diff_index = np.where(arr_diff > min_gap)[0]
    arr_zero_intvl_start = arr_index[arr_diff_index]
    arr_zero_intvl_end = arr_index[arr_diff_index + 1]

    arr_start = np.insert(arr_zero_intvl_end, 0, arr_index[0])
    arr_end = np.append(arr_zero_intvl_start, arr_index[-1])
    arr_end += 1

    return arr_start, arr_end

def recursive_xy_cut(boxes: np.ndarray, indices: List[int], res: List[int]):
    """
    Recursive XY Cut algorithm tailored for Document Layout Analysis.
    Prioritizes Vertical Cuts (Columns) over Horizontal Cuts (Rows) to handle dual-column layouts correctly.
    
    Args:
        boxes: (N, 4) [x1, y1, x2, y2]
        indices: Original indices of the boxes
        res: Result list to append sorted indices to
    """
    if len(boxes) == 0:
        return

    # 1. Try splitting by X-axis (Vertical Columns) first
    # We sort boxes by X to ensure cleaner projection handling if needed, 
    # though projection_by_bboxes handles unsorted.
    # But for the recursive call, we need to partition properly.
    
    x_projection = projection_by_bboxes(boxes, axis=0)
    pos_x = split_projection_profile(x_projection, 0, 1)

    # Check if we found valid splits (more than 1 group)
    if pos_x is not None and len(pos_x[0]) > 1:
        arr_x0, arr_x1 = pos_x
        
        # Sort by X to group them effectively for the next recursion or just iterate ranges
        # Actually, we filter boxes based on the split ranges.
        # We must process columns from Left to Right.
        
        # To ensure correct order, we rely on the fact that split_projection_profile returns sorted ranges.
        
        for c0, c1 in zip(arr_x0, arr_x1):
            # Filter boxes that fall significantly into this column range
            # Strict containment might be too harsh if boxes overlap slightly,
            # but usually XY cut assumes clear gaps.
            # We look for boxes whose center x is within [c0, c1) or just intersection.
            # Standard XY cut usually partitions based on the gap.
            
            # Criteria: Box center is within the band? Or Box overlaps the band?
            # Since we split by *gaps*, the boxes should be entirely within the bands (mostly).
            
            # Calculate centers
            box_centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
            mask = (box_centers_x >= c0) & (box_centers_x < c1)
            
            if np.any(mask):
                recursive_xy_cut(boxes[mask], indices[mask], res)
        return

    # 2. If X-split failed (Single Column), try splitting by Y-axis (Rows)
    y_projection = projection_by_bboxes(boxes, axis=1)
    pos_y = split_projection_profile(y_projection, 0, 1)

    if pos_y is not None and len(pos_y[0]) > 1:
        arr_y0, arr_y1 = pos_y
        
        # Process rows Top to Bottom
        for r0, r1 in zip(arr_y0, arr_y1):
            box_centers_y = (boxes[:, 1] + boxes[:, 3]) / 2
            mask = (box_centers_y >= r0) & (box_centers_y < r1)
            
            if np.any(mask):
                recursive_xy_cut(boxes[mask], indices[mask], res)
        return

    # 3. If both splits failed, we have a leaf node (indivisible block or cluster).
    # Sort these items by standard reading order: Top-to-Bottom, then Left-to-Right.
    # Note: Since we couldn't split, they likely overlap or are very close.
    
    # Sort criteria: y1 (top) major, x1 (left) minor
    # Zip indices with boxes to sort
    combined = list(zip(indices, boxes))
    # Sort by y1, then x1
    combined.sort(key=lambda x: (x[1][1], x[1][0]))
    
    sorted_indices = [x[0] for x in combined]
    res.extend(sorted_indices)