"""Minimal physical shape descriptors from original NIfTI voxel cells."""
from itertools import product

import nibabel as nib
import numpy as np


def shape_descriptor(mask: np.ndarray, affine: np.ndarray) -> dict:
    """Measure size, centroid and SI extent in NIfTI RAS+ world coordinates.

    Centroids average foreground voxel centres. SI extent is the world-z span
    of complete occupied voxel cells, including gaps, not a centreline length.
    Normalized SI centroid uses inferior/superior scan voxel-centre limits
    (0/1), not anatomical registration. The affine handles axis flips,
    permutations and obliquity. The caller must validate millimetre units.
    """
    affine = np.asarray(affine, dtype=float)
    if mask.ndim != 3 or affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("Expected a 3D mask and finite 4x4 affine")
    voxel_volume = abs(float(np.linalg.det(affine[:3, :3])))
    if voxel_volume <= 0:
        raise ValueError("Degenerate spatial affine")
    coordinates = np.nonzero(mask)
    count = len(coordinates[0])
    corners = np.array(list(product(*[(0, n - 1) for n in mask.shape])))
    scan_z = nib.affines.apply_affine(affine, corners)[:, 2]
    inferior, superior = float(scan_z.min()), float(scan_z.max())
    centroid = np.array([a.mean() for a in coordinates]) if count else np.full(3, np.nan)
    world = nib.affines.apply_affine(affine, centroid)
    if count:
        occupied_z = sum(affine[2, axis] * coordinates[axis] for axis in range(3)) + affine[2, 3]
        # Project the full affine-transformed unit voxel cell onto world SI.
        cell_thickness = float(np.abs(affine[2, :3]).sum())
        extent = float(occupied_z.max() - occupied_z.min() + cell_thickness)
        normalized = float((world[2] - inferior) / (superior - inferior)) if superior > inferior else .5
    else:
        extent, normalized = np.nan, np.nan
    return {"voxel_count": count, "volume_mm3": count * voxel_volume,
            "volume_ml": count * voxel_volume / 1000,
            "centroid_i": centroid[0], "centroid_j": centroid[1], "centroid_k": centroid[2],
            "centroid_world_x_mm": world[0], "centroid_world_y_mm": world[1],
            "centroid_world_z_mm": world[2], "normalized_si_centroid": normalized,
            "si_extent_mm": extent, "axis_codes": "".join(nib.aff2axcodes(affine)),
            "voxel_volume_mm3": voxel_volume, "scan_inferior_center_z_mm": inferior,
            "scan_superior_center_z_mm": superior}
