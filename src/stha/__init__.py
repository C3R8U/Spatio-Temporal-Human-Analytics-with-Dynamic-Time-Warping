"""STHA Sports Analytics: reproducible reference implementation."""

from .io import load_skeleton_csv, resample_skeleton, normalize_skeleton, smooth_skeleton
from .geometry import compute_torso_normal, rotation_angle_from_normals
from .dtw import dtw_distance
from .scoring import gymnastics_score

__all__ = [
    "load_skeleton_csv",
    "resample_skeleton",
    "normalize_skeleton",
    "smooth_skeleton",
    "compute_torso_normal",
    "rotation_angle_from_normals",
    "dtw_distance",
    "gymnastics_score",
]
