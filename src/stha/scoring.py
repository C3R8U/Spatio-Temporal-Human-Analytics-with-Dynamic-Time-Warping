from __future__ import annotations
import numpy as np
from .dtw import dtw_distance


def gymnastics_score(test_feat: np.ndarray, template_feat: np.ndarray, base_score: float = 10.0, alpha: float = 0.8, band_ratio: float = 0.10) -> dict:
    dtw = dtw_distance(test_feat, template_feat, band_ratio=band_ratio)
    penalty = float(dtw["avg_cost"])
    score = float(base_score - alpha * penalty)
    return {
        "score": score,
        "penalty": penalty,
        "dtw_distance": float(dtw["distance"]),
        "dtw_path_length": int(dtw["path_length"]),
    }
