from __future__ import annotations
import numpy as np


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def compute_torso_normal(skel: np.ndarray, left_shoulder: int, right_shoulder: int, left_hip: int, right_hip: int) -> np.ndarray:
    s = skel[:, right_shoulder, :] - skel[:, left_shoulder, :]
    h = skel[:, right_hip, :] - skel[:, left_hip, :]
    s = _normalize(s)
    h = _normalize(h)
    n = np.cross(s, h)
    return _normalize(n)


def rotation_angle_from_normals(normals: np.ndarray, min_proj_norm: float = 1e-6) -> dict:
    zhat = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    proj = normals.copy()
    proj[:, 2] = 0.0
    proj_norm = np.linalg.norm(proj, axis=1)
    valid = proj_norm >= float(min_proj_norm)
    proj[valid] = proj[valid] / proj_norm[valid, None]

    deltas = []
    ignored = 0
    for t in range(len(normals) - 1):
        if not (valid[t] and valid[t + 1]):
            ignored += 1
            deltas.append(0.0)
            continue
        a = proj[t]
        b = proj[t + 1]
        num = float(np.dot(zhat, np.cross(a, b)))
        den = float(np.dot(a, b))
        deltas.append(float(np.arctan2(num, den)))

    deltas = np.array(deltas, dtype=np.float32)
    unwrapped = np.unwrap(deltas.astype(np.float64)).astype(np.float32)
    total_deg = float(unwrapped.sum() * 180.0 / np.pi)

    return {
        "total_deg": total_deg,
        "deltas_deg": unwrapped * (180.0 / np.pi),
        "ignored_frames": int(ignored),
        "valid_ratio": float(valid.mean()),
    }
