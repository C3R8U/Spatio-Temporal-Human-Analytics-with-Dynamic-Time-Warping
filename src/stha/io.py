from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SkeletonSequence:
    data: np.ndarray  # (T, J, 3)
    fps: float = 60.0


def load_skeleton_csv(path: str, joints: int) -> SkeletonSequence:
    """Load skeleton from CSV with columns: frame,joint,x,y,z."""
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    frames = arr["frame"].astype(int)
    joint = arr["joint"].astype(int)
    x, y, z = arr["x"].astype(float), arr["y"].astype(float), arr["z"].astype(float)

    T = frames.max() + 1
    J = joints
    data = np.zeros((T, J, 3), dtype=np.float32)
    data[frames, joint, 0] = x
    data[frames, joint, 1] = y
    data[frames, joint, 2] = z
    return SkeletonSequence(data=data, fps=60.0)


def resample_skeleton(seq: SkeletonSequence, target_fps: float) -> SkeletonSequence:
    if abs(seq.fps - target_fps) < 1e-6:
        return seq
    T, J, C = seq.data.shape
    duration = (T - 1) / seq.fps
    new_T = int(round(duration * target_fps)) + 1
    t_old = np.linspace(0.0, duration, T)
    t_new = np.linspace(0.0, duration, new_T)

    out = np.empty((new_T, J, C), dtype=np.float32)
    for j in range(J):
        for c in range(C):
            out[:, j, c] = np.interp(t_new, t_old, seq.data[:, j, c]).astype(np.float32)
    return SkeletonSequence(data=out, fps=target_fps)


def normalize_skeleton(
    seq: SkeletonSequence,
    root_joint: int,
    left_shoulder: int,
    right_shoulder: int,
    eps: float = 1e-8,
) -> SkeletonSequence:
    data = seq.data.copy()
    root = data[:, root_joint:root_joint+1, :]
    data = data - root

    shoulder_vec = data[:, right_shoulder, :] - data[:, left_shoulder, :]
    shoulder_width = np.linalg.norm(shoulder_vec, axis=1)
    scale = float(np.median(shoulder_width))
    scale = max(scale, eps)
    data = data / scale
    return SkeletonSequence(data=data.astype(np.float32), fps=seq.fps)


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    w = min(window, len(x))
    k = w // 2
    y = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i - k)
        hi = min(len(x), i + k + 1)
        y[i] = x[lo:hi].mean()
    return y


def _try_savgol(x: np.ndarray, window: int, polyorder: int) -> np.ndarray:
    try:
        from scipy.signal import savgol_filter
        w = min(window, len(x) if len(x) % 2 == 1 else len(x) - 1)
        if w < 3:
            return x
        if w % 2 == 0:
            w -= 1
        p = min(polyorder, w - 1)
        return savgol_filter(x, window_length=w, polyorder=p, mode="interp")
    except Exception:
        return _moving_average(x, window)


def smooth_skeleton(seq: SkeletonSequence, method: str = "savgol", window: int = 11, polyorder: int = 3) -> SkeletonSequence:
    if method == "none":
        return seq
    data = seq.data.copy()
    T, J, C = data.shape
    for j in range(J):
        for c in range(C):
            x = data[:, j, c]
            if method == "savgol":
                data[:, j, c] = _try_savgol(x, window, polyorder)
            elif method == "moving_average":
                data[:, j, c] = _moving_average(x, window)
            else:
                raise ValueError(f"Unknown smoothing method: {method}")
    return SkeletonSequence(data=data.astype(np.float32), fps=seq.fps)
