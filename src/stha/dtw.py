from __future__ import annotations
import numpy as np


def dtw_distance(X: np.ndarray, Y: np.ndarray, band_ratio: float = 0.10) -> dict:
    T, D = X.shape
    U, Dy = Y.shape
    assert D == Dy

    w = int(np.ceil(max(T, U) * float(band_ratio)))
    w = max(w, abs(T - U))

    inf = 1e18
    dp = np.full((T + 1, U + 1), inf, dtype=np.float64)
    dp[0, 0] = 0.0
    ptr = np.full((T + 1, U + 1), -1, dtype=np.int8)  # 0 diag, 1 up, 2 left

    for i in range(1, T + 1):
        j_start = max(1, i - w)
        j_end = min(U, i + w)
        xi = X[i - 1]
        for j in range(j_start, j_end + 1):
            yj = Y[j - 1]
            cost = float(np.sum((xi - yj) ** 2))
            cand = (dp[i - 1, j - 1], dp[i - 1, j], dp[i, j - 1])
            k = int(np.argmin(cand))
            dp[i, j] = cost + cand[k]
            ptr[i, j] = k

    dist = float(dp[T, U])

    i, j = T, U
    path = []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        k = int(ptr[i, j])
        if k == 0:
            i -= 1; j -= 1
        elif k == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()

    return {
        "distance": dist,
        "path_length": int(len(path)),
        "path": path,
        "avg_cost": float(dist / max(1, len(path))),
    }
