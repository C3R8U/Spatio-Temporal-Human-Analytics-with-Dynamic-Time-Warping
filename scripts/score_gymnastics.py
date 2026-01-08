from __future__ import annotations
import argparse
import numpy as np

from stha import load_skeleton_csv
from stha.io import resample_skeleton, normalize_skeleton, smooth_skeleton
from stha.scoring import gymnastics_score
from scripts._utils import load_cfg, set_seed


def frame_features(skel: np.ndarray) -> np.ndarray:
    T, J, C = skel.shape
    return skel.reshape(T, J * C).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--template", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    set_seed(int(cfg.get("seed", 2024)))

    J = int(cfg["data"]["joints"])
    sm = cfg.get("smoothing", {})

    def load_prepare(p: str):
        seq = load_skeleton_csv(p, joints=J)
        seq = resample_skeleton(seq, target_fps=float(cfg["data"]["target_fps"]))
        seq = smooth_skeleton(seq, method=sm.get("method", "savgol"),
                              window=int(sm.get("window", 11)),
                              polyorder=int(sm.get("polyorder", 3)))
        seq = normalize_skeleton(seq, root_joint=0, left_shoulder=5, right_shoulder=6)
        return seq

    test = load_prepare(args.test)
    templ = load_prepare(args.template)

    out = gymnastics_score(
        test_feat=frame_features(test.data),
        template_feat=frame_features(templ.data),
        base_score=float(cfg["scoring"]["base_score"]),
        alpha=float(cfg["scoring"]["alpha"]),
        band_ratio=float(cfg["dtw"]["band_ratio"]),
    )

    print("Gymnastics scoring (DTW-aligned penalty)")
    print(f"Score:        {out['score']:.3f}")
    print(f"Penalty(avg): {out['penalty']:.6f}")
    print(f"DTW distance: {out['dtw_distance']:.3f}")
    print(f"Path length:  {out['dtw_path_length']}")


if __name__ == "__main__":
    main()
