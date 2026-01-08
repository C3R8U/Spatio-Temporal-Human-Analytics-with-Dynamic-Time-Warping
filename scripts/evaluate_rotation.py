from __future__ import annotations
import argparse

from stha import load_skeleton_csv
from stha.io import resample_skeleton, normalize_skeleton, smooth_skeleton
from stha.geometry import compute_torso_normal, rotation_angle_from_normals
from scripts._utils import load_cfg, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--input", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    set_seed(int(cfg.get("seed", 2024)))

    J = int(cfg["data"]["joints"])
    seq = load_skeleton_csv(args.input, joints=J)
    seq = resample_skeleton(seq, target_fps=float(cfg["data"]["target_fps"]))

    sm = cfg.get("smoothing", {})
    seq = smooth_skeleton(seq, method=sm.get("method", "savgol"),
                          window=int(sm.get("window", 11)),
                          polyorder=int(sm.get("polyorder", 3)))

    seq = normalize_skeleton(seq, root_joint=0, left_shoulder=5, right_shoulder=6)

    rot_cfg = cfg["rotation"]
    normals = compute_torso_normal(
        seq.data,
        left_shoulder=int(rot_cfg["left_shoulder"]),
        right_shoulder=int(rot_cfg["right_shoulder"]),
        left_hip=int(rot_cfg["left_hip"]),
        right_hip=int(rot_cfg["right_hip"]),
    )
    out = rotation_angle_from_normals(normals, min_proj_norm=float(rot_cfg["min_proj_norm"]))

    print("Rotation angle estimation (geometric)")
    print(f"Total rotation (deg): {out['total_deg']:.3f}")
    print(f"Ignored frames:       {out['ignored_frames']}")
    print(f"Valid ratio:          {out['valid_ratio']:.3f}")


if __name__ == "__main__":
    main()
