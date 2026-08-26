#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("motion", type=Path)
args = parser.parse_args()

with np.load(args.motion, allow_pickle=False) as data:
    required = (
        "fps", "joint_pos", "joint_vel", "body_pos_w",
        "body_quat_w", "body_lin_vel_w", "body_ang_vel_w",
    )
    missing = [name for name in required if name not in data.files]
    if missing:
        raise RuntimeError(f"Missing arrays: {missing}")

    n = data["joint_pos"].shape[0]
    expected = {
        "joint_pos": (n, 29),
        "joint_vel": (n, 29),
        "body_pos_w": (n, 30, 3),
        "body_quat_w": (n, 30, 4),
        "body_lin_vel_w": (n, 30, 3),
        "body_ang_vel_w": (n, 30, 3),
    }
    for name, shape in expected.items():
        if data[name].shape != shape:
            raise RuntimeError(
                f"{name}: expected {shape}, got {data[name].shape}"
            )
        if not np.isfinite(data[name]).all():
            raise RuntimeError(f"{name} contains NaN/Inf")

    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    quat_norm = np.linalg.norm(data["body_quat_w"], axis=-1)
    print("Validation: PASS")
    print(f"frames={n}")
    print(f"fps={fps:g}")
    print(f"duration={(n-1)/fps:.2f}s")
    print(f"joint_pos={data['joint_pos'].shape}")
    print(f"body_pos_w={data['body_pos_w'].shape}")
    print(
        "quaternion norm error="
        f"{np.max(np.abs(quat_norm-1.0)):.3e}"
    )
