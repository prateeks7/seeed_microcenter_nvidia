"""Wrist-camera check for whether the gripper is holding something.

The camera is mounted on the gripper (eye-in-hand), so the gap between the jaws always
appears at the same place in the image. After lifting, a held object fills that gap a few
centimetres from the camera; an empty gripper shows the table or scene far behind it.

check_object_in_gripper(): look into the jaw gap in a depth image and decide held / empty.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class HeldCheck:
    held: Optional[bool]               # None = inconclusive
    detail: str
    roi_xyxy: tuple[int, int, int, int] = (0, 0, 0, 0)


def _project(K: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.array([K[0, 0] * p[0] / p[2] + K[0, 2], K[1, 1] * p[1] / p[2] + K[1, 2]])


def check_object_in_gripper(
    depth_mm: np.ndarray,
    K: np.ndarray,
    T_cam2gripper: np.ndarray,
    jaw_gap_m: Optional[float],
    near_margin_m: float = 0.03,
    reach_m: Optional[float] = None,
    min_gap_m: float = 0.015,
) -> HeldCheck:
    """Decide whether something sits between the jaws.

    Samples a strip in the middle of the jaw gap (40% of the gap wide, so the fingers
    themselves stay out of it) running from the gripper point out along the approach
    direction. The grasp aims the gripper point at the object's near surface, so a held
    object extends from there by about its own size: reach_m defaults to the jaw gap.

    Empty jaws show the table well behind (valid, far depth). A held object is near, or
    closer than the depth camera's minimum range (no depth), so:
      mostly far             -> empty
      mostly near / no depth -> held
      otherwise              -> inconclusive (held=None)
    """
    if depth_mm is None:
        return HeldCheck(None, "no depth frame")
    if jaw_gap_m is None or jaw_gap_m < min_gap_m:
        return HeldCheck(None, "jaws nearly closed; cannot look between them")

    T_g2c = np.linalg.inv(np.asarray(T_cam2gripper, dtype=np.float64))
    center = T_g2c[:3, 3]                     # gripper point in the camera frame
    if center[2] <= 0.02:
        return HeldCheck(None, "gripper point is not in front of the camera")
    approach_axis = T_g2c[:3, 0]              # TCP X = tool-forward / approach direction
    open_axis = T_g2c[:3, 1]                  # TCP Y = jaw opening direction
    if reach_m is None:
        reach_m = float(np.clip(jaw_gap_m, 0.02, 0.08))
    far = center + approach_axis * reach_m    # where the far side of a held object can be
    if far[2] <= 0.02:
        return HeldCheck(None, "approach direction leaves the camera view")
    tip_a = _project(K, center + open_axis * jaw_gap_m / 2)
    tip_b = _project(K, center - open_axis * jaw_gap_m / 2)
    p_near, p_far = _project(K, center), _project(K, far)

    # Narrow across the gap (40% of it, clear of the fingers); along the approach, cover
    # from the gripper point to reach_m beyond it, plus a small pad.
    du, dv = np.abs(tip_a - tip_b)
    pad_px = K[1, 1] * 0.01 / center[2]
    xs, ys = sorted((p_near[0], p_far[0])), sorted((p_near[1], p_far[1]))
    if du >= dv:
        half_gap = max(0.2 * du, 2.0)
        mid = (p_near[0] + p_far[0]) / 2
        x_lo, x_hi, y_lo, y_hi = mid - half_gap, mid + half_gap, ys[0] - pad_px, ys[1] + pad_px
    else:
        half_gap = max(0.2 * dv, 2.0)
        mid = (p_near[1] + p_far[1]) / 2
        x_lo, x_hi, y_lo, y_hi = xs[0] - pad_px, xs[1] + pad_px, mid - half_gap, mid + half_gap

    h, w = depth_mm.shape[:2]
    x1, x2 = int(max(x_lo, 0)), int(min(x_hi, w))
    y1, y2 = int(max(y_lo, 0)), int(min(y_hi, h))
    u, v = p_near
    roi_xyxy = (x1, y1, x2, y2)
    roi = depth_mm[y1:y2, x1:x2]
    if roi.size == 0:
        return HeldCheck(None, f"jaw gap projects outside the image (pixel {u:.0f},{v:.0f})", roi_xyxy)

    near_limit_mm = (max(center[2], far[2]) + near_margin_m) * 1000.0
    valid = roi > 0
    near = valid & (roi < near_limit_mm)
    near_frac = float(near.mean())
    far_frac = float((valid & ~near).mean())
    blind_frac = 1.0 - float(valid.mean())    # no depth: usually too close to measure
    summary = (f"jaw gap: {near_frac:.0%} near, {far_frac:.0%} far (beyond {near_limit_mm / 10:.0f} cm), "
               f"{blind_frac:.0%} no depth")
    if far_frac >= 0.5:
        return HeldCheck(False, f"{summary} -> gap is empty (table visible)", roi_xyxy)
    if near_frac + blind_frac >= 0.6:
        how = "too close to measure" if blind_frac > near_frac else "close to the camera"
        return HeldCheck(True, f"{summary} -> object between jaws ({how})", roi_xyxy)
    return HeldCheck(None, f"{summary} -> unclear", roi_xyxy)
