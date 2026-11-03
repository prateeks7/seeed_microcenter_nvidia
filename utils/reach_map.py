"""Reach zones on the table, drawn on the wrist-camera view at the ready pose.

Uses the same SDK IK solver and the same grasp-approach rules as the grasp code to decide,
on a grid of table points, whether the arm can:
  pick:  reach the pregrasp and grasp poses (approach along the camera line of sight,
         checked for two wrist twists so object orientation does not matter), and
  place: reach the release pose above the point (set.py releases at roll 0, pitch 1.2, yaw 0).

ReachMap(...).compute_all(): blocking; call before the arm control loop starts.
ReachMap.request_place_height(): call every frame with the wanted release height; the place
zone is recomputed (in a background thread) only for a lasting change, never for depth noise.
ReachMap.draw(): draw the zones on the camera image: green tint = pick (fixed for the run),
blue stripes + outline = place. The green tint and outline never change during a run; where
the zones overlap the stripes lie on top, instead of re-tinting the pick zone.
ReachMap.pose_reachable(): exact IK check of one pose, for checking a task before moving.
"""
from __future__ import annotations

import threading
import time
from typing import Optional, Sequence

import cv2
import numpy as np
from reBotArm_control_py.kinematics import (
    compute_fk,
    get_end_effector_frame_id,
    load_robot_model,
    pad_q_for_model,
    pos_rot_to_se3,
    solve_ik,
)
from reBotArm_control_py.kinematics.inverse_kinematics import IKParams

try:
    from .transforms import grasp_axes_to_rebot_tcp_rotation, transform_grasp_pose_to_base
except ImportError:
    from transforms import grasp_axes_to_rebot_tcp_rotation, transform_grasp_pose_to_base

# Same solver settings RebotArmEndPose.move_to_traj uses.
_IK_PARAMS = IKParams(max_iter=200, tolerance=1e-4, step_size=0.5, damping=1e-6)
PLACE_RPY = (0.0, 1.2, 0.0)
_PICK_COLOR = (80, 220, 80)     # BGR green
_PLACE_COLOR = (230, 150, 40)   # BGR blue
PLACE_STEP_M = 0.03     # the place zone follows release-height changes of at least this much...
PLACE_HOLD_S = 1.5      # ...that last this long (a missed detection or depth noise does not)


class ReachMap:
    def __init__(
        self,
        K: np.ndarray,
        image_size: tuple[int, int],
        T_hand_eye: np.ndarray,
        T_compensation: np.ndarray,
        ready_pose: dict,
        n_joints: int = 6,
        pregrasp_offset_m: float = 0.08,
        pick_height_m: float = 0.03,
        cell_m: float = 0.025,
        x_range: tuple[float, float] = (0.05, 0.75),
        y_range: tuple[float, float] = (-0.50, 0.50),
    ) -> None:
        self._model = load_robot_model()
        self._frame_id = get_end_effector_frame_id(self._model)
        self._data_main = self._model.createData()     # for calls from the caller's thread
        self._data_worker = self._model.createData()   # for the background thread
        self._n = n_joints
        self._K = np.asarray(K, dtype=np.float64)
        self._w, self._h = image_size
        self._pregrasp = float(pregrasp_offset_m)
        self._pick_z = float(pick_height_m)

        # Where the camera is at the ready pose, from the ready pose's IK solution.
        rp = ready_pose
        self.q_ready = self._solve(
            (rp.get("x", 0.25), rp.get("y", 0.0), rp.get("z", 0.35),
             rp.get("roll", 0.0), rp.get("pitch", 1.2), rp.get("yaw", 0.0)),
            np.zeros(n_joints), self._data_main)
        if self.q_ready is None:
            raise RuntimeError("Ready pose has no IK solution")
        pos, rot, _ = compute_fk(self._model, pad_q_for_model(self._model, self.q_ready, n_joints))
        T_tcp2base = np.eye(4)
        T_tcp2base[:3, :3] = rot
        T_tcp2base[:3, 3] = pos
        self.T_cam2base = np.asarray(T_compensation) @ T_tcp2base @ np.asarray(T_hand_eye)
        self._T_base2cam = np.linalg.inv(self.T_cam2base)

        # Grid cells whose table-level center is in view; keep their projected outlines.
        half = cell_m / 2
        self._cells: list[tuple[float, float]] = []
        self._polys: list[np.ndarray] = []
        for x in np.arange(x_range[0], x_range[1] + 1e-9, cell_m):
            for y in np.arange(y_range[0], y_range[1] + 1e-9, cell_m):
                corners = [(x - half, y - half), (x + half, y - half), (x + half, y + half), (x - half, y + half)]
                px = [self._project((cx, cy, 0.0)) for cx, cy in corners]
                center = self._project((x, y, 0.0))
                if center is None or any(p is None for p in px):
                    continue
                if not (0 <= center[0] < self._w and 0 <= center[1] < self._h):
                    continue
                self._cells.append((float(x), float(y)))
                self._polys.append(np.round(np.array(px)).astype(np.int32))

        self._pick_ok = np.zeros(len(self._cells), dtype=bool)
        self._place_ok = np.zeros(len(self._cells), dtype=bool)
        self._place_z: Optional[float] = None
        self._masks_dirty = True
        self._pick_mask = np.zeros((self._h, self._w), dtype=np.uint8)
        self._place_mask = np.zeros((self._h, self._w), dtype=np.uint8)
        self._contours: list[tuple[list, tuple[int, int, int]]] = []
        self._color_layer = np.zeros((self._h, self._w, 3), dtype=np.uint8)
        yy, xx = np.indices((self._h, self._w))
        self._stripes = ((xx + yy) // 8) % 3 == 0          # diagonal hatching for the place zone
        self._region = np.zeros((self._h, self._w), dtype=np.uint8)
        self._lock = threading.Lock()
        self._wanted_place_z: Optional[float] = None
        self._pending_since: Optional[float] = None
        self._wake = threading.Event()
        self._running = threading.Event()
        self._running.set()
        self._worker: Optional[threading.Thread] = None

    # ---- geometry / IK ----

    def _project(self, p_base: Sequence[float]) -> Optional[tuple[float, float]]:
        p = self._T_base2cam @ np.array([p_base[0], p_base[1], p_base[2], 1.0])
        if p[2] <= 0.05:
            return None
        return (self._K[0, 0] * p[0] / p[2] + self._K[0, 2], self._K[1, 1] * p[1] / p[2] + self._K[1, 2])

    def _solve(self, pose6d: Sequence[float], seed: np.ndarray, data) -> Optional[np.ndarray]:
        x, y, z, roll, pitch, yaw = (float(v) for v in pose6d)
        target = pos_rot_to_se3(np.array([x, y, z]), roll=roll, pitch=pitch, yaw=yaw)
        result = solve_ik(self._model, data, self._frame_id, target, seed, _IK_PARAMS, controlled_joints=self._n)
        return np.asarray(result.q)[: self._n] if result.success else None

    def pose_reachable(self, pose6d: Sequence[float], seed: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Joint solution for a pose (seeded from the ready pose by default), or None if unreachable."""
        return self._solve(pose6d, self.q_ready if seed is None else seed, self._data_main)

    def _pick_reachable(self, x: float, y: float, data) -> bool:
        p_cam = (self._T_base2cam @ np.array([x, y, self._pick_z, 1.0]))[:3]
        approach = -p_cam / np.linalg.norm(p_cam)          # as estimate_grasp: object -> camera
        for hint in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])):   # two wrist twists
            open_axis = hint - float(np.dot(hint, approach)) * approach
            open_axis /= np.linalg.norm(open_axis)
            if open_axis[0] < 0:
                open_axis = -open_axis
            grip = np.cross(open_axis, approach)
            grip /= np.linalg.norm(grip)
            open_axis = np.cross(approach, grip)
            tcp_rot = grasp_axes_to_rebot_tcp_rotation(grip, open_axis, approach)
            grasp6d, pre6d = transform_grasp_pose_to_base(p_cam, tcp_rot, self.T_cam2base, self._pregrasp)
            q_pre = self._solve(pre6d, self.q_ready, data)
            if q_pre is None or self._solve(grasp6d, q_pre, data) is None:
                return False
        return True

    # ---- zone computation ----

    def compute_all(self, place_height_m: float) -> None:
        """Compute both zones now (blocking). Call before the arm control loop starts."""
        t = time.perf_counter()
        pick = np.array([self._pick_reachable(x, y, self._data_main) for x, y in self._cells], dtype=bool)
        place = self._compute_place(place_height_m, self._data_main, yield_s=0.0)
        with self._lock:
            self._pick_ok, self._place_ok, self._place_z = pick, place, place_height_m
            self._masks_dirty = True
        print(f"[Reach] {pick.sum()}/{len(self._cells)} cells pickable, {place.sum()} placeable "
              f"at {place_height_m * 100:.0f} cm ({time.perf_counter() - t:.1f} s)")

    def _compute_place(self, z: float, data, yield_s: float) -> Optional[np.ndarray]:
        ok = np.zeros(len(self._cells), dtype=bool)
        for i, (x, y) in enumerate(self._cells):
            self._running.wait()                      # paused while the arm moves
            if self._wanted_place_z is not None and self._wanted_place_z != z:
                return None                           # a newer height was requested
            ok[i] = self._solve((x, y, z, *PLACE_RPY), self.q_ready, data) is not None
            if yield_s:
                time.sleep(yield_s)                   # let the arm control thread run
        return ok

    def request_place_height(self, z: float) -> None:
        """Call every frame with the wanted release height. The place zone is recomputed only when
        the height stays at least PLACE_STEP_M away from the zone's height for PLACE_HOLD_S."""
        current = self._wanted_place_z if self._wanted_place_z is not None else self._place_z
        if current is not None and abs(z - current) < PLACE_STEP_M:
            self._pending_since = None
            return
        now = time.monotonic()
        if self._pending_since is None:
            self._pending_since = now
        elif now - self._pending_since >= PLACE_HOLD_S:
            self._pending_since = None
            self.set_place_height(round(z, 2))

    def set_place_height(self, z: float) -> None:
        """Recompute the place zone for a new release height, in the background."""
        if z == self._wanted_place_z or (self._wanted_place_z is None and z == self._place_z):
            return                                    # already computed or in progress
        self._wanted_place_z = z
        if self._worker is None:
            self._worker = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker.start()
        self._wake.set()

    def _worker_loop(self) -> None:
        while True:
            self._wake.wait()
            self._wake.clear()
            z = self._wanted_place_z
            if z is None:
                continue
            place = self._compute_place(z, self._data_worker, yield_s=0.0005)
            if place is not None and self._wanted_place_z == z:
                with self._lock:
                    self._place_ok, self._place_z = place, z
                    self._masks_dirty = True
                print(f"[Reach] place zone updated for {z * 100:.0f} cm: {place.sum()} cells")

    @property
    def place_zone(self) -> tuple[Optional[float], int]:
        """(release height the place zone was computed for, number of reachable cells in view)."""
        with self._lock:
            return self._place_z, int(self._place_ok.sum())

    def pause(self) -> None:
        self._running.clear()

    def resume(self) -> None:
        self._running.set()

    # ---- drawing ----

    def _rebuild_layers(self) -> None:
        """Rasterize the zone cells (smoothed) into a color layer + mask; runs only when zones change."""
        for mask, ok in ((self._pick_mask, self._pick_ok), (self._place_mask, self._place_ok)):
            mask[:] = 0
            for poly, inside in zip(self._polys, ok):
                if inside:
                    cv2.fillConvexPoly(mask, poly, 255)
            blurred = cv2.GaussianBlur(mask, (31, 31), 0)       # round off the grid staircase
            mask[:] = np.where(blurred >= 128, 255, 0).astype(np.uint8)
        pick, place = self._pick_mask > 0, self._place_mask > 0
        hatch = place & self._stripes
        self._color_layer[:] = 0
        self._color_layer[pick] = _PICK_COLOR               # solid tint: unchanged by the place zone
        self._color_layer[hatch] = _PLACE_COLOR
        self._region[:] = np.where(pick | hatch, 255, 0).astype(np.uint8)
        self._contours = [
            (cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], color)
            for mask, color in ((self._pick_mask, _PICK_COLOR), (self._place_mask, _PLACE_COLOR))
        ]
        self._masks_dirty = False

    def draw(self, image: np.ndarray, alpha: float = 0.28, legend: bool = True) -> None:
        with self._lock:
            if self._masks_dirty:
                self._rebuild_layers()
            color_layer, region, contours, place_z = self._color_layer, self._region, self._contours, self._place_z
        if image.shape[:2] != region.shape:
            return
        blended = cv2.addWeighted(image, 1 - alpha, color_layer, alpha, 0)
        cv2.copyTo(blended, region, image)
        for outline, color in contours:
            cv2.drawContours(image, outline, -1, color, 2)
        if not legend:
            return
        text = "green = can pick here" + (f"   blue stripes = can place here ({place_z * 100:.0f} cm)" if place_z else "")
        cv2.putText(image, text, (10, image.shape[0] - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
