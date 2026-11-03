"""Pick-and-place pipeline shared by scripts/set.py (OpenCV window) and the web UI (ui/).

PickPlaceSession: camera, detection model, reach zones and arm, set up step by step, plus
  detect(): run the detector on a frame.
  plan(): detect the pick object and the destination on a frame and work out every arm
          pose for the task, checked with IK before anything moves (raises PlanError).
  execute(): run a plan on the arm.
  check_held(): look between the jaws with the wrist camera.
  close(): release the gripper, home, disconnect and close the camera.
TopTracker: steady height of the destination's top from noisy per-frame measurements.

Lower-level steps:
  move_ready(): move to the configured ready pose and wait for it.
  execute_place_sequence(): grasp -> lift -> check held -> place -> return to ready.
  best_match(): highest-confidence valid detection for a text label.
  top_height(): measured height of a detection's top (cup rim; ~0 for a flat spot).
  render_display(): draw detections and the pick/place positions on a frame.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
from drivers.camera import make_camera
from drivers.robot.grasp_driver import GraspDriver, check_motor_zeros, selected_arm_config
from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.controllers import RebotArmEndPose
from utils.camera_utils import compose_cam_to_base_transform, hand_eye_compensation_matrix, load_hand_eye
from utils.grasp_check import HeldCheck, check_object_in_gripper
from utils.ordinary_grasp import GraspPose, estimate_grasps
from utils.reach_map import PLACE_RPY, ReachMap
from utils.transforms import mask_top_height_in_base, transform_grasp_pose_to_base
from utils.yolo_utils import detection_mask, load_yolo

DEFAULT_READY = {"x": 0.25, "y": 0.0, "z": 0.35, "roll": 0.0, "pitch": 1.2, "yaw": 0.0, "duration": 3.0}


class PlanError(Exception):
    """A task that cannot be run as seen. reason: no_pick | no_place | pick_unreachable | place_unreachable."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


class ZeroCheckError(RuntimeError):
    """The startup zero check failed; report holds one line per motor."""

    def __init__(self, report: list[str]):
        super().__init__("Motor zero check failed: fold the arm to its rest pose and close the gripper, "
                         "or re-zero the motors that lost their zero (--skip-zero-check to override).")
        self.report = report


@dataclass
class PickPlacePlan:
    pick: GraspPose
    place: GraspPose
    grasp6d: tuple[float, ...]
    pre6d: tuple[float, ...]
    place_center: np.ndarray        # release point in the base frame
    place_top_m: Optional[float]    # measured top of the destination (None: not measurable)
    results: list[Any]              # detector output the plan was made from, for display
    grasps: list[GraspPose]


class TopTracker:
    """Median of recent destination-top measurements: steady through depth noise, and kept while
    the destination is briefly out of view, so a missed detection does not drop it to table level."""

    def __init__(self, size: int = 15):
        self._history: deque[float] = deque(maxlen=size)

    def add(self, top_m: Optional[float]) -> Optional[float]:
        if top_m is not None:
            self._history.append(top_m)
        return self.value

    @property
    def value(self) -> Optional[float]:
        return float(np.median(self._history)) if self._history else None

    def clear(self) -> None:
        self._history.clear()


class PickPlaceSession:
    """Everything the pick-and-place demo runs on. Set up in order: open_camera(), load_model(),
    build_reach() (before connect_robot(), so its IK work does not compete with the arm control loop),
    connect_robot(). Each step can be skipped by a caller that does not need it."""

    def __init__(self, cfg: dict[str, Any], project_root: Path):
        self.cfg = cfg
        self.project_root = Path(project_root)
        task_cfg = cfg.get("place_task", {})
        self.pick_name = str(task_cfg.get("pick", "yellow banana"))
        self.place_name = str(task_cfg.get("place", "box"))
        self.clearance_m = float(task_cfg.get("release_clearance_m", 0.05))
        self.conf = float(cfg.get("detection", {}).get("conf_threshold", 0.25))
        self.presets = [str(p) for p in cfg.get("yolo", {}).get("custom_classes") or []]

        self.robot_cfg = cfg.get("robot", {})
        self.ready_cfg = self.robot_cfg.get("ready_pose", DEFAULT_READY)
        gp_cfg = cfg.get("grasp_pipeline", {})
        grasp_cfg = gp_cfg.get("grasp", {})
        self.pregrasp_offset_m = float(grasp_cfg.get("pregrasp_offset_m", 0.08))
        self.depth_quantile = float(grasp_cfg.get("depth_quantile", 0.75))
        self.infer_every = max(1, int(gp_cfg.get("infer_every_live", 2)))
        self.held_cfg = gp_cfg.get("held_check", {})

        self.cam = None
        self.K: Optional[np.ndarray] = None
        self.T_hand_eye: Optional[np.ndarray] = None
        self.model = None
        self.yolo_opts: dict[str, Any] = {}
        self.open_vocab = False
        self.reach: Optional[ReachMap] = None
        self.rebotarm: Optional[RebotArm] = None
        self.controller: Optional[RebotArmEndPose] = None
        self.grasp_driver: Optional[GraspDriver] = None

    # ---- setup ----

    def open_camera(self) -> None:
        cam_cfg = self.cfg.get("camera", {})
        print(f"=== Camera: {cam_cfg.get('type')} ===")
        cam = make_camera(self.cfg)
        cam.open()
        self.cam = cam
        cam.warm_up(15)
        self.K = cam.K.astype(np.float32)
        T_hand_eye, mode = load_hand_eye(self.project_root, str(cam_cfg.get("type", "")).lower())
        if T_hand_eye is None or mode != "eye_in_hand":
            print("[WARN] Hand-eye calibration unavailable; grasp execution disabled")
            T_hand_eye = None
        self.T_hand_eye = T_hand_eye

    def load_model(self) -> None:
        yolo_cfg = self.cfg.setdefault("yolo", {})
        model_name = yolo_cfg.get("model_name", "yoloe-26s-seg.pt")
        print(f"=== Load YOLO: {model_name} ===")
        # Detect only the two task labels, so other classes cannot compete for them.
        yolo_cfg["custom_classes"] = [self.pick_name, self.place_name]
        self.model, self.yolo_opts = load_yolo(self.cfg, project_root=self.project_root)
        # Text labels can only be changed live on open-vocabulary models.
        self.open_vocab = bool(yolo_cfg.get("use_world", True)) and any(
            k in str(model_name).lower() for k in ("yoloe", "world"))

    def build_reach(self) -> Optional[ReachMap]:
        """Compute the reach zones (a few seconds). Needs the camera and hand-eye calibration."""
        if self.T_hand_eye is None or self.cam is None:
            return None
        probe = None
        for _ in range(30):
            probe, _ = self.cam.get_frame()
            if probe is not None:
                break
        if probe is None:
            return None
        print("=== Computing reach zones (a few seconds) ===")
        try:
            reach = ReachMap(self.K, (probe.shape[1], probe.shape[0]), self.T_hand_eye,
                             hand_eye_compensation_matrix(self.cfg), self.ready_cfg,
                             pregrasp_offset_m=self.pregrasp_offset_m)
            reach.compute_all(self.clearance_m)          # until a destination is seen: table level
        except Exception as exc:
            print(f"[Reach] Zones disabled: {exc}")
            return None
        self.reach = reach
        return reach

    def check_zeros(self) -> list[str]:
        """Startup zero check; returns the report, raises ZeroCheckError if a motor is off zero."""
        ok, report = check_motor_zeros(self.robot_cfg.get("repo_root"),
                                       float(self.robot_cfg.get("zero_check_tol_rad", 0.15)))
        print("[Zero check] Motor positions (expected: arm folded at rest, gripper closed):")
        print("\n".join(report))
        if not ok:
            raise ZeroCheckError(report)
        return report

    def connect_robot(self) -> None:
        """Enable the arm and the gripper (does not move them). Run check_zeros() first."""
        print("=== Init robot ===")
        selected = selected_arm_config(self.robot_cfg.get("repo_root"))
        self.rebotarm = RebotArm()
        self.controller = RebotArmEndPose(self.rebotarm, arm_control_mode=selected.controller_mode)
        self.grasp_driver = GraspDriver(self.rebotarm, self.controller,
                                        gripper_config=self.robot_cfg.get("gripper"),
                                        repo_root=self.robot_cfg.get("repo_root"))
        self.grasp_driver.start()
        print(f"[Robot] mode: {selected.controller_mode}")

    @property
    def robot_running(self) -> bool:
        return self.controller is not None and bool(getattr(self.controller, "_running", False))

    @property
    def can_grasp(self) -> bool:
        return self.T_hand_eye is not None and self.grasp_driver is not None

    # ---- perception ----

    def set_task(self, pick: str, place: str) -> None:
        if self.open_vocab and self.model is not None:
            self.model.set_classes([pick, place])
        self.pick_name, self.place_name = pick, place

    def detect(self, color: np.ndarray, depth: np.ndarray, conf: Optional[float] = None,
               ) -> tuple[list[Any], list[GraspPose]]:
        results = self.model.predict(
            color,
            verbose=False,
            device=self.yolo_opts.get("device", "cpu"),
            conf=float(self.conf if conf is None else conf),
            iou=float(self.yolo_opts.get("iou", 0.45)),
        )
        return results, estimate_grasps(results, depth, self.K, depth_quantile=self.depth_quantile)

    def cam_to_base(self) -> np.ndarray:
        """Camera -> base transform at the arm's current pose."""
        return cam_to_base(self.T_hand_eye, self.grasp_driver, self.cfg)

    def measure_top(self, results: list[Any], grasps: list[GraspPose], place: Optional[GraspPose],
                    depth: np.ndarray) -> Optional[float]:
        """Height of the destination's top on this frame (needs the arm, for the camera pose)."""
        if not self.can_grasp:
            return None
        return top_height(results, grasps, place, depth, self.K, self.cam_to_base())

    # ---- task ----

    def plan(self, color: np.ndarray, depth: np.ndarray, conf: Optional[float] = None,
             clearance_m: Optional[float] = None) -> PickPlacePlan:
        """Detect both objects on this frame and plan the arm poses. Raises PlanError."""
        pick_name, place_name = self.pick_name, self.place_name
        clearance_m = self.clearance_m if clearance_m is None else clearance_m
        results, grasps = self.detect(color, depth, conf)
        pick = best_match(grasps, pick_name)
        place = best_match(grasps, place_name)
        if pick is None:
            raise PlanError("no_pick", f"No valid '{pick_name}' detected")
        if place is None:
            raise PlanError("no_place", f"No valid '{place_name}' detected")
        print(f"[G] Pick: class={pick.class_name} conf={pick.conf:.3f} xyz={pick.position.tolist()}")
        print(f"[G] Place: class={place.class_name} conf={place.conf:.3f} xyz={place.position.tolist()}")

        T_cam2base = self.cam_to_base()
        grasp6d, pre6d = transform_grasp_pose_to_base(pick.position, pick.tcp_rotation,
                                                      T_cam2base, self.pregrasp_offset_m)
        pos = place.position
        place_base = T_cam2base @ np.array([float(pos[0]), float(pos[1]), float(pos[2]), 1.0])
        # Release at the destination's measured top + clearance (table level if not measurable).
        top = top_height(results, grasps, place, depth, self.K, T_cam2base)
        if top is None:
            print(f"[G] Could not measure the height of '{place_name}'; assuming table level")
        release_z = (top or 0.0) + clearance_m
        print(f"[G] '{place_name}' top at {(top or 0.0) * 100:.1f} cm -> release at {release_z * 100:.1f} cm")
        place_center = np.array([float(place_base[0]), float(place_base[1]), release_z])

        # Check every pose with IK before moving, so nothing is picked up that cannot be placed.
        if self.reach is not None:
            q_pre = self.reach.pose_reachable(pre6d)
            if q_pre is None or self.reach.pose_reachable(grasp6d, seed=q_pre) is None:
                raise PlanError("pick_unreachable", f"'{pick_name}' is out of reach: move it into the green zone")
            if self.reach.pose_reachable((*place_center, *PLACE_RPY)) is None:
                raise PlanError("place_unreachable",
                                f"Release point above '{place_name}' ({release_z * 100:.0f} cm high) is out of "
                                f"reach: move it into the blue zone, or lower the clearance")
        return PickPlacePlan(pick, place, grasp6d, pre6d, place_center, top, results, grasps)

    def check_held(self, color: Optional[np.ndarray], depth: Optional[np.ndarray],
                   save_dir: Optional[Path] = None) -> HeldCheck:
        """Wrist-camera check for something between the jaws (call at the ready pose, after lifting)."""
        result = check_object_in_gripper(depth, self.K, self.T_hand_eye, self.grasp_driver.jaw_opening_m(),
                                         near_margin_m=float(self.held_cfg.get("near_margin_m", 0.03)))
        if save_dir is not None:
            save_held_check_images(save_dir, color, depth, result)
        return result

    @property
    def held_check_enabled(self) -> bool:
        return self.T_hand_eye is not None and bool(self.held_cfg.get("enabled", True))

    def execute(
        self,
        plan: PickPlacePlan,
        dry_run: bool = False,
        check_held: Optional[Callable[[], HeldCheck]] = None,
        controller: Any = None,
        grasp_driver: Any = None,
    ) -> bool:
        """Run a plan on the arm. controller / grasp_driver override the session's own (e.g. wrapped
        with a stop check)."""
        if self.reach is not None:
            self.reach.pause()                    # keep IK work off the CPU while the arm moves
        try:
            return execute_place_sequence(
                controller or self.controller, grasp_driver or self.grasp_driver,
                plan.grasp6d, plan.pre6d, plan.place_center, self.ready_cfg, dry_run=dry_run,
                jaw_width_m=plan.pick.jaw_width_m, pick_name=self.pick_name, place_name=self.place_name,
                check_held=check_held,
            )
        finally:
            if self.reach is not None:
                self.reach.resume()

    def move_ready(self, controller: Any = None) -> None:
        move_ready(controller or self.controller, self.ready_cfg)

    def close(self, release_gripper: bool = True) -> None:
        """Release the gripper (unless release_gripper is False), home (SDK safe_home) and disconnect
        the arm; close the camera."""
        try:
            if release_gripper and self.robot_running and self.grasp_driver is not None:
                self.grasp_driver.release_gripper()
        except Exception as exc:
            print(f"[Exit] {exc}")
        try:
            if self.robot_running:
                self.controller.end()
            elif self.rebotarm is not None:
                self.rebotarm.disconnect()
        except Exception as exc:
            print(f"[Exit] {exc}")
        try:
            if self.cam is not None:
                self.cam.close()
        except Exception:
            pass


def save_held_check_images(save_dir: Path, color: Optional[np.ndarray], depth: Optional[np.ndarray],
                           result: HeldCheck) -> None:
    """Keep what the held check looked at (color + depth, with its region), for tuning."""
    if color is None:
        return
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    x1, y1, x2, y2 = result.roi_xyxy
    snap = color.copy()
    cv2.rectangle(snap, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(snap, result.detail, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.imwrite(str(save_dir / "held_check_latest.jpg"), snap)
    if depth is not None:                   # depth 0-50 cm as a color map (black = no depth)
        depth_vis = cv2.applyColorMap(
            cv2.convertScaleAbs(np.clip(depth, 0, 500), alpha=255.0 / 500.0), cv2.COLORMAP_JET)
        depth_vis[depth == 0] = 0
        cv2.rectangle(depth_vis, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.imwrite(str(save_dir / "held_check_depth.jpg"), depth_vis)


def wait_motion(controller: RebotArmEndPose, duration: float, extra: float = 0.6) -> None:
    thread = getattr(controller, "_send_thread", None)
    if thread is not None and thread.is_alive():
        thread.join(timeout=duration + extra + 2.0)
    else:
        time.sleep(duration + extra)


def move_ready(controller: RebotArmEndPose, ready_cfg: dict[str, Any]) -> None:
    duration = float(ready_cfg.get("duration", 3.0))
    controller.move_to_traj(
        x=float(ready_cfg.get("x", 0.25)),
        y=float(ready_cfg.get("y", 0.0)),
        z=float(ready_cfg.get("z", 0.35)),
        roll=float(ready_cfg.get("roll", 0.0)),
        pitch=float(ready_cfg.get("pitch", 1.2)),
        yaw=float(ready_cfg.get("yaw", 0.0)),
        duration=duration,
    )
    wait_motion(controller, duration)


def cam_to_base(T_hand_eye: np.ndarray, grasp_driver: GraspDriver, cfg: dict[str, Any]) -> np.ndarray:
    return compose_cam_to_base_transform(grasp_driver.get_tcp_pose(), T_hand_eye, cfg)


def execute_place_sequence(
    controller: RebotArmEndPose,
    grasp_driver: GraspDriver,
    grasp6d: tuple[float, ...],
    pre6d: tuple[float, ...],
    place_center: np.ndarray,
    ready_cfg: dict[str, Any],
    dry_run: bool,
    jaw_width_m: Optional[float] = None,
    pick_name: str = "object",
    place_name: str = "destination",
    check_held: Optional[Callable[[], HeldCheck]] = None,
) -> bool:
    """Execute the complete place sequence: grasp -> lift -> place -> return."""
    xg, yg, zg, rxg, ryg, rzg = grasp6d
    xp, yp, zp, rxp, ryp, rzp = pre6d

    print(f"[Step2] pregrasp  xyz=({xp:+.3f},{yp:+.3f},{zp:+.3f})  rpy=({rxp:+.3f},{ryp:+.3f},{rzp:+.3f})")
    print(f"[Step2] grasp     xyz=({xg:+.3f},{yg:+.3f},{zg:+.3f})  rpy=({rxg:+.3f},{ryg:+.3f},{rzg:+.3f})")
    print(f"[Step2] gripper   {grasp_driver.size_plan(jaw_width_m)}")

    if not grasp_driver.fits(jaw_width_m):
        print(f"[Step2] Object too wide for the gripper "
              f"({jaw_width_m * 100:.1f} cm, max {grasp_driver.MAX_DISTANCE_M * 100:.0f} cm); skipping")
        return False

    if dry_run:
        print("[Step2] dry run; skip motion")
        return True

    print("[Step2] Open gripper")
    grasp_driver.open_for_object(jaw_width_m)

    print("[Step2] Move to pregrasp")
    if not controller.move_to_traj(xp, yp, zp, rxp, ryp, rzp, duration=2.0):
        print("[Step2] Pregrasp IK failed")
        return False
    wait_motion(controller, 2.0)

    print("[Step2] Move to grasp")
    if not controller.move_to_traj(xg, yg, zg, rxg, ryg, rzg, duration=1.5):
        print("[Step2] Grasp IK failed")
        return False
    wait_motion(controller, 1.5)

    print("[Step2] Closing gripper")
    felt = grasp_driver.grasp(object_width_m=jaw_width_m)
    reason = grasp_driver.last_grasp_reason
    print(f"[Step2] Gripper: {'object felt' if felt else 'nothing felt'} ({reason})")
    if reason.startswith("closed fully"):
        print("[Step2] Empty grasp")
        return False

    # A light, size-limited grasp may not be felt by the motor, so lift and let the camera look.
    print(f"[Step3] Lift to ready position with {pick_name}")
    move_ready(controller, ready_cfg)

    held = felt
    if check_held is not None:
        result = check_held()
        print(f"[Step3] Camera: {result.detail}")
        if result.held is not None:
            held = result.held
    if not held:
        print("[Step3] Empty grasp: nothing in the gripper; skipping place")
        return False
    print(f"[Step3] Holding {pick_name}")

    print(f"[Step4] Move above {place_name}: "
          f"xyz=({place_center[0]:+.3f},{place_center[1]:+.3f},{place_center[2]:+.3f})")
    roll, pitch, yaw = PLACE_RPY
    if not controller.move_to_traj(place_center[0], place_center[1], place_center[2], roll, pitch, yaw, duration=2.0):
        print("[Step4] Place position IK failed")
        return False
    wait_motion(controller, 2.0)

    print(f"[Step4] Release gripper - {pick_name} drops onto {place_name}")
    grasp_driver.open_gripper(timeout=0.5)

    print("[Step5] Return to ready position")
    move_ready(controller, ready_cfg)

    print("[Step5] Close gripper")                           # only once the arm is back at ready
    grasp_driver.open_gripper(0.0)

    return True


def render_display(
    image: np.ndarray,
    grasps: list[GraspPose],
    best_pick: Optional[GraspPose],
    best_place: Optional[GraspPose],
    status_text: str,
    pick_name: str = "pick",
    place_name: str = "place",
) -> np.ndarray:
    display = image.copy()
    for grasp in grasps:
        color = (0, 255, 0) if grasp.is_valid else (0, 165, 255)
        cv2.rectangle(display,
                      (int(grasp.bbox_xyxy[0]), int(grasp.bbox_xyxy[1])),
                      (int(grasp.bbox_xyxy[2]), int(grasp.bbox_xyxy[3])),
                      color, 2)
        cv2.putText(display, f"{grasp.class_name} {grasp.conf:.2f}",
                    (int(grasp.bbox_xyxy[0]), int(grasp.bbox_xyxy[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    if status_text:
        cv2.putText(display, status_text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)

    if best_pick is not None:
        x_m, y_m, z_m = best_pick.position.tolist()
        cv2.putText(display, f"{pick_name}: xyz=({x_m:+.3f},{y_m:+.3f},{z_m:+.3f})",
                    (10, display.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 255, 140), 2)

    if best_place is not None:
        x_m, y_m, z_m = best_place.position.tolist()
        cv2.putText(display, f"{place_name}: xyz=({x_m:+.3f},{y_m:+.3f},{z_m:+.3f})",
                    (10, display.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 120), 2)

    return display


def top_height(
    results: list[Any],
    grasps: list[GraspPose],
    target: Optional[GraspPose],
    depth_mm: np.ndarray,
    K: np.ndarray,
    T_cam2base: np.ndarray,
) -> Optional[float]:
    """Measured base-frame height of a detection's top (cup rim; ~0 for a flat spot), or None."""
    if target is None or not results:
        return None
    # estimate_grasps keeps detection order, so the grasp's position in the list is its detection index.
    index = next((i for i, g in enumerate(grasps) if g is target), None)
    if index is None:
        return None
    mask = detection_mask(results[0], index, depth_mm.shape[:2], target.bbox_xyxy)
    top = mask_top_height_in_base(mask, depth_mm, K, T_cam2base)
    return None if top is None else max(top, 0.0)


def best_match(grasps: list[GraspPose], label: str) -> Optional[GraspPose]:
    """Highest-confidence valid detection for a label; exact class matches win over partial ones."""
    label = label.lower().strip()
    valid = [g for g in grasps if g.is_valid]
    exact = [g for g in valid if g.class_name.lower().strip() == label]
    candidates = exact or [g for g in valid if label in g.class_name.lower() or g.class_name.lower() in label]
    return max(candidates, key=lambda g: g.conf) if candidates else None
