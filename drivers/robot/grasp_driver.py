"""Small grasp-side helper for reBotArm visual grasping.

The SDK owns arm connection, mode switching, Cartesian planning, gravity
compensation, and the control loop. This module provides only the extra
gripper and pose helpers used by the vision workflows.

selected_arm_config(): read the SDK hardware YAML and choose controller mode.

GraspDriver:
  start(): start SDK control and attach gripper tick handling.
  open_gripper(): open to a requested jaw distance.
  open_for_object(): open to the measured object width plus a margin.
  fits(): whether an object of a measured width fits between the jaws.
  grasp(): close with force control and report object contact. Given the
           measured object width, never close past width - squeeze.
  release_gripper(): open and return the gripper to closed rest.
  get_gripper_state(): return cached position, velocity, and torque.
  get_tcp_pose(): return the current TCP pose as a 4x4 matrix.
"""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml


_CAMERAWS_ROOT = Path(__file__).resolve().parents[2]
_REBOT_REPO_NAME = "reBotArm_control_py"
_DEFAULT_REBOT_REPO = _CAMERAWS_ROOT / "sdk" / _REBOT_REPO_NAME

GRIPPER_MAX_DISTANCE_M = 0.09


@dataclass(frozen=True)
class SelectedArmConfig:
    arm_type: str
    controller_mode: str


def _is_rebot_repo_root(path: Path) -> bool:
    pkg = path / _REBOT_REPO_NAME
    return (
        path.is_dir()
        and (pkg / "actuator" / "rebotarm.py").is_file()
        and (path / "config" / "rebotarm.yaml").is_file()
    )


def find_rebot_repo_root(hint: Optional[str] = None) -> Path:
    repo = Path(hint).expanduser() if hint else _DEFAULT_REBOT_REPO
    if not repo.is_absolute():
        repo = (_CAMERAWS_ROOT / repo).resolve()
    else:
        repo = repo.resolve()
    if _is_rebot_repo_root(repo):
        return repo
    raise FileNotFoundError(f"reBotArm_control_py repo not found: {repo}")


def ensure_rebot_sdk_in_syspath(hint: Optional[str] = None) -> Path:
    repo = find_rebot_repo_root(hint)
    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a YAML mapping")
    return data


def selected_hardware_yaml(repo_root: Optional[str] = None) -> Path:
    repo = find_rebot_repo_root(repo_root)
    config_dir = repo / "config"
    global_cfg = _read_yaml(config_dir / "rebotarm.yaml")
    hw_yaml = global_cfg.get("hardware_yaml")
    if not hw_yaml:
        raise ValueError(f"{config_dir / 'rebotarm.yaml'} missing hardware_yaml")

    hw_path = Path(str(hw_yaml))
    if not hw_path.is_absolute():
        hw_path = config_dir / hw_path
    hw_path = hw_path.resolve()
    if not hw_path.is_file():
        raise FileNotFoundError(f"Hardware config not found: {hw_path}")
    return hw_path


def selected_arm_config(repo_root: Optional[str] = None) -> SelectedArmConfig:
    """Return the selected arm type and matching SDK controller mode."""
    hw_path = selected_hardware_yaml(repo_root)
    stem = hw_path.stem.lower()
    if stem.endswith("_dm") or stem == "dm":
        return SelectedArmConfig(arm_type="dm", controller_mode="posvel")
    if stem.endswith("_rs") or stem == "rs":
        return SelectedArmConfig(arm_type="rs", controller_mode="mit")
    raise ValueError(f"Cannot infer arm type from hardware config: {hw_path}")


def check_motor_zeros(repo_root: Optional[str] = None, tol_rad: float = 0.15) -> tuple[bool, list[str]]:
    """Read every motor's position (read-only, nothing enabled) and check it is near zero.

    The scripts expect to start with the arm folded at its rest pose and the gripper closed,
    which is where the motors were zeroed. A reading far from zero means a zero was lost
    (e.g. the arm was moved by hand after a crash) and commanded targets would be wrong.
    Returns (all_ok, report_lines). Only RobStride motors can be read this way.
    """
    from motorbridge import Controller

    cfg = _read_yaml(selected_hardware_yaml(repo_root))
    lines: list[str] = []
    all_ok = True
    ctrl = Controller(str(cfg["channel"]))
    try:
        for jc in cfg["joints"]:
            name = jc["name"]
            if jc.get("vendor") != "robstride":
                lines.append(f"  {name:8s} not checked (vendor {jc.get('vendor')})")
                continue
            motor = ctrl.add_robstride_motor(int(jc["motor_id"]), int(jc["feedback_id"]), str(jc["model"]))
            try:
                pos = float(motor.robstride_get_param_f32(0x7019))   # mechPos, rad
            except Exception as exc:
                all_ok = False
                lines.append(f"  {name:8s} NO RESPONSE ({exc}) - is the arm powered and CAN up?")
                continue
            ok = abs(pos) <= tol_rad
            note = ""
            if not ok:
                turns = round(pos / (2 * np.pi))
                if turns and abs(pos - turns * 2 * np.pi) <= tol_rad:
                    note = f"  <- off by {turns:+d} full turn(s): zero lost"
                else:
                    note = "  <- not at zero"
            all_ok &= ok
            lines.append(f"  {name:8s} {pos:+7.3f} rad ({np.degrees(pos):+7.1f} deg){note}")
    finally:
        try:
            ctrl.close_bus()
        except Exception:
            pass
    return all_ok, lines


class GraspDriver:
    MAX_DISTANCE_M = GRIPPER_MAX_DISTANCE_M
    _STATE_IDLE = "idle"
    _STATE_POSITION = "position"
    _STATE_CLOSING = "closing"
    _STATE_HOLDING = "holding"

    def __init__(
        self,
        arm: Any,
        controller: Any,
        gripper_config: Optional[dict] = None,
        repo_root: Optional[str] = None,
    ) -> None:
        self._arm = arm
        self._controller = controller
        self._arm_group = arm.groups.get("arm")
        self._gripper_group = arm.groups.get("gripper")
        if self._arm_group is None:
            raise ValueError("Hardware config missing groups.arm")
        if self._gripper_group is None or not arm.has_gripper:
            raise ValueError("Hardware config missing groups.gripper")
        gripper_jcfgs = getattr(self._gripper_group, "_jcfgs", [])
        if not gripper_jcfgs:
            raise ValueError("groups.gripper has no joints")
        self._gripper_name = gripper_jcfgs[0].name
        self._gripper_motor: Any = None

        from reBotArm_control_py.kinematics import compute_fk, load_robot_model, pad_q_for_model

        self._compute_fk = compute_fk
        self._pad_q_for_model = pad_q_for_model
        self._model = load_robot_model()
        self._n = self._arm_group.num_joints

        selected = selected_arm_config(repo_root)
        defaults = {
            "dm": {"angle_open": 5.0, "counterclockwise": True, "tau_max": 1.5, "close_torque": 1.0, "default_force": 0.30},
            "rs": {"angle_open": 5.0, "counterclockwise": False, "tau_max": 1.5, "close_torque": 1.0, "default_force": 0.30},
        }[selected.arm_type]
        gcfg = {**defaults, **((gripper_config or {}).get(selected.arm_type) or {})}
        motion_sign = 1.0 if bool(gcfg.get("counterclockwise")) else -1.0
        self._angle_open = -motion_sign * abs(float(gcfg["angle_open"]))
        self._tau_max = abs(float(gcfg["tau_max"]))
        self._open_sign = 1.0 if self._angle_open >= 0.0 else -1.0
        self._close_sign = motion_sign
        self._close_torque = self._close_sign * abs(float(gcfg["close_torque"]))
        self._default_force = self._close_sign * abs(float(gcfg["default_force"]))
        self._open_soft_limit = 0.98 * self._angle_open
        self._open_lo = min(self._open_soft_limit, 0.0)
        self._open_hi = max(self._open_soft_limit, 0.0)
        self._hard_stop_angle = self._open_sign * 0.05
        self._arrive_tol = 0.12
        self._kp_move = 5.0
        self._kd_move = 1.0
        self._kd_close = 0.5
        self._stall_vel = 0.05
        self._startup_dist = 0.30

        # Size-aware grasping: use the object width measured by vision to bound the jaws.
        size_cfg = (gripper_config or {}).get("size_aware") or {}
        self._size_aware = bool(size_cfg.get("enabled", False))
        self._open_margin_m = abs(float(size_cfg.get("open_margin_m", 0.03)))
        self._squeeze_m = abs(float(size_cfg.get("squeeze_m", 0.01)))
        self._fit_clearance_m = abs(float(size_cfg.get("fit_clearance_m", 0.01)))
        self._contact_speed_ratio = float(size_cfg.get("contact_speed_ratio", 0.8))
        self._close_limit: Optional[float] = None   # motor angle the jaws must not close past
        self._close_vel_ema = 0.0                   # smoothed closing speed, rad/s
        self._peak_close_vel = 0.0
        self._grasp_reason = ""

        self._state_lock = threading.Lock()
        self._state = self._STATE_IDLE
        self._target_pos = 0.0
        self._start_pos = 0.0
        self._contact_pos = 0.0
        self._hold_torque = self._default_force
        self._position_reached = True
        self._grasp_result: Optional[bool] = None
        self._last_gripper_state: Optional[tuple[float, float, float]] = None

    def start(self) -> None:
        """Start the SDK arm controller and let this driver own the gripper."""
        if getattr(self._controller, "_running", False):
            return

        self._controller._has_gripper = False
        self._arm.connect()
        self._gripper_motor = self._gripper_group._mm[self._gripper_name]
        if self._arm_group:
            if self._controller._arm_control_mode == "mit":
                self._arm_group.mode_mit(
                    kp=self._arm_group._mit_kp,
                    kd=self._arm_group._mit_kd,
                )
            else:
                self._arm_group.mode_pos_vel()
            self._arm_group.enable()

        self._gripper_group.mode_mit()
        self._gripper_group.enable()
        self._prime_arm_target()
        self._prime_gripper_state()
        self._arm.start_control_loop(self._loop_cb)
        self._controller._running = True

    def _loop_cb(self, r: Any, dt: float) -> None:
        self._controller._loop_cb(r, dt)
        self.gripper_tick(dt)

    def _ensure_running(self) -> None:
        if not getattr(self._controller, "_running", False):
            raise RuntimeError("GraspDriver is not started; call grasp_driver.start() first")

    def _send_gripper_mit(
        self,
        pos: float,
        vel: float = 0.0,
        kp: float = 0.0,
        kd: float = 0.0,
        tau: float = 0.0,
    ) -> None:
        pos_cmd = float(np.clip(pos, self._open_lo, self._open_hi))
        tau_cmd = float(np.clip(tau, -self._tau_max, self._tau_max))
        self._gripper_group.send_mit(
            np.array([pos_cmd], dtype=np.float64),
            vel=np.array([vel], dtype=np.float64),
            kp=np.array([kp], dtype=np.float64),
            kd=np.array([kd], dtype=np.float64),
            tau=np.array([tau_cmd], dtype=np.float64),
        )

    def _prime_arm_target(self) -> None:
        q_now = self._arm.get_state()[0][: self._n]
        self._controller._q_target[:] = q_now
        self._controller._qd_target[:] = 0.0

    def _prime_gripper_state(self, timeout: float = 1.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self._gripper_group._request_feedback()
            state = self._read_gripper_state_cached()
            if state is not None:
                with self._state_lock:
                    self._target_pos = state[0]
                    self._contact_pos = state[0]
                    self._state = self._STATE_IDLE
                    self._position_reached = True
                    self._grasp_result = None
                return
            time.sleep(0.02)

    def _read_gripper_state_cached(self) -> Optional[tuple[float, float, float]]:
        if self._gripper_motor is None:
            return self._last_gripper_state
        st = self._gripper_motor.get_state()
        if st is None:
            return self._last_gripper_state
        self._last_gripper_state = (float(st.pos), float(st.vel), float(st.torq))
        return self._last_gripper_state

    def _wait_gripper_state(self, timeout: float = 1.0) -> tuple[float, float, float]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            state = self._read_gripper_state_cached()
            if state is not None:
                return state
            time.sleep(0.02)
        raise RuntimeError("Gripper feedback is not ready")

    def get_gripper_state(self) -> tuple[float, float, float]:
        state = self._read_gripper_state_cached()
        if state is None:
            raise RuntimeError("Gripper feedback is not ready")
        return state

    def _set_position_target(self, target: float) -> None:
        with self._state_lock:
            self._target_pos = float(target)
            self._state = self._STATE_POSITION
            self._position_reached = False
            self._grasp_result = None

    def _wait_until(self, predicate, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return predicate()

    def _position_done(self) -> bool:
        with self._state_lock:
            return self._position_reached

    def _grasp_done(self) -> bool:
        with self._state_lock:
            return self._grasp_result is not None

    def gripper_tick(self, dt: float = 0.0) -> None:
        del dt
        pos_vel_torq = self._read_gripper_state_cached()
        with self._state_lock:
            state = self._state
            target = self._target_pos
            command: Optional[tuple[float, float, float, float, float]] = None

            if state == self._STATE_POSITION:
                command = (target, 0.0, self._kp_move, self._kd_move, 0.0)
                if pos_vel_torq is not None and abs(pos_vel_torq[0] - target) < self._arrive_tol:
                    self._position_reached = True

            elif state == self._STATE_CLOSING:
                command = (0.0, 0.0, 0.0, self._kd_close, self._close_torque)
                if pos_vel_torq is not None:
                    pos, vel, _ = pos_vel_torq
                    self._contact_pos = pos
                    self._close_vel_ema = 0.8 * self._close_vel_ema + 0.2 * abs(vel)
                    self._peak_close_vel = max(self._peak_close_vel, self._close_vel_ema)
                    moved = abs(pos - self._start_pos) >= self._startup_dist
                    at_hard_stop = self._open_sign * pos < self._open_sign * self._hard_stop_angle
                    at_limit = (
                        self._close_limit is not None
                        and self._open_sign * pos <= self._open_sign * self._close_limit
                    )
                    if at_limit:
                        # Reached width - squeeze: stop here instead of crushing on to fully closed.
                        # Jaws still moving near full speed means nothing resisted them.
                        contact = self._close_vel_ema < self._contact_speed_ratio * self._peak_close_vel
                        self._target_pos = self._close_limit
                        self._hold_torque = 0.0
                        self._state = self._STATE_HOLDING
                        self._grasp_result = contact
                        self._grasp_reason = "stopped at width limit" + ("" if contact else ", no object felt")
                        command = (self._close_limit, 0.0, self._kp_move, self._kd_move, 0.0)
                    elif moved and at_hard_stop:
                        self._target_pos = 0.0
                        self._state = self._STATE_POSITION
                        self._position_reached = False
                        self._grasp_result = False
                        self._grasp_reason = "closed fully, no object"
                        command = (0.0, 0.0, self._kp_move, self._kd_move, 0.0)
                    elif moved and abs(vel) < self._stall_vel:
                        self._target_pos = pos
                        self._state = self._STATE_HOLDING
                        self._grasp_result = True
                        self._grasp_reason = "contact"
                        command = (pos, 0.0, self._kp_move, self._kd_move, self._hold_torque)

            elif state == self._STATE_HOLDING:
                command = (self._target_pos, 0.0, self._kp_move, self._kd_move, self._hold_torque)

        if command is not None:
            pos, vel, kp, kd, tau = command
            self._send_gripper_mit(pos, vel=vel, kp=kp, kd=kd, tau=tau)

    def _distance_to_angle(self, distance_m: float) -> float:
        d = float(np.clip(distance_m, 0.0, self.MAX_DISTANCE_M))
        raw_target = (d / self.MAX_DISTANCE_M) * self._angle_open
        return float(np.clip(raw_target, self._open_lo, self._open_hi))

    def open_gripper(self, distance_m: float = GRIPPER_MAX_DISTANCE_M, timeout: float = 3.0) -> None:
        self._ensure_running()
        self._set_position_target(self._distance_to_angle(distance_m))
        self._wait_until(self._position_done, timeout)

    def fits(self, object_width_m: Optional[float]) -> bool:
        """Whether an object of this measured width fits between the open jaws."""
        if object_width_m is None or not self._size_aware:
            return True
        return object_width_m + self._fit_clearance_m <= self.MAX_DISTANCE_M

    def size_plan(self, object_width_m: Optional[float]) -> str:
        """One-line description of how the jaws will be bounded for this object."""
        if object_width_m is None or not self._size_aware:
            return "size-aware gripping off: full open, close until contact"
        open_m = min(object_width_m + self._open_margin_m, self.MAX_DISTANCE_M)
        limit_m = max(object_width_m - self._squeeze_m, 0.0)
        return (f"object {object_width_m * 100:.1f} cm: open to {open_m * 100:.1f} cm, "
                f"never close below {limit_m * 100:.1f} cm")

    def open_for_object(self, object_width_m: Optional[float], timeout: float = 3.0) -> None:
        """Open just wider than the object (full open when size-aware gripping is off)."""
        if object_width_m is None or not self._size_aware:
            self.open_gripper(timeout=timeout)
            return
        self.open_gripper(object_width_m + self._open_margin_m, timeout=timeout)

    def jaw_opening_m(self) -> Optional[float]:
        """Current jaw opening, using the same linear angle <-> distance mapping as open_gripper()."""
        state = self._read_gripper_state_cached()
        if state is None:
            return None
        return float(np.clip(state[0] / self._angle_open, 0.0, 1.0)) * self.MAX_DISTANCE_M

    @property
    def last_grasp_reason(self) -> str:
        with self._state_lock:
            return self._grasp_reason

    def grasp(
        self,
        force: Optional[float] = None,
        timeout: float = 5.0,
        object_width_m: Optional[float] = None,
    ) -> bool:
        """Close until contact. With object_width_m (and size_aware enabled), never close
        past object_width_m - squeeze_m, so a soft or mis-detected object is not crushed."""
        self._ensure_running()
        start_pos, _, _ = self._wait_gripper_state()
        hold_torque = self._close_sign * float(
            np.clip(abs(force if force is not None else self._default_force), 0.05, self._tau_max)
        )
        close_limit = None
        if object_width_m is not None and self._size_aware:
            close_limit = self._distance_to_angle(max(object_width_m - self._squeeze_m, 0.0))
        with self._state_lock:
            self._start_pos = start_pos
            self._contact_pos = start_pos
            self._target_pos = 0.0
            self._hold_torque = hold_torque
            self._close_limit = close_limit
            self._close_vel_ema = 0.0
            self._peak_close_vel = 0.0
            self._grasp_reason = ""
            self._state = self._STATE_CLOSING
            self._position_reached = False
            self._grasp_result = None

        if not self._wait_until(self._grasp_done, timeout):
            with self._state_lock:
                if self._grasp_result is None:
                    self._target_pos = self._contact_pos
                    self._state = self._STATE_POSITION
                    self._position_reached = False
                    self._grasp_result = False
                    self._grasp_reason = "timed out"

        with self._state_lock:
            return bool(self._grasp_result)

    def release_gripper(self, timeout: float = 4.0) -> None:
        self._ensure_running()
        self.open_gripper(timeout=min(2.0, timeout))
        self._set_position_target(0.0)
        self._wait_until(self._position_done, timeout)

    def get_tcp_pose(self) -> np.ndarray:
        q_arm = self._arm.get_state(request_feedback=False)[0][: self._n]
        q = self._pad_q_for_model(self._model, q_arm, self._n)
        pos, rot, _ = self._compute_fk(self._model, q)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = rot
        T[:3, 3] = pos
        return T
