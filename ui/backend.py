"""Robot side of the web UI: camera + detection loop, pick-and-place worker, E-stop and Go Home.

Same pipeline as scripts/set.py (utils/pick_place.py PickPlaceSession), driven by HTTP
commands instead of keys:

  camera thread  grabs frames, runs YOLO, measures the destination's height and
                 renders the annotated view (served as an MJPEG stream).
  worker thread  runs one motion command at a time: pick & place, Go Home, gripper.
  E-stop         cancels the running trajectory and holds the current joint angles
                 (motors stay powered, the gripper keeps what it holds). Every arm and
                 gripper call the worker makes goes through a guard that refuses to run
                 once the stop is pressed, so the sequence ends at its next step.
                 Go Home clears the stop and folds the arm to its rest pose.
  Quit / exit    parks the arm (rest pose, gripper closed; a held object is let go
                 first) - from the Quit button, and from shutdown() when the program
                 is stopped with Ctrl+C, kill or a closed terminal (see ui/server.py).
"""

from __future__ import annotations

import re
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drivers.robot.grasp_driver import GRIPPER_MAX_DISTANCE_M
from ui.arm_motion import arm_joint_count, hold_position, move_joints_to_zero
from utils.camera_utils import load_config
from utils.grasp_check import HeldCheck
from utils.ordinary_grasp import GraspPose
from utils.pick_place import (
    PickPlaceSession,
    PlanError,
    TopTracker,
    ZeroCheckError,
    best_match,
    render_display,
)

REST_MAX_VEL = 0.5          # rad/s joint speed limit when folding to rest (same as the SDK's safe_home)
STEP_RE = re.compile(r"\[Step(\d)\]")
DEFAULT_LOGOS = [{"file": "seeed.png", "alt": "Seeed Studio", "height": 30}]


class Aborted(Exception):
    """Raised inside the worker once the emergency stop is pressed."""


class _Guard:
    """Wraps the controller / gripper driver: every public call checks the E-stop before and after."""

    def __init__(self, target: Any, app: "RobotApp", on_call: Optional[Callable[[str, tuple], None]] = None):
        self._target = target
        self._app = app
        self._on_call = on_call

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._target, name)
        if name.startswith("_") or not callable(attr):
            return attr                       # wait_motion() reads controller._send_thread

        def call(*args, **kwargs):
            self._app.check_stop()
            result = attr(*args, **kwargs)
            if self._on_call is not None:
                self._on_call(name, args)
            self._app.check_stop()            # a move planned while the stop was pressed is frozen here
            return result

        return call


class LogTee:
    """Copies stdout into the app's log (shown in the staff panel) and tracks the [StepN] progress."""

    def __init__(self, app: "RobotApp", stream):
        self._app = app
        self._stream = stream
        self._partial = ""

    def write(self, text: str) -> int:
        try:
            self._stream.write(text)
        except (OSError, ValueError):          # terminal gone (closed window / SIGHUP): keep the in-app log
            pass
        self._partial += text
        while "\n" in self._partial:
            line, self._partial = self._partial.split("\n", 1)
            if line.strip():
                self._app.on_log_line(line)
        return len(text)

    def flush(self) -> None:
        try:
            self._stream.flush()
        except (OSError, ValueError):
            pass

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class RobotApp:
    def __init__(self, config_path: Path, dry_run: bool = False, no_robot: bool = False,
                 skip_zero_check: bool = False):
        cfg = load_config(config_path)
        self.session = PickPlaceSession(cfg, PROJECT_ROOT)
        self.dry_run = dry_run
        self.no_robot = no_robot
        self.skip_zero_check = skip_zero_check
        ui_cfg = cfg.get("ui") or {}
        self.branding = {
            "title": str(ui_cfg.get("title", "AI Pick & Place")),
            "logos": ui_cfg.get("logos") or DEFAULT_LOGOS,
        }

        # Status shown in the UI
        self.robot_ready = False
        self.status = "starting"        # starting | idle | busy | stopped | error | quitting
        self.message = "Starting up..."
        self.error: Optional[str] = None
        self.zero_report: list[str] = []
        self.pose = "unknown"           # rest | ready | other | unknown
        self.action: Optional[str] = None
        self.step = 0                   # 1..5 while a pick runs (see [StepN] in utils/pick_place.py)
        self.result: Optional[dict[str, Any]] = None
        self.holding = False
        self.stats = {"attempts": 0, "success": 0}
        self.parked = False             # arm folded at rest with the gripper closed, nothing moved since
        self.quitting = False
        self.fps = 0.0
        self.top = TopTracker()         # measured height of the destination's top
        self.detections: list[dict[str, Any]] = []
        self.log: deque[str] = deque(maxlen=300)
        self._log_mark = 0              # log length when the current pick started
        self._log_count = 0

        # Threading
        self._stop = threading.Event()             # E-stop latch
        self._shutdown = threading.Event()
        self._freeze_lock = threading.Lock()
        self._model_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        self._frame: tuple[Optional[np.ndarray], Optional[np.ndarray]] = (None, None)
        self._frame_seq = 0
        self._last_results: list[Any] = []
        self._last_grasps: list[GraspPose] = []
        self.jpeg: Optional[bytes] = None
        self.jpeg_seq = 0
        self._worker: Optional[threading.Thread] = None
        self._worker_lock = threading.Lock()
        self._camera_thread: Optional[threading.Thread] = None
        self._init_thread: Optional[threading.Thread] = None

    @property
    def presets(self) -> list[str]:
        return self.session.presets

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        self._init_thread = threading.Thread(target=self._init, name="init", daemon=True)
        self._init_thread.start()

    def retry_init(self, skip_zero_check: bool = False) -> bool:
        if self.status != "error":
            return False
        self.skip_zero_check = self.skip_zero_check or skip_zero_check
        self.status, self.error, self.message = "starting", None, "Starting up..."
        self.start()
        return True

    def _init(self) -> None:
        s = self.session
        try:
            if s.cam is None:
                self.message = "Starting the camera..."
                s.open_camera()
            if s.model is None:
                self.message = "Loading the vision model..."
                s.load_model()

            if self._shutdown.is_set():
                return
            # Reach zones before the arm control loop starts, so the IK work does not compete with it.
            if s.reach is None and s.T_hand_eye is not None:
                self.message = "Working out where the arm can reach..."
                s.build_reach()

            if self._shutdown.is_set():                  # exiting: never connect the arm now
                return
            if self._camera_thread is None:
                self._camera_thread = threading.Thread(target=self._camera_loop, name="camera", daemon=True)
                self._camera_thread.start()

            if self.no_robot:
                print("[Robot] --no-robot: camera and detection only")
                self.status, self.message = "idle", "Camera only (no robot connected)"
                return

            if not self.robot_ready:
                self.message = "Connecting to the robot arm..."
                if not self.skip_zero_check:
                    self.zero_report = s.check_zeros()
                if self._shutdown.is_set():
                    return
                s.connect_robot()
                self.robot_ready = True
                self.pose = "rest"

            if self._stop.is_set():                       # E-stop pressed during startup: stay put
                self.status, self.message = "stopped", "Emergency stop - press Go Home"
                return
            self.status = "idle"
            self.message = "Ready"
            self._submit("ready", self._do_ready)
        except Exception as exc:
            if isinstance(exc, ZeroCheckError):
                self.zero_report = exc.report
            print(f"[Init] {exc}")
            self.status, self.error = "error", str(exc)
            self.message = "Startup problem - ask a staff member"

    def shutdown(self) -> None:
        """Program exit (Quit button, Ctrl+C, kill, closed terminal): park the arm, close the gripper,
        disconnect, close the camera. Skips parking when the Quit button already did it."""
        print("\n[Exit] Parking the arm and closing the gripper - please wait")
        self.quitting = True
        self._shutdown.set()                               # stops the camera loop and startup
        if self._init_thread is not None and self._init_thread.is_alive():
            print("[Exit] Waiting for startup to finish its current step")
            self._init_thread.join(timeout=60.0)
        if self._worker_alive() and self._worker is not threading.current_thread():
            self._stop.set()                               # end the running command at its next step
            self._worker.join(timeout=15.0)
        self._stop.clear()
        s = self.session
        arm_on = self.robot_ready and s.robot_running
        release = False
        if arm_on and not self.parked:
            try:
                self._freeze()
                self._park(release_held=True)
            except Exception as exc:
                print(f"[Exit] Parking failed ({exc}); releasing the gripper, SDK safe home")
                release = True
        if self._camera_thread is not None:
            self._camera_thread.join(timeout=3.0)
        s.close(release_gripper=release)                   # SDK safe_home (no motion when at rest) + disconnect
        print("Done.")

    # ------------------------------------------------------------------ log / status

    def on_log_line(self, line: str) -> None:
        self.log.append(line)
        self._log_count += 1
        m = STEP_RE.search(line)
        if m and self.action == "pick":
            self.step = int(m.group(1))

    def _log_since_mark(self) -> list[str]:
        n = self._log_count - self._log_mark
        return list(self.log)[-n:] if 0 < n <= len(self.log) else []

    def state(self, with_log: bool = False) -> dict[str, Any]:
        s = self.session
        jaw = None
        if self.robot_ready and s.grasp_driver is not None:
            try:
                jaw = s.grasp_driver.jaw_opening_m()
            except Exception:
                jaw = None
        top = self.top.value
        out = {
            "status": self.status,
            "message": self.message,
            "error": self.error,
            "zero_report": self.zero_report if self.status == "error" else [],
            "pose": self.pose,
            "action": self.action,
            "step": self.step,
            "result": self.result,
            "holding": self.holding,
            "stats": self.stats,
            "pick": s.pick_name,
            "place": s.place_name,
            "conf": s.conf,
            "clearance_cm": int(round(s.clearance_m * 100)),
            "place_top_cm": None if top is None else round(top * 100, 1),
            "detections": self.detections,
            "fps": round(self.fps, 1),
            "jaw_cm": None if jaw is None else round(jaw * 100, 1),
            "jaw_max_cm": round(GRIPPER_MAX_DISTANCE_M * 100, 1),
            "drop_zone": self._drop_zone(),
            "dry_run": self.dry_run,
            "robot": self.robot_ready,
            "parked": self.parked,
            "can_grasp": s.T_hand_eye is not None,
        }
        if with_log:
            out["log"] = list(self.log)[-120:]
        return out

    def _drop_zone(self) -> Optional[dict[str, Any]]:
        """Height the blue zone was computed for and its reachable cells (0 = nowhere to drop)."""
        if self.session.reach is None:
            return None
        z, cells = self.session.reach.place_zone
        return None if z is None else {"height_cm": round(z * 100, 1), "cells": cells}

    # ------------------------------------------------------------------ E-stop

    def check_stop(self) -> None:
        if self._stop.is_set():
            self._freeze()
            raise Aborted()

    def _freeze(self) -> None:
        """Cancel the trajectory being sent and hold the arm at its current joint angles."""
        s = self.session
        if not s.robot_running:
            return
        with self._freeze_lock:
            hold_position(s.controller, s.rebotarm)

    def emergency_stop(self) -> None:
        self._stop.set()
        self._freeze()
        self.status = "stopped"
        self.pose = "other" if self.pose != "rest" else "rest"
        self.message = "Emergency stop - the arm is holding still"
        print("[E-STOP] Pressed: trajectory cancelled, arm holding position")

    def clear_stop(self) -> bool:
        """Staff: release the latch without moving (only once the interrupted command has ended)."""
        if self.status != "stopped" or self._worker_alive():
            return False
        self._stop.clear()
        self.status, self.message = "idle", "Ready"
        print("[E-STOP] Cleared")
        return True

    # ------------------------------------------------------------------ commands

    def _worker_alive(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def _submit(self, action: str, fn: Callable[[], None], allow_when_stopped: bool = False) -> tuple[bool, str]:
        with self._worker_lock:
            if self.quitting and action != "quit":
                return False, "Shutting down"
            if not self.robot_ready:
                return False, "The robot arm is not connected"
            if self.status in ("starting", "error"):
                return False, self.message
            if self.status == "stopped" and not allow_when_stopped:
                return False, "Emergency stop is active - press Go Home first"
            if self._worker_alive() and (not allow_when_stopped or self._worker.name == action):
                return False, "The arm is busy"
            self.action = action
            self._worker = threading.Thread(target=self._run, args=(action, fn), name=action, daemon=True)
            self._worker.start()
            return True, "ok"

    def _run(self, action: str, fn: Callable[[], None]) -> None:
        self.parked = False
        if action == "quit":
            self.status = "quitting"
        elif self.status != "stopped":
            self.status = "busy"
        try:
            fn()
        except Aborted:
            print(f"[{action}] Stopped by the emergency stop")
            if action == "pick":
                self.result = {"ok": False, "text": "Stopped by the emergency stop"}
            if action == "quit":
                self.quitting = False
                print("[Quit] Cancelled: the program keeps running")
        except Exception as exc:
            print(f"[{action}] Error: {exc}")
            if action == "pick":
                self.result = {"ok": False, "text": "Something went wrong - ask a staff member"}
            if action == "quit":
                self.quitting = False
                self.status, self.message = "error", "Parking failed - ask a staff member"
                self.error = f"Parking before quitting failed: {exc}"
            self.pose = "other"
        finally:
            if threading.current_thread() is self._worker:   # not a stopped command Go Home replaced
                self.action = None
                self.step = 0
                if self.status == "busy":
                    self.status, self.message = "idle", "Ready"

    def go_home(self) -> tuple[bool, str]:
        """Always available: stops whatever is running first, then parks the arm."""
        if self._worker_alive() and self._worker.name != "home" and not self._stop.is_set():
            self.emergency_stop()
        return self._submit("home", self._do_home, allow_when_stopped=True)

    def request_quit(self, on_parked: Callable[[], None]) -> tuple[bool, str]:
        """Quit button: park the arm and close the gripper, then call on_parked (stops the server)."""
        if self.quitting:
            return False, "Already shutting down"
        if not self.robot_ready or not self.session.robot_running or self.status in ("starting", "error"):
            # Nothing to show progress for: shutdown() parks the arm (if connected) as the program exits.
            self.quitting, self.status, self.message = True, "quitting", "Shutting down..."
            on_parked()
            return True, "ok"
        if self._worker_alive() and self._worker.name != "quit" and not self._stop.is_set():
            self.emergency_stop()                          # end what is running before parking

        def run() -> None:
            self._wait_previous()
            self._stop.clear()
            self.status, self.message = "quitting", "Parking the arm and shutting down..."
            self._park(release_held=True)
            print("[Quit] Arm parked; shutting down")
            on_parked()

        self.quitting = True
        ok, msg = self._submit("quit", run, allow_when_stopped=True)
        if not ok:
            self.quitting = False
        return ok, msg

    def go_ready(self) -> tuple[bool, str]:
        return self._submit("ready", self._do_ready)

    def gripper(self, open_: bool) -> tuple[bool, str]:
        def run():
            gd = self._guarded_gripper()
            gd.open_gripper() if open_ else gd.open_gripper(0.0)
        return self._submit("gripper", run)

    def start_pick(self) -> tuple[bool, str]:
        if not self.robot_ready:
            return False, "The robot arm is not connected"
        if self.session.T_hand_eye is None:
            return False, "Camera calibration missing - picking disabled"
        return self._submit("pick", self._do_pick)

    def set_task(self, pick: str, place: str) -> tuple[bool, str]:
        pick, place = pick.strip(), place.strip()
        if not pick or not place:
            return False, "Choose both an object and a destination"
        if pick.lower() == place.lower():
            return False, "The object and the destination must be different"
        if self._worker_alive():
            return False, "Wait until the arm has finished"
        with self._model_lock:
            self.session.set_task(pick, place)
            self._last_results, self._last_grasps = [], []
            self.top.clear()                      # heights belonged to the old destination
        print(f"[UI] Task: pick '{pick}' -> place on/into '{place}'")
        return True, "ok"

    def set_settings(self, conf: Optional[float] = None, clearance_cm: Optional[int] = None) -> None:
        if conf is not None:
            self.session.conf = float(np.clip(conf, 0.01, 0.95))
        if clearance_cm is not None:
            self.session.clearance_m = int(np.clip(clearance_cm, 1, 30)) / 100.0

    # ------------------------------------------------------------------ motion

    def _note_gripper(self, name: str, args: tuple) -> None:
        if name == "grasp":
            self.holding = True
        elif name in ("open_for_object", "release_gripper") or (name == "open_gripper" and not (args and args[0] == 0.0)):
            self.holding = False

    def _guarded_controller(self) -> Any:
        return _Guard(self.session.controller, self)

    def _guarded_gripper(self) -> Any:
        return _Guard(self.session.grasp_driver, self, on_call=self._note_gripper)

    def _do_ready(self) -> None:
        self.message = "Moving to the ready position..."
        self.pose = "other"
        self.session.move_ready(self._guarded_controller())
        self.pose = "ready"

    def _wait_previous(self) -> None:
        """Wait for the command this one replaced (stopped by the E-stop) to end at its next step."""
        self.message = "Stopping the current step..."
        me = threading.current_thread()
        for t in threading.enumerate():
            if t is not me and t.name in ("pick", "ready", "gripper", "home") and t.is_alive():
                t.join(timeout=15.0)

    def _do_home(self) -> None:
        """Clear the stop once the interrupted command has ended, then fold the arm to rest."""
        self._wait_previous()
        self._stop.clear()
        self.status = "busy"
        self._park(release_held=False)
        self.message = "Parked"

    def _park(self, release_held: bool) -> None:
        """Fold the arm to its rest pose (via the ready pose when it is out working) and close the gripper.

        release_held: let go of a held object first (quit / exit), so the jaws can close fully;
        otherwise (Go Home) the gripper stays closed on it.
        """
        s = self.session
        print("[Park] Folding the arm to its rest pose")
        gripper = self._guarded_gripper()
        if self.holding and release_held:
            print("[Park] Letting go of the held object first")
            self.message = "Letting go of the object..."
            gripper.open_gripper()
        if self.pose != "rest":
            q, _, _ = s.rebotarm.get_state()
            if np.max(np.abs(q[: arm_joint_count(s.controller)])) > 0.3:
                # Come up and away from the table first, along the same path a pick returns on.
                self.message = "Going home..."
                try:
                    s.move_ready(self._guarded_controller())
                except Aborted:
                    raise
                except Exception as exc:
                    print(f"[Park] Ready pose skipped: {exc}")
            self.message = "Folding the arm..."
            move_joints_to_zero(s.controller, s.rebotarm, REST_MAX_VEL, check_stop=self.check_stop)
            time.sleep(0.5)                               # settle
        self.pose = "rest"
        if not self.holding:
            gripper.open_gripper(0.0)           # close the jaws: the rest pose expects them shut
            self.parked = True
        else:
            print("[Park] Still holding an object: gripper left closed on it (staff: Open gripper)")

    def _check_held(self) -> HeldCheck:
        self.check_stop()
        seq = self._frame_seq
        deadline = time.monotonic() + 2.0
        while self._frame_seq < seq + 5 and time.monotonic() < deadline:   # skip frames from while moving
            time.sleep(0.02)
        with self._frame_lock:
            color, depth = self._frame
        result = self.session.check_held(color, depth)
        if result.held is False:
            self.holding = False
        return result

    def _wait_fresh_detections(self, frames: int) -> None:
        seq = self._frame_seq
        deadline = time.monotonic() + 3.0
        while self._frame_seq < seq + frames and time.monotonic() < deadline:
            self.check_stop()
            time.sleep(0.02)

    def _fail(self, text: str) -> None:
        print(f"[G] {text}")
        self.result = {"ok": False, "text": text}

    def _do_pick(self) -> None:
        s = self.session
        pick, place = s.pick_name, s.place_name
        self.result = None
        self._log_mark = self._log_count
        self.stats["attempts"] += 1

        if self.pose != "ready":
            self.step = 0
            self._do_ready()
            self._wait_fresh_detections(s.infer_every * 3)
        self.message = f"Picking up the {pick}"

        print(f"\n[G] Capture and execute: pick '{pick}' -> '{place}', "
              f"conf >= {s.conf:.2f}, drop {s.clearance_m * 100:.0f} cm above it")
        print(f"[Step1] Detect {pick} and {place}")
        with self._frame_lock:
            color, depth = self._frame
        if color is None or depth is None:
            return self._fail("The camera has no picture")
        try:
            with self._model_lock:
                plan = s.plan(color, depth)
        except PlanError as exc:
            print(f"[G] {exc}")
            self.result = {"ok": False, "text": {
                "no_pick": f"I can't see the {pick}",
                "no_place": f"I can't see the {place}",
                "pick_unreachable": f"The {pick} is out of reach - move it into the green area",
                "place_unreachable": f"The {place} is out of reach - move it into the blue area",
            }.get(exc.reason, str(exc))}
            return

        check_held = self._check_held if s.held_check_enabled else None
        self.pose = "other"
        ok = s.execute(plan, dry_run=self.dry_run, check_held=check_held,
                       controller=self._guarded_controller(), grasp_driver=self._guarded_gripper())

        if ok:                                    # the sequence ends at the ready pose
            self.pose = "ready"
            self.stats["success"] += 1
            self.result = {"ok": True, "text": f"Done! The {pick} is on the {place}."}
            print("[G] Place completed successfully!")
        else:
            self.result = {"ok": False, "text": self._failure_text(pick)}
            print("[G] Place failed")
            self.message = "Going back..."
            self._do_ready()

    def _failure_text(self, pick: str) -> str:
        text = " ".join(self._log_since_mark()).lower()
        if "too wide" in text:
            return f"The {pick} is too wide for the gripper"
        if "empty grasp" in text:
            return "Missed it - the gripper came up empty. Try again!"
        if "ik failed" in text:
            return "The arm couldn't reach that spot - try moving the object closer"
        return "That didn't work - try again"

    # ------------------------------------------------------------------ camera

    def _camera_loop(self) -> None:
        frame_index = 0
        fps_count, fps_timer = 0, time.perf_counter()
        while not self._shutdown.is_set():
            try:
                color, depth = self.session.cam.get_frame()
            except Exception as exc:
                print(f"[Camera] {exc}")
                time.sleep(0.5)
                continue
            if color is None or depth is None:
                continue
            with self._frame_lock:
                self._frame = (color, depth)
                self._frame_seq += 1

            frame_index += 1
            fps_count += 1
            now = time.perf_counter()
            if now - fps_timer >= 1.0:
                self.fps = fps_count / (now - fps_timer)
                fps_count, fps_timer = 0, now

            try:
                self._update_view(color, depth, frame_index)
            except Exception as exc:
                print(f"[Camera] view: {exc}")

    def _update_view(self, color: np.ndarray, depth: np.ndarray, frame_index: int) -> None:
        s = self.session
        pick, place = s.pick_name, s.place_name
        inferred = False
        if frame_index % s.infer_every == 0 or not self._last_results:
            with self._model_lock:
                self._last_results, self._last_grasps = s.detect(color, depth)
            inferred = True
        results, grasps = self._last_results, self._last_grasps

        best_pick = best_match(grasps, pick)
        best_place = best_match(grasps, place)
        # Measured only at the lookout pose: while the arm moves, the frame and the arm pose read
        # for it are not in sync. Kept while the destination is out of view (median of recent values).
        at_lookout = self.pose == "ready" and self.action is None
        if inferred and at_lookout and self.robot_ready:
            self.top.add(s.measure_top(results, grasps, best_place, depth))

        if inferred:
            self.detections = [
                {"name": g.class_name, "conf": round(float(g.conf), 2), "valid": g.is_valid,
                 "role": "pick" if g is best_pick else "place" if g is best_place else None,
                 "width_cm": None if g.jaw_width_m is None else round(float(g.jaw_width_m) * 100, 1)}
                for g in grasps
            ]

        view = color
        # Zones are computed for the ready pose: only meaningful while the arm is parked there.
        if s.reach is not None and at_lookout:
            s.reach.request_place_height((self.top.value or 0.0) + s.clearance_m)
            view = color.copy()
            s.reach.draw(view, legend=False)               # the page shows its own legend
        display = render_display(view, grasps, best_pick, best_place, "", pick, place)
        ok, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            self.jpeg = buf.tobytes()
            self.jpeg_seq += 1
