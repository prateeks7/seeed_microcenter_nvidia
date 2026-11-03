"""
Pick-and-place demo based on YOLO: pick one object and drop it into/onto another.

Everything is adjustable live in the camera window: the pick object and destination
labels (typed in the window), the detection confidence threshold and the release clearance
(sliders). Starting values come from config place_task / detection, or --pick / --place.

The object is released at (measured top of the destination + clearance): a cup's rim
height is measured from depth, a flat spot on the table is ~0.

Reach zones are drawn on the live view (green = can pick, blue stripes = can place), and
every pose is checked with IK before the arm moves.

Workflow:
  1. Detect the pick object and the destination
  2. Go to grasp the pick object
  3. Grasp it and lift to ready position (the wrist camera checks it is held)
  4. Move above the destination and release
  5. Return to ready position

Keys:
  P: type a new pick object (Enter = apply, Esc = cancel)
  D: type a new destination (Enter = apply, Esc = cancel)
  G: capture and execute pick and place
  R: resume live preview
  Q/Esc: release gripper, home, and exit

Usage:
    python scripts/set.py
    python scripts/set.py --pick "orange ball" --place "cup" --release-clearance 0.05
    python scripts/set.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pick-and-place demo")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--dry-run", action="store_true", help="estimate only; do not move the arm")
    parser.add_argument("--skip-zero-check", action="store_true",
                        help="do not check that all motors read ~0 (arm at rest, gripper closed) before starting")
    parser.add_argument("--pick", help="starting pick object label (default: config place_task.pick)")
    parser.add_argument("--place", help="starting destination label (default: config place_task.place)")
    parser.add_argument("--release-clearance", type=float,
                        help="starting drop height above the destination's top, m (default: config place_task)")
    return parser.parse_args()


_KEYS_TEXT = "[P] pick  [D] destination  [G] go  [R] resume  [Q] quit"


def _draw_panel(
    display: np.ndarray,
    pick_name: str,
    place_name: str,
    conf: float,
    clearance_cm: int,
    editing: Optional[str],
    edit_text: str,
    place_top_m: Optional[float] = None,
) -> None:
    """Settings panel in the top-right corner; shows the label being typed while editing."""
    top_text = f"top {place_top_m * 100:.0f} cm" if place_top_m is not None else "top: not seen"
    lines = [
        ("PICK:  " + (edit_text + "_" if editing == "pick" else pick_name), (120, 255, 140)),
        ("PLACE: " + (edit_text + "_" if editing == "place" else place_name) + f"   ({top_text})",
         (255, 200, 120)),
        (f"conf >= {conf:.2f}    drop {clearance_cm} cm above destination", (230, 230, 230)),
        ("typing...  Enter = apply   Esc = cancel" if editing else _KEYS_TEXT,
         (0, 215, 255) if editing else (180, 180, 180)),
    ]
    width, line_h = 600, 26
    x0, y0 = display.shape[1] - width - 10, 10
    shade = display.copy()
    cv2.rectangle(shade, (x0, y0), (x0 + width, y0 + line_h * len(lines) + 14), (0, 0, 0), -1)
    cv2.addWeighted(shade, 0.55, display, 0.45, 0, dst=display)
    for i, (text, color) in enumerate(lines):
        cv2.putText(display, text, (x0 + 10, y0 + 28 + line_h * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def main() -> int:
    args = parse_args()
    cfg = load_config(PROJECT_ROOT / args.config)
    session = PickPlaceSession(cfg, PROJECT_ROOT)
    if args.pick:
        session.pick_name = args.pick
    if args.place:
        session.place_name = args.place
    if args.release_clearance is not None:
        session.clearance_m = args.release_clearance
    if session.pick_name.lower().strip() == session.place_name.lower().strip():
        print("[Error] Pick object and destination must be different")
        return 1
    print(f"=== Task: pick '{session.pick_name}' -> place on/into '{session.place_name}', "
          f"release {session.clearance_m * 100:.0f} cm above its top ===")

    last_results: list[Any] = []
    last_grasps: list[GraspPose] = []
    frozen = False
    last_display: Optional[np.ndarray] = None
    frame_index = 0
    fps_counter = 0
    fps_timer = time.perf_counter()
    fps_value = 0.0

    window_name = "Pick & Place"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("conf %", window_name, int(np.clip(round(session.conf * 100), 1, 95)), 95, lambda _v: None)
    cv2.createTrackbar("clearance cm", window_name, int(np.clip(round(session.clearance_m * 100), 1, 30)), 30,
                       lambda _v: None)
    editing: Optional[str] = None   # "pick" / "place" while a label is being typed
    edit_text = ""
    top = TopTracker()              # measured height of the destination's top (live, smoothed)
    print(f"\n[Keys]  {_KEYS_TEXT}   (sliders: confidence threshold, release height)\n")

    try:
        session.open_camera()
        session.load_model()
        reach = session.build_reach()    # before the arm control loop starts, so the IK work does not compete
        if not args.skip_zero_check:
            session.check_zeros()
        session.connect_robot()
        print("[Robot] Move ready")
        session.move_ready()
        cam = session.cam

        def _check_held() -> HeldCheck:
            color = depth = None
            for _ in range(5):                  # skip frames buffered while the arm was moving
                color, depth = cam.get_frame()
            return session.check_held(color, depth, save_dir=PROJECT_ROOT / "Log")

        check_held = _check_held if session.held_check_enabled else None

        while True:
            color_bgr, depth_mm = cam.get_frame()
            if color_bgr is None or depth_mm is None:
                continue

            frame_index += 1
            fps_counter += 1
            now = time.perf_counter()
            if now - fps_timer >= 1.0:
                fps_value = fps_counter / (now - fps_timer)
                fps_counter = 0
                fps_timer = now

            # Sliders apply immediately (trackbars cannot go below 0, so clamp the minimums).
            session.conf = max(cv2.getTrackbarPos("conf %", window_name), 1) / 100.0
            clearance_cm = max(cv2.getTrackbarPos("clearance cm", window_name), 1)
            session.clearance_m = clearance_cm / 100.0
            pick_name, place_name = session.pick_name, session.place_name

            inferred = False
            if not frozen and (frame_index % session.infer_every == 0 or not last_results):
                last_results, last_grasps = session.detect(color_bgr, depth_mm)
                inferred = True

            best_pick = best_match(last_grasps, pick_name)
            best_place = best_match(last_grasps, place_name)
            if inferred:                                # destination top, measured on the same frame
                top.add(session.measure_top(last_results, last_grasps, best_place, depth_mm))

            status = f"{'FROZEN' if frozen else 'LIVE'} {fps_value:.1f}fps"
            if frozen and last_display is not None:
                display = last_display.copy()
                cv2.putText(display, "[FROZEN]", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 215, 255), 2)
            else:
                view = color_bgr
                if reach is not None:                   # zones under the detection boxes
                    # Blue zone at the destination's release height; follows lasting changes only.
                    reach.request_place_height((top.value or 0.0) + session.clearance_m)
                    view = color_bgr.copy()
                    reach.draw(view)
                display = render_display(view, last_grasps, best_pick, best_place, status,
                                         pick_name, place_name)
            _draw_panel(display, pick_name, place_name, session.conf, clearance_cm, editing, edit_text, top.value)

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            if editing is not None:
                if key in (13, 10):                     # Enter: apply the typed label
                    text = edit_text.strip()
                    other = place_name if editing == "pick" else pick_name
                    if not text:
                        print("[UI] Empty label ignored")
                    elif text.lower() == other.lower().strip():
                        print("[UI] Pick object and destination must be different")
                    else:
                        if editing == "pick":
                            session.set_task(text, place_name)
                        else:
                            session.set_task(pick_name, text)
                        last_results, last_grasps = [], []
                        frozen, last_display = False, None
                        top.clear()                     # heights belonged to the old destination
                        print(f"[UI] Task: pick '{session.pick_name}' -> place on/into '{session.place_name}'")
                    editing = None
                elif key == 27:                         # Esc: cancel typing (does not quit)
                    editing = None
                elif key in (8, 127):                   # Backspace
                    edit_text = edit_text[:-1]
                elif 32 <= key <= 126:
                    edit_text += chr(key)
                continue

            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("p"), ord("P")):
                editing, edit_text = "pick", ""
                continue
            if key in (ord("d"), ord("D")):
                editing, edit_text = "place", ""
                continue
            if key in (ord("r"), ord("R")):
                frozen = False
                last_display = None
                continue

            if key in (ord("g"), ord("G")):
                print(f"\n[G] Capture and execute: pick '{pick_name}' -> '{place_name}', "
                      f"conf >= {session.conf:.2f}, drop {clearance_cm} cm above it")
                print(f"[Step1] Detect {pick_name} and {place_name}")
                if not session.can_grasp:
                    print("[G] Hand-eye calibration unavailable")
                    continue
                snap_color, snap_depth = cam.get_frame()
                if snap_color is None or snap_depth is None:
                    print("[G] Frame capture failed")
                    continue
                try:
                    plan = session.plan(snap_color, snap_depth)
                except PlanError as exc:
                    print(f"[G] {exc}")
                    continue

                frozen = True
                last_results, last_grasps = plan.results, plan.grasps
                last_display = render_display(snap_color, plan.grasps, plan.pick, plan.place, "SNAPSHOT",
                                              pick_name, place_name)

                ok = session.execute(plan, dry_run=args.dry_run, check_held=check_held)
                print("[G] Place completed successfully!" if ok else "[G] Place failed")
                session.move_ready()

    except ZeroCheckError as exc:
        print(f"[Zero check] FAILED - not starting the arm. {exc}")
        return 1
    finally:
        print("\n[Exit] Release gripper and home")
        session.close()
        cv2.destroyAllWindows()
        print("Done.")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        raise SystemExit(130)
