"""Stop-and-hold and fold-to-rest moves for the web UI, which the arm SDK has no public call for.

RebotArmEndPose runs each trajectory in a background send loop. These functions use its
internals (_stop_send, _send_thread, _send_loop, _traj, _q_target, _qd_target, _vlim_override,
_moving, _n), as of reBotArm_control_py 0.1.0 (commit 6415d43). They are kept in this one file
so they are easy to replace once the SDK offers a public stop / joint move.

hold_position(): cancel the trajectory being sent and hold the current joint angles.
move_joints_to_zero(): minimum-jerk joint move to all zeros (the folded rest pose), through the
    SDK's own send loop, so hold_position() can interrupt it like any other move.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

import numpy as np


def arm_joint_count(controller: Any) -> int:
    return int(controller._n)


def hold_position(controller: Any, rebotarm: Any) -> None:
    """Cancel the running trajectory and hold the arm where it is (motors stay powered)."""
    controller._stop_send.set()
    thread = controller._send_thread
    if thread is not None and thread is not threading.current_thread():
        thread.join(timeout=0.5)
    q, _, _ = rebotarm.get_state()
    controller._q_target[:] = q[: controller._n]
    controller._qd_target[:] = 0.0
    controller._vlim_override = None
    controller._moving = False


def move_joints_to_zero(controller: Any, rebotarm: Any, max_vel: float,
                        check_stop: Callable[[], None] = lambda: None) -> None:
    """Fold the arm to all-zero joint angles at up to max_vel rad/s and wait for it.

    check_stop() is called before and after the move; it raises to abandon the move
    (the caller then holds the arm with hold_position()).
    """
    q0 = rebotarm.get_state()[0][: controller._n].copy()
    err = float(np.max(np.abs(q0)))
    if err < 0.01:
        return
    duration = max(2.0, 2.0 * err / max_vel)          # min-jerk peak speed = 1.875 x mean speed
    steps = max(2, int(duration * 50))
    s = np.linspace(0.0, 1.0, steps)
    blend = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    check_stop()
    controller._stop_send.set()
    if controller._send_thread is not None:
        controller._send_thread.join(timeout=5.0)
    controller._traj = [q0 * (1.0 - b) for b in blend]
    controller._moving = True
    controller._stop_send.clear()
    controller._send_thread = threading.Thread(target=controller._send_loop, args=(duration,), daemon=True)
    controller._send_thread.start()
    controller._send_thread.join(timeout=duration + 5.0)
    check_stop()
