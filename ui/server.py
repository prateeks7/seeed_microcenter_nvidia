"""Web UI for the pick-and-place demo (scripts/set.py as a touch-friendly web page).

Serves a page with the live camera view, object / destination choice, Start, an
always-visible EMERGENCY STOP and GO HOME, gripper and settings controls, and a staff panel
(ready pose, clear stop, log). E-stop holds the arm where it is without ending this program;
Go Home folds it to rest. Header logos and title come from config ui.logos / ui.title.

Quitting always parks the arm (rest pose, gripper closed): the Quit button parks first and then
stops the server, and Ctrl+C, `kill <pid>` (SIGTERM) and closing the terminal (SIGHUP) stop the
server and park on the way out. Only `kill -9` / power loss skip it.

Usage (rebotarm env):
    python ui/server.py                      # then open http://localhost:8000
    python ui/server.py --dry-run            # detect and plan, but do not run the pick motion
    python ui/server.py --no-robot           # camera + detection only (arm unplugged)
    python ui/server.py --host 0.0.0.0       # also reachable from a tablet / phone on the network
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ui.backend import LogTee, RobotApp

STATIC = Path(__file__).resolve().parent / "static"


class TaskIn(BaseModel):
    pick: str
    place: str


class SettingsIn(BaseModel):
    conf: Optional[float] = None
    clearance_cm: Optional[int] = None


class GripperIn(BaseModel):
    open: bool


class RetryIn(BaseModel):
    skip_zero_check: bool = False


def create_app(robot: RobotApp, request_exit: Callable[[], None],
               is_exiting: Callable[[], bool] = lambda: False) -> FastAPI:
    """request_exit stops the server; main() then parks the arm (robot.shutdown).
    is_exiting tells the endless video stream to end, so shutdown does not wait on open pages."""
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        robot.start()
        yield

    app = FastAPI(title="Pick & Place", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def _result(ok_msg: tuple[bool, str]) -> dict:
        ok, msg = ok_msg
        if not ok:
            raise HTTPException(status_code=409, detail=msg)
        return {"ok": True}

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/api/state")
    def state(log: bool = False):
        return robot.state(with_log=log)

    @app.get("/api/config")
    def config():
        return {"presets": robot.presets, "branding": robot.branding}

    # Plain `def` endpoints run in a thread pool, so a slow command never delays the E-stop.
    @app.post("/api/estop")
    def estop():
        robot.emergency_stop()
        return {"ok": True}

    @app.post("/api/home")
    def home():
        return _result(robot.go_home())

    @app.post("/api/clear_stop")
    def clear_stop():
        return _result((robot.clear_stop(), "Wait until the arm has stopped"))

    @app.post("/api/start")
    def start():
        return _result(robot.start_pick())

    @app.post("/api/ready")
    def ready():
        return _result(robot.go_ready())

    @app.post("/api/task")
    def task(body: TaskIn):
        return _result(robot.set_task(body.pick, body.place))

    @app.post("/api/settings")
    def settings(body: SettingsIn):
        robot.set_settings(body.conf, body.clearance_cm)
        return {"ok": True}

    @app.post("/api/gripper")
    def gripper(body: GripperIn):
        return _result(robot.gripper(body.open))

    @app.post("/api/quit")
    def quit_app():
        return _result(robot.request_quit(request_exit))

    @app.post("/api/retry")
    def retry(body: RetryIn):
        return _result((robot.retry_init(body.skip_zero_check), "Not in an error state"))

    @app.get("/video")
    async def video():
        async def frames():
            seq = -1
            while not is_exiting():
                if robot.jpeg is None or robot.jpeg_seq == seq:
                    await asyncio.sleep(0.02)
                    continue
                seq, jpg = robot.jpeg_seq, robot.jpeg
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + \
                      b"\r\n\r\n" + jpg + b"\r\n"

        return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pick-and-place web UI")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to allow other devices on the network")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dry-run", action="store_true", help="detect and plan only; no pick motion")
    parser.add_argument("--no-robot", action="store_true", help="camera and detection only")
    parser.add_argument("--skip-zero-check", action="store_true",
                        help="do not check that all motors read ~0 before starting")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    robot = RobotApp(PROJECT_ROOT / args.config, dry_run=args.dry_run, no_robot=args.no_robot,
                     skip_zero_check=args.skip_zero_check)
    sys.stdout = LogTee(robot, sys.stdout)

    server: Optional[uvicorn.Server] = None

    def request_exit() -> None:
        if server is not None:
            server.should_exit = True

    # Our handlers for Ctrl+C, kill and a closed terminal. uvicorn swaps in its own for SIGINT/SIGTERM
    # while serving and, once stopped, re-raises the signal into these, so the process is never ended
    # by the default handler before the arm is parked below.
    def on_signal(sig: int, _frame) -> None:
        if robot.quitting:
            print(f"[Exit] {signal.Signals(sig).name} ignored: parking the arm first "
                  "(kill -9 forces an exit WITHOUT parking)")
        else:
            request_exit()

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        if sig == signal.SIGHUP and signal.getsignal(sig) == signal.SIG_IGN:
            continue                                   # started with nohup: keep running when the terminal closes
        signal.signal(sig, on_signal)

    host = "localhost" if args.host in ("127.0.0.1", "0.0.0.0") else args.host
    print(f"=== Web UI: http://{host}:{args.port} ===")
    # The video stream never ends on its own: give open pages 2 s, then cancel them.
    app = create_app(robot, request_exit, is_exiting=lambda: server is not None and server.should_exit)
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="warning", timeout_graceful_shutdown=2)
    server = uvicorn.Server(config)
    try:
        server.run()
    finally:
        robot.shutdown()                               # park the arm, close the gripper, disconnect
        sys.stdout.flush()
        # Hardware is safe and closed: exit now, so native threads (torch, RealSense) still winding
        # down cannot abort the interpreter's teardown.
        os._exit(0)

if __name__ == "__main__":
    raise SystemExit(main())
