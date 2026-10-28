#!/usr/bin/env python3
"""Raw SLAM fed live by a phone camera over Wi-Fi.

This script hosts a small web server. Open its address in Safari (or any browser) on the phone,
tap Start, and the phone streams its camera as JPEG frames to the PC. The PC runs the same raw SLAM
as the video app (raw_slam.RawSlam through raw_session.RawSession: live tracking + ORB anchors,
nothing else) and draws the map and trajectory in the usual 3D map window. A second window shows
the phone's picture with the tracked points.

  python phone_slam.py

Phone and PC must be on the same network. iPhones only allow camera access on https pages, so the
server uses https with a self-signed certificate (made once with openssl, in .phone-cert/): Safari
warns about it, tap "Show Details" -> "visit this website". The address printed at start-up carries
a random access token; without it the page and the stream are refused. Hold the phone in one
orientation (landscape works best): the camera model is fixed by the first frame. If the PC's
firewall blocks the port, open it (Fedora: sudo firewall-cmd --add-port=8443/tcp).

Keys in the PC windows: R = start a new map, Q / Esc = quit.
"""

from __future__ import annotations

import argparse
import asyncio
import hmac
import secrets
import socket
import ssl
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from aiohttp import WSMsgType, web

from raw_session import RawSession

HERE = Path(__file__).resolve().parent
PAGE = HERE / "phone_page.html"
CERT_DIR = HERE / ".phone-cert"
PREVIEW_WINDOW = "Phone camera"
PREVIEW_WIDTH = 640


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8443, help="port of the web server (default: %(default)s)")
    parser.add_argument("--fov-deg", type=float, default=65.0, help="field of view along the LONGER side of the picture, in degrees (default: %(default)g; iPhone main camera is about 65-70)")
    parser.add_argument("--size", type=int, default=640, help="the phone scales its pictures so the longer side is this many pixels (default: %(default)s)")
    parser.add_argument("--jpeg", type=float, default=0.8, help="JPEG quality sent by the phone, 0-1 (default: %(default)g)")
    parser.add_argument("--token", default=None, help="access token (default: random for every run)")
    parser.add_argument("--host-ip", default=None, help="address to put in the printed URL and certificate (default: this PC's LAN address)")
    parser.add_argument("--no-tls", action="store_true", help="plain http (only the PC's own browser can use the camera that way, via localhost)")
    parser.add_argument("--no-window", action="store_true", help="no PC windows, just serve and print progress")
    return parser.parse_args(argv)


def lan_address() -> str:
    """The address other devices on the network reach this PC at."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("10.255.255.255", 1))  # nothing is sent; this only picks the outgoing interface
        return probe.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        probe.close()


def ensure_certificate(address: str) -> tuple[Path, Path]:
    """Self-signed certificate for `address` (and localhost), created with openssl and reused while the address stays the same."""
    CERT_DIR.mkdir(exist_ok=True)
    certificate, key, subject = CERT_DIR / "cert.pem", CERT_DIR / "key.pem", CERT_DIR / "address.txt"
    if certificate.is_file() and key.is_file() and subject.is_file() and subject.read_text() == address:
        return certificate, key
    try:
        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "365",
                "-keyout", str(key), "-out", str(certificate), "-subj", "/CN=drone-sim-phone",
                "-addext", f"subjectAltName=IP:{address},IP:127.0.0.1,DNS:localhost",
            ],
            check=True, capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"Could not create a certificate with openssl ({error}). Install openssl or use --no-tls.")
    key.chmod(0o600)
    subject.write_text(address)
    return certificate, key


class PhoneSlam:
    """The SLAM session plus the lock that lets the server thread feed it while the main thread draws it."""

    def __init__(self, fov_deg: float) -> None:
        self.fov_deg = fov_deg
        self.lock = threading.Lock()
        self.session: RawSession | None = None
        self.generation = 0  # grows whenever a new session (map) is started
        self.size: tuple[int, int] | None = None
        self.last_frame: np.ndarray | None = None
        self.frames = 0
        self.reset_requested = False
        self.started = time.monotonic()

    def intrinsic(self, width: int, height: int) -> np.ndarray:
        focal = max(width, height) / (2.0 * np.tan(np.radians(self.fov_deg) / 2.0))
        return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])

    def request_reset(self) -> None:
        with self.lock:
            self.reset_requested = True

    def process(self, frame: np.ndarray) -> str:
        """Feed one BGR frame; returns the line the phone shows."""
        with self.lock:
            height, width = frame.shape[:2]
            if self.reset_requested:
                self.session, self.reset_requested = None, False
            if self.session is None:
                self.session = RawSession(frame, self.intrinsic(width, height))
                self.size = (width, height)
                self.generation += 1
                print(f"New map: {width}x{height} pictures, fov {self.fov_deg:g} deg over the longer side")
            elif (width, height) != self.size:
                return f"Picture is {width}x{height}, the map expects {self.size[0]}x{self.size[1]}: turn the phone back or tap Reset map"
            else:
                self.session.process(frame)
            self.frames += 1
            self.last_frame = frame
            return self.session.describe()


def build_app(state: PhoneSlam, args: argparse.Namespace) -> web.Application:
    def allowed(request: web.Request) -> bool:
        return hmac.compare_digest(request.query.get("token", ""), args.token)

    async def page(request: web.Request) -> web.StreamResponse:
        if not allowed(request):
            return web.Response(status=403, text="Open the address printed in the terminal; it contains the access token.")
        return web.FileResponse(PAGE)

    async def config(request: web.Request) -> web.StreamResponse:
        if not allowed(request):
            return web.Response(status=403)
        return web.json_response({"size": args.size, "jpeg": args.jpeg})

    async def stream(request: web.Request) -> web.StreamResponse:
        if not allowed(request):
            return web.Response(status=403)
        socket_ = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024)
        await socket_.prepare(request)
        print(f"Phone connected from {request.remote}")
        loop = asyncio.get_running_loop()
        async for message in socket_:
            if message.type == WSMsgType.TEXT and message.data == "reset":
                state.request_reset()
            elif message.type == WSMsgType.BINARY:
                frame = cv2.imdecode(np.frombuffer(message.data, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    await socket_.send_str("Could not decode that picture")
                    continue
                await socket_.send_str(await loop.run_in_executor(None, state.process, frame))
        print("Phone disconnected")
        return socket_

    app = web.Application()
    app.add_routes([web.get("/", page), web.get("/config", config), web.get("/ws", stream)])
    return app


def serve_forever(app: web.Application, args: argparse.Namespace, context: ssl.SSLContext | None, stop: threading.Event, failed: list[str]) -> None:
    """Runs the web server on its own thread (the main thread belongs to the OpenGL window)."""

    async def run() -> None:
        runner = web.AppRunner(app)
        await runner.setup()
        try:
            await web.TCPSite(runner, "0.0.0.0", args.port, ssl_context=context).start()
        except OSError as error:
            failed.append(f"Cannot listen on port {args.port}: {error}")
            return
        while not stop.is_set():
            await asyncio.sleep(0.2)
        await runner.cleanup()

    asyncio.run(run())


def draw_preview(state: PhoneSlam) -> np.ndarray | None:
    with state.lock:
        if state.session is None or state.last_frame is None:
            return None
        image = state.last_frame.copy()
        points = state.session.track_points.copy()
        text = state.session.describe()
    for x, y in points:
        cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), -1)
    scale = PREVIEW_WIDTH / image.shape[1]
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    for color, thickness in (((0, 0, 0), 3), ((255, 255, 255), 1)):
        cv2.putText(image, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, thickness, cv2.LINE_AA)
    return image


def main() -> None:
    args = parse_args()
    if args.token is None:
        args.token = secrets.token_urlsafe(6)
    address = args.host_ip or lan_address()
    context = None
    if not args.no_tls:
        certificate, key = ensure_certificate(address)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(certificate, key)

    state = PhoneSlam(args.fov_deg)
    stop, failed = threading.Event(), []
    server = threading.Thread(target=serve_forever, args=(build_app(state, args), args, context, stop, failed), daemon=True)
    server.start()
    time.sleep(0.5)
    if failed:
        raise SystemExit(failed[0])

    scheme = "http" if args.no_tls else "https"
    print(f"\nOpen this on the phone (same network):\n\n    {scheme}://{address}:{args.port}/?token={args.token}\n")
    if not args.no_tls:
        print("Safari warns about the certificate: Show Details -> visit this website. Then tap Start.\n")

    map_view = None
    if not args.no_window:
        from map_renderer import MapRenderer

        map_view = MapRenderer(960, 540)
        map_view.setup_window()
        map_view.set_mesh(None)
    last_report = 0.0
    try:
        while True:
            if map_view is not None:
                if not map_view.poll():
                    break
                with state.lock:
                    session = state.session
                    if session is not None:
                        map_view.camera_rotation = session.world_rotation
                        map_view.show(session.trajectory, session.map_store)
                preview = draw_preview(state)
                if preview is not None:
                    cv2.imshow(PREVIEW_WINDOW, preview)
                key = cv2.waitKey(15) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    state.request_reset()
            else:
                time.sleep(0.2)
                if time.monotonic() - last_report > 2.0 and state.session is not None:
                    last_report = time.monotonic()
                    with state.lock:
                        print(f"frames {state.frames}  {state.session.describe()}")
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.join(timeout=3.0)
        if state.session is not None:
            print(f"\nStopped after {state.frames} frames: {state.session.describe()}")
        if map_view is not None:
            map_view.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
