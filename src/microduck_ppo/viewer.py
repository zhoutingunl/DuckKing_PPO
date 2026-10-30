"""Loopback-only experiment dashboard; no cloud service or CDN is used."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .navigation import NavigationConfig, resolve_assets
from .navigation_artifacts import read_json, run_summary

STATIC = Path(__file__).with_name("web")
ACTIVE = {"starting", "training", "rendering", "evaluating"}


def confined(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Path outside run directory")
    return path


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, runs_root, **kwargs):
        self.runs_root = runs_root
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def log_message(self, format, *args):
        pass

    def json_response(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def run_path(self, run_id):
        if not re.fullmatch(r"[\w.-]+", run_id) or run_id in {".", ".."}:
            raise ValueError("Invalid run id")
        return confined(self.runs_root, run_id)

    def do_GET(self):
        parts = unquote(urlparse(self.path).path).strip("/").split("/")
        try:
            if parts == ["api", "runs"]:
                runs = []
                for path in self.runs_root.glob("*/run.json"):
                    manifest = read_json(path)
                    runs.append(
                        {
                            "id": path.parent.name,
                            "status": manifest["status"],
                            "created_at": manifest["created_at"],
                            "timesteps": manifest.get("timesteps", 0),
                        }
                    )
                return self.json_response(sorted(runs, key=lambda r: r["created_at"], reverse=True))
            if len(parts) == 3 and parts[:2] == ["api", "runs"]:
                run = self.run_path(parts[2])
                payload = run_summary(run)
                episodes = run / "episodes.jsonl"
                if episodes.exists():
                    rows = []
                    for line in episodes.read_text().splitlines():
                        try:
                            rows.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                    stride = max(1, len(rows) // 500)
                    payload["episodes"] = rows[::stride]
                return self.json_response(payload)
            if len(parts) >= 3 and parts[0] == "artifacts":
                run = self.run_path(parts[1])
                path = confined(run, "/".join(parts[2:]))
                if path.suffix not in {".mp4", ".jpg", ".json", ".zip", ".log"}:
                    raise ValueError("Unsupported artifact")
                return self.artifact(path)
            return super().do_GET()
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as error:
            return self.json_response({"error": str(error)}, 404)

    def artifact(self, path):
        size = path.stat().st_size
        start, end = 0, size - 1
        partial_content = False
        if self.headers.get("Range"):
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers["Range"])
            if not match or not any(match.groups()):
                self.send_error(416)
                return
            left, right = match.groups()
            start = int(left) if left else max(0, size - int(right))
            end = min(int(right), size - 1) if left and right else size - 1
            if start > end or start >= size:
                self.send_error(416)
                return
            partial_content = True
        self.send_response(206 if partial_content else 200)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if partial_content:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with path.open("rb") as stream:
            stream.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = stream.read(min(65536, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    break
                remaining -= len(chunk)

    def do_POST(self):
        origin = self.headers.get("Origin")
        if origin and origin != f"http://{self.headers.get('Host')}":
            return self.json_response({"error": "Local requests only"}, 403)
        parts = urlparse(self.path).path.strip("/").split("/")
        try:
            if parts == ["api", "train"]:
                length = int(self.headers.get("Content-Length", 0))
                if length > 4096:
                    raise ValueError("Request too large")
                data = json.loads(self.rfile.read(length))
                budget = int(data.get("timesteps", 100000))
                envs = int(data.get("envs", 2))
                if budget not in {10000, 100000, 300000, 1000000} or envs not in {1, 2, 4}:
                    raise ValueError("Unsupported training preset")
                resolve_assets(NavigationConfig())
                with self.server.launch_lock:
                    if self.server.learner and self.server.learner.poll() is None:
                        return self.json_response({"error": "已有训练正在运行"}, 409)
                    for path in self.runs_root.glob("*/run.json"):
                        record = read_json(path)
                        if record["status"] in ACTIVE and record.get("pid"):
                            try:
                                os.kill(record["pid"], 0)
                            except ProcessLookupError:
                                continue
                            return self.json_response({"error": "已有本地训练正在运行"}, 409)
                    run_id = time.strftime("navigation-%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
                    run = self.run_path(run_id)
                    run.mkdir(parents=True)
                    with (run / "learner.log").open("w") as log:
                        self.server.learner = subprocess.Popen(
                            [
                                sys.executable,
                                "-m",
                                "microduck_ppo.navigation_train",
                                "--output",
                                str(run),
                                "--timesteps",
                                str(budget),
                                "--envs",
                                str(envs),
                            ],
                            stdout=log,
                            stderr=log,
                        )
                return self.json_response({"id": run_id}, 202)
            if len(parts) == 4 and parts[:2] == ["api", "runs"] and parts[3] == "stop":
                run = self.run_path(parts[2])
                manifest = read_json(run / "run.json")
                if manifest["status"] not in {"starting", "training"}:
                    return self.json_response({"error": "当前训练已停止或正在评估"}, 409)
                (run / "STOP").touch()
                return self.json_response({"status": "stop_requested"})
            self.json_response({"error": "Unknown operation"}, 404)
        except (ValueError, FileNotFoundError) as error:
            self.json_response({"error": str(error)}, 400)


def make_server(root, port=8780):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, runs_root=root))
    server.launch_lock = threading.Lock()
    server.learner = None
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--port", type=int, default=8780)
    args = parser.parse_args()
    server = make_server(args.runs_root, args.port)
    print(f"MicroDuck learning viewer: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
