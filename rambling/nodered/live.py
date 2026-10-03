"""Deploy flows to a real headless Node-RED and probe HTTP endpoints.

Requires `npm install` in ./nodered (installs node-red locally).
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

RED_JS = Path(__file__).resolve().parents[2] / "nodered" / "node_modules" / "node-red" / "red.js"


def available() -> bool:
    return RED_JS.exists() and shutil.which("node") is not None


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class NodeRedServer:
    def __init__(self):
        self.port = _free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.userdir = tempfile.mkdtemp(prefix="rambling-nr-")
        self.proc = None

    def __enter__(self):
        self.proc = subprocess.Popen(
            ["node", str(RED_JS), "-u", self.userdir, "-p", str(self.port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env={**os.environ, "NODE_ENV": "test"})
        deadline = time.time() + 60
        while time.time() < deadline:
            try:
                self.request("GET", "/flows")
                return self
            except OSError:
                time.sleep(0.3)
        self.__exit__(None, None, None)
        raise RuntimeError("Node-RED did not start")

    def __exit__(self, *exc):
        if self.proc:
            self.proc.terminate()
            self.proc.wait(timeout=20)
        shutil.rmtree(self.userdir, ignore_errors=True)

    def request(self, method, path, body=None, headers=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode()

    def node_types(self) -> set[str]:
        _, body = self.request("GET", "/nodes", headers={"Accept": "application/json"})
        return {t for module in json.loads(body) for t in module.get("types", [])}

    def deploy(self, nodes: list[dict]) -> tuple[bool, str]:
        status, body = self.request("POST", "/flows", nodes, {"Node-RED-Deployment-Type": "full"})
        if status not in (200, 204):
            return False, f"deploy returned {status}: {body[:200]}"
        status, body = self.request("GET", "/flows")
        deployed = {n["id"] for n in json.loads(body)}
        missing = {n["id"] for n in nodes} - deployed
        if missing:
            return False, f"{len(missing)} nodes missing after deploy"
        time.sleep(0.5)
        return True, "deployed"

    def probe(self, p: dict) -> tuple[bool, str]:
        """Probe: {method, path, json?, expect_status?, expect_contains?, expect_file?}."""
        status, body = self.request(p["method"], p["path"], p.get("json"))
        if status != p.get("expect_status", 200):
            return False, f"{p['method']} {p['path']}: status {status}"
        if "expect_contains" in p and p["expect_contains"] not in body:
            return False, f"{p['method']} {p['path']}: body {body[:80]!r} lacks {p['expect_contains']!r}"
        if "expect_file" in p:
            f = Path(p["expect_file"]["path"])
            if not f.exists() or p["expect_file"]["contains"] not in f.read_text():
                return False, f"file {f} lacks {p['expect_file']['contains']!r}"
        return True, "ok"
