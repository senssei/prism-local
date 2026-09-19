"""
foundry_wsl.proxy: Stable Reverse Proxy for Microsoft Foundry Local
Resolves active ephemeral daemon ports from ~/.foundry/daemon.json and provides
a stable static endpoint (e.g. http://127.0.0.1:5272/v1) for agents and tools.
"""

import http.server
import json
import logging
import socketserver
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger("foundry_wsl.proxy")

def resolve_daemon_url() -> Optional[str]:
    """Reads the active daemon URL from ~/.foundry/daemon.json."""
    daemon_json = Path.home() / ".foundry/daemon.json"
    if not daemon_json.exists():
        return None
    try:
        data = json.loads(daemon_json.read_text())
        urls = data.get("urls", [])
        if urls:
            return urls[0].rstrip("/")
    except Exception:
        pass
    return None

class FoundryProxyHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self._proxy_request("GET")

    def do_POST(self):
        self._proxy_request("POST")

    def do_PUT(self):
        self._proxy_request("PUT")

    def do_DELETE(self):
        self._proxy_request("DELETE")

    def do_OPTIONS(self):
        self._proxy_request("OPTIONS")

    def _proxy_request(self, method: str):
        target_base = resolve_daemon_url()
        if not target_base:
            self.send_response(502)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            err_payload = json.dumps({"error": "Foundry local daemon is not running or no URLs in daemon.json"}).encode()
            self.wfile.write(err_payload)
            return

        target_url = f"{target_base}{self.path}"
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else None

        # Copy headers (excluding Host)
        req_headers = {}
        for key, val in self.headers.items():
            if key.lower() not in ["host", "content-length"]:
                req_headers[key] = val

        try:
            req = urllib.request.Request(target_url, data=body, headers=req_headers, method=method)
            with urllib.request.urlopen(req, timeout=120) as resp:
                self.send_response(resp.status)
                for k, v in resp.getheaders():
                    if k.lower() not in ["transfer-encoding", "content-length"]:
                        self.send_header(k, v)
                resp_body = resp.read()
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            for k, v in e.headers.items():
                if k.lower() not in ["transfer-encoding", "content-length"]:
                    self.send_header(k, v)
            err_body = e.read()
            self.send_header("Content-Length", str(len(err_body)))
            self.end_headers()
            self.wfile.write(err_body)
        except Exception as ex:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": f"Proxy forward error: {str(ex)}"}).encode())

    def log_message(self, format, *args):
        # Suppress verbose standard HTTP server logging unless DEBUG
        pass

def run_proxy(port: int = 5272, host: str = "127.0.0.1"):
    """Starts the static reverse proxy server."""
    class ReusableServer(socketserver.TCPServer):
        allow_reuse_address = True

    with ReusableServer((host, port), FoundryProxyHandler) as httpd:
        target = resolve_daemon_url() or "Waiting for daemon..."
        print(f"🚀 Foundry WSL Proxy listening at http://{host}:{port}/")
        print(f"   Forwarding requests to active daemon: {target}")
        print("   Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down proxy.")
