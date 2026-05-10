#!/usr/bin/env python3
"""
M75 serve: tiny local HTTP server that hosts the label tool and accepts
POST /save_label requests. Saves are written directly to
data/manual labeling/, no browser picker required.

Usage:
    py scripts/m75_serve.py
Then open the printed URL in Chrome/Edge.
"""

from __future__ import annotations

import http.server
import json
import os
import socketserver
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR = ROOT / 'data' / 'manual labeling'
PORT = 8765


class Handler(http.server.SimpleHTTPRequestHandler):
    def _send_json(self, code, payload):
        body = json.dumps(payload).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.split('?', 1)[0] == '/save_label':
            self._send_json(200, {'ok': True, 'labels_dir': str(LABELS_DIR)})
            return
        return super().do_GET()

    def do_POST(self):
        if self.path.split('?', 1)[0] != '/save_label':
            self._send_json(404, {'ok': False, 'error': 'unknown endpoint'})
            return
        try:
            length = int(self.headers.get('Content-Length', 0))
            raw = self.rfile.read(length).decode('utf-8')
            payload = json.loads(raw)
            key = str(payload.get('key', ''))
            safe_key = ''.join(c for c in key if c.isalnum() or c in ('_', '-'))
            if not safe_key:
                raise ValueError('missing or invalid key')
            LABELS_DIR.mkdir(parents=True, exist_ok=True)
            out_file = LABELS_DIR / f'm75_labels_{safe_key}.json'
            out_file.write_text(json.dumps(payload, indent=2), encoding='utf-8')
            self._send_json(200, {'ok': True, 'path': str(out_file)})
        except Exception as e:
            self._send_json(500, {'ok': False, 'error': str(e)})


def main():
    os.chdir(ROOT)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    url = f'http://127.0.0.1:{PORT}/output/m75_label_tool.html'
    print(f'Serving project root: {ROOT}')
    print(f'Saves go to:           {LABELS_DIR}')
    print(f'Open:                  {url}')
    print('(Ctrl+C to stop)')
    try:
        webbrowser.open(url)
    except Exception:
        pass
    with socketserver.TCPServer(('127.0.0.1', PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print('\nstopped.')


if __name__ == '__main__':
    main()
