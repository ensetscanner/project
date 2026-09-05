"""
EnsetScan Vision — WSGI Application Entry Point
Serves the offline-first PWA (HTML, JS, CSS, JSON data, ML model files)
for hosting on platforms like PythonAnywhere or other WSGI servers.

Run locally:  python main.py   (defaults to http://0.0.0.0:8080)
WSGI callable: application    (for gunicorn / uWSGI / mod_wsgi)
"""

import os
import mimetypes
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import unquote

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = 'index.html'

# ---------------------------------------------------------------------------
# MIME types that Python's mimetypes may not detect perfectly.
# ---------------------------------------------------------------------------
EXTRA_MIME = {
    '.webmanifest': 'application/manifest+json',
    '.js': 'application/javascript',
    '.mjs': 'application/javascript',
    '.css': 'text/css',
    '.json': 'application/json',
    '.bin': 'application/octet-stream',
    '.png': 'image/png',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon',
    '.html': 'text/html',
    '.htm': 'text/html',
    '.woff2': 'font/woff2',
    '.woff': 'font/woff',
    '.ttf': 'font/ttf',
    '.map': 'application/json',
}

# Security: only serve safe extensions.
ALLOWED_EXTENSIONS = {
    '.html', '.htm', '.js', '.css', '.json', '.bin', '.png', '.svg',
    '.ico', '.webmanifest', '.woff2', '.woff', '.ttf', '.map',
}


def _get_mime(path):
    """Return the MIME type for the given file path/extension."""
    ext = os.path.splitext(path)[1].lower()
    return EXTRA_MIME.get(ext) or mimetypes.guess_type(path)[0] or 'application/octet-stream'


def _resolve_path(url_path):
    """
    Map a URL path to a file on disk.  Protect against path traversal.

    Returns the absolute file path if safe and exists, otherwise None.
    """
    # Strip query string / fragment, URL-decode
    path = unquote(url_path).split('?')[0].split('#')[0]

    # Root URL -> index.html
    if path in ('', '/'):
        return os.path.join(BASE_DIR, INDEX_FILE)

    # Normalize and protect against traversal
    rel = path.lstrip('/')
    rel = os.path.normpath(rel)

    if rel.startswith('..') or os.path.isabs(rel):
        return None

    full = os.path.join(BASE_DIR, rel)

    # If it's a directory, serve index.html inside it
    if os.path.isdir(full):
        full = os.path.join(full, INDEX_FILE)

    # Must exist and be a file with an allowed extension
    if not os.path.isfile(full):
        return None

    ext = os.path.splitext(full)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return None

    return full


def _read_file_safe(full_path):
    """Read a file as bytes, or raise if not readable."""
    with open(full_path, 'rb') as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# WSGI Application (PEP 3333) — works with gunicorn / uWSGI / mod_wsgi
# ---------------------------------------------------------------------------
def application(environ, start_response):
    path = environ.get('PATH_INFO', '/')

    full_path = _resolve_path(path)

    if full_path is None:
        # If no static file matched, fall back to index.html for SPA routes.
        # But only for paths that look like HTML navigation, not missing assets.
        full_path = os.path.join(BASE_DIR, INDEX_FILE)

    try:
        body = _read_file_safe(full_path)
        status = '200 OK'
    except (IOError, OSError):
        body = b'500 Internal Server Error\n'
        status = '500 Internal Server Error'
        full_path = ''
    except Exception:
        body = b'500 Internal Server Error\n'
        status = '500 Internal Server Error'
        full_path = ''

    headers = [
        ('Content-Type', _get_mime(full_path) if full_path else 'text/plain'),
        ('Content-Length', str(len(body))),
        ('Cache-Control', 'no-cache'),
    ]

    # Service worker must be allowed at scope root and never cached
    if full_path.endswith('sw.js'):
        headers.append(('Service-Worker-Allowed', '/'))

    start_response(status, headers)
    return [body]


# ---------------------------------------------------------------------------
# Convenience: direct run with `python main.py`
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Serve EnsetScan Vision')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()

    print(f'Serving EnsetScan at http://{args.host}:{args.port}')
    print('Press Ctrl+C to stop.')


    class QuietHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def _send_file(self, path):
            full = _resolve_path(path)
            if full is None:
                self.send_error(404, 'Not Found')
                return
            try:
                data = _read_file_safe(full)
            except (IOError, OSError):
                self.send_error(500, 'Internal Server Error')
                return
            self.send_response(200)
            self.send_header('Content-Type', _get_mime(full))
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-cache')
            if full.endswith('sw.js'):
                self.send_header('Service-Worker-Allowed', '/')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send_file(self.path)

        def do_HEAD(self):
            self._send_file(self.path)


    server = HTTPServer((args.host, args.port), QuietHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nShutting down…')
        server.server_close()