import json
import socketserver
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import ssl
import argparse
import datetime
from typing import Tuple
from collections import defaultdict

PORT = 13366

class ServerHandler(BaseHTTPRequestHandler):

    save_dir = defaultdict(str)

    def __init__(self, request: bytes, client_address: Tuple[str, int], server: socketserver.BaseServer) -> None:
        super().__init__(request, client_address, server)

    def send_file(self):
        if self.path == "/":
            self.path="./index.html"
        if self.path.startswith('/'):
            self.path = '.' + self.path
        
        try:
            if self.path.endswith(".html"):
                mimetype='text/html'
            elif self.path.endswith(".js"):
                mimetype='application/javascript'
            elif self.path.endswith('json') or self.path.endswith('.js.map'):
                mimetype='application/json'
            elif self.path.endswith('wasm'):
                mimetype='application/wasm'
            elif self.path.endswith(".css"):
                mimetype='text/css'
            elif self.path.endswith(".jpg"):
                mimetype='image/jpg'
            elif self.path.endswith(".gif"):
                mimetype='image/gif'
            else:
                mimetype='application/octet-stream'
            self.send_response(200)
            self.send_header('Content-type', mimetype)
            # self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
            # self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
            self.end_headers()
            if mimetype == 'application/wasm' or mimetype == 'application/octet-stream':
                with open(os.path.join(".", self.path), 'rb') as f:
                    self.wfile.write(f.read())
            else:
                with open(os.path.join(".", self.path), 'r', encoding="utf-8") as f:
                    self.wfile.write(f.read().encode("utf-8"))
        except IOError:
            self.send_error(404, f'File Not Found: {self.path}')

    def do_GET(self):
        # print(f"do_GET debug: get filepath = {self.path}")
        self.send_file()
        

    def do_POST(self):
        print("DEBUG: do_POST at:", self.path)
        self.send_response(200)
        
        def save_data(filename):
            print(self.path, ServerHandler.save_dir[self.client_address[0]], filename)
            if not ServerHandler.save_dir[self.client_address[0]]:
                ServerHandler.save_dir[self.client_address[0]] = os.path.join("data", self.client_address[0], datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d_%H-%M-%S"))
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            os.makedirs(ServerHandler.save_dir[self.client_address[0]], exist_ok=True)
            with open(os.path.join(ServerHandler.save_dir[self.client_address[0]], filename), 'w') as f:
                json.dump(json.loads(post_data), f, indent=2)
            self.end_headers()
            

        if 'data/hardware' in self.path:
            subdir = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
            ServerHandler.save_dir[self.client_address[0]] = os.path.join('data', self.client_address[0], subdir)
            os.makedirs(ServerHandler.save_dir[self.client_address[0]], exist_ok=True)
            save_data("hardware.json")

        elif 'data/tfjs/wasm-kernel' in self.path:
            save_data("tfjs-wasm-profile.json")
        elif 'data/tfjs/webgl-kernel' in self.path:
            save_data("tfjs-webgl-profile.json")
        elif 'data/tfjs/wasm-log' in self.path:
            save_data("tfjs-wasm-log.json")
        elif 'data/tfjs/webgl-log' in self.path:
            save_data("tfjs-webgl-log.json")
        
        elif 'data/ort/wasm-log' in self.path:
            save_data("ort-wasm-profile.json")
        elif 'data/ort/webgl-log' in self.path:
            save_data("ort-webgl-profile.json")
        elif 'data/monitor' in self.path:
            save_data("monitor.json")

        elif 'data/timestamp' in self.path:
            save_data("timestamp.json")

        else:
            self.send_file()
 


parser = argparse.ArgumentParser()
parser.add_argument('--https', action='store_true')
args = parser.parse_args()

cfg = None
if os.path.exists("config/config.json"):
    with open("config/config.json") as f:
        cfg = json.load(f)
if cfg:
    PORT = cfg["PORT"]

try:
    httpd = HTTPServer(("", PORT), ServerHandler)
    if args.https:
        httpd.socket = ssl.wrap_socket(httpd.socket, server_side=True, certfile='./key/cert.pem', keyfile="./key/key.pem", ssl_version=ssl.PROTOCOL_TLSv1_1)
        # httpd.socket = ssl.wrap_socket (httpd.socket, certfile='./key/server.pem', server_side=True)
    print("serving at port", PORT)
    httpd.serve_forever()
except KeyboardInterrupt:
    print('Ctrl+C received, shutting down server')
    httpd.shutdown()


