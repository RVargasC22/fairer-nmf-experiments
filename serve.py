import http.server
import socketserver
import os

PORT = 8000

class MyHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Disable caching for easier development
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

def run_server():
    print(f"Lanzando servidor en http://localhost:{PORT}")
    print("Para ver la presentación, abre: http://localhost:8000/index.html")
    print("Presiona Ctrl+C para detener.")
    
    with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServidor detenido por el usuario.")
            httpd.shutdown()

if __name__ == "__main__":
    run_server()
