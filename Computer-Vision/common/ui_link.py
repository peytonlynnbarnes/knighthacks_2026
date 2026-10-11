"""Preview link between hand_target_node.py (--ui-port) and the app window (ui/app.py).

The app listens on 127.0.0.1:<port> and starts the node with --ui-port <port>; the
node connects (NodeLink) and, instead of cv2.imshow, sends each preview frame:

    !II header (json length, image length) + JSON meta + raw BGR bytes (h x w x 3)

meta: {"w", "h", "lines": [status lines], "arm": [text, alarm] | null,
       "kind": "calib" | "track" | "hold" | "lost", "loading": str (no image)}.
The app sends one-byte commands back: b"c" recalibrate, b"q" quit. A closed
socket counts as "q" (the app went away: the node stops).

Sending never blocks the caller: a sender thread takes only the newest frame (a slow
reader drops frames, it doesn't delay the pipeline).
"""
import json
import socket
import struct
import threading

HEADER = struct.Struct("!II")


def pack(meta, img=None):
    body = b"" if img is None else memoryview(img).cast("B")
    if img is not None:
        meta = dict(meta, h=int(img.shape[0]), w=int(img.shape[1]))
    js = json.dumps(meta).encode()
    return HEADER.pack(len(js), len(body)), js, body


def recv_exact(sock, n):
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("closed")
        got += k
    return buf


def recv_frame(sock):
    """App side: blocks for one message -> (meta, image bytes or None)."""
    jl, il = HEADER.unpack(recv_exact(sock, HEADER.size))
    meta = json.loads(bytes(recv_exact(sock, jl)))
    return meta, (bytes(recv_exact(sock, il)) if il else None)


class NodeLink:
    """Node side: send(img, meta) latest-only; key() -> ord('c') / ord('q') / 255 like cv2.waitKey."""

    def __init__(self, port):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=5.0)
        self.sock.settimeout(None)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.cv = threading.Condition()
        self.pending, self.keys, self.closed = None, [], False
        threading.Thread(target=self._send_loop, name="ui.send", daemon=True).start()
        threading.Thread(target=self._recv_loop, name="ui.recv", daemon=True).start()

    def send(self, meta, img=None):
        with self.cv:
            self.pending = (meta, None if img is None else img.copy())
            self.cv.notify()

    def key(self):
        with self.cv:
            if self.keys:
                return self.keys.pop(0)
            return ord("q") if self.closed else 255

    def _send_loop(self):
        try:
            while True:
                with self.cv:
                    while self.pending is None and not self.closed:
                        self.cv.wait()
                    if self.closed:
                        return
                    msg, self.pending = self.pending, None
                for part in pack(*msg):
                    self.sock.sendall(part)
        except OSError:
            self._close()

    def _recv_loop(self):
        try:
            while True:
                b = self.sock.recv(64)
                if not b:
                    break
                with self.cv:
                    self.keys.extend(b)
        except OSError:
            pass
        self._close()

    def _close(self):
        with self.cv:
            self.closed = True
            self.cv.notify_all()

    def close(self):
        self._close()
        try:
            self.sock.close()
        except OSError:
            pass
