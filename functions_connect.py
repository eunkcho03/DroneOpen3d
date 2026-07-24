import json
import socket
import struct
import numpy as np

def recv_exact(sock, n_bytes):
    data = b""

    while len(data) < n_bytes:
        packet = sock.recv(n_bytes - len(data))

        if not packet:
            return None

        data += packet

    return data


def create_server(host, port):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(1)

    print("Waiting for Unity...")
    conn, addr = server.accept()
    print("Connected to Unity:", addr)

    return server, conn


def receive_header(conn):
    header = recv_exact(conn, 52)

    if header is None:
        return None

    return struct.unpack("<iifffffffffff", header)


def receive_depth(conn, width, height):
    n_bytes = width * height * 4
    depth_bytes = recv_exact(conn, n_bytes)

    if depth_bytes is None:
        return None

    depth = np.frombuffer(depth_bytes, dtype=np.float32)
    depth = depth.reshape((height, width))

    return depth

def connect_to_unity_nbv(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))
    print("Connected to Unity NBV receiver")
    return sock

def send_next_view(nbv_sock, best_view):
    q = best_view["quat"]
    pos = best_view["pos"]

    msg = {
        "type": "next_view",
        "view_id": int(best_view["view_id"]),
        "position": {
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2]),
        },
        "rotation": {
            "x": float(q[0]),
            "y": float(q[1]),
            "z": float(q[2]),
            "w": float(q[3]),
        },
    }

    payload = json.dumps(msg) + "\n"
    nbv_sock.sendall(payload.encode("utf-8"))

    print("Sent NBV to Unity:", msg)