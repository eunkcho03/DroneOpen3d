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

def send_next_view(nbv_sock, best_view, best_score):
    msg = {
        "type": "next_view",
        "view_id": int(best_view["view_id"]),
        "angle_deg": float(best_view["angle_deg"]),
        "position": {
            "x": float(best_view["position"][0]),
            "y": float(best_view["position"][1]),
            "z": float(best_view["position"][2]),
        },
        "look_at": {
            "x": float(best_view["look_at"][0]),
            "y": float(best_view["look_at"][1]),
            "z": float(best_view["look_at"][2]),
        },
        "score": float(best_score),
    }

    # Unity NBVReceiver expects newline-ended JSON text.
    payload = json.dumps(msg) + "\n"

    nbv_sock.sendall(payload.encode("utf-8"))

    print("Sent NBV to Unity:", msg)