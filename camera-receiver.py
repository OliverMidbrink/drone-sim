import socket
import struct
import cv2
import numpy as np

HOST = "127.0.0.1"
PORT = 5555


def receive_exact(conn, size):
    data = bytearray()

    while len(data) < size:
        chunk = conn.recv(size - len(data))

        if not chunk:
            raise ConnectionError("Godot disconnected")

        data.extend(chunk)

    return bytes(data)


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind((HOST, PORT))
server.listen(1)

print("Waiting for Godot...")

while True:
    conn, address = server.accept()
    print("Godot connected!")

    try:
        with conn:
            while True:
                # 1 byte camera ID + 4 bytes image length
                header = receive_exact(conn, 5)
                camera_id, length = struct.unpack("<BI", header)

                if camera_id not in (0, 1) or length > 10_000_000:
                    raise ValueError("Invalid camera packet")

                image_bytes = receive_exact(conn, length)

                image = cv2.imdecode(
                    np.frombuffer(image_bytes, dtype=np.uint8),
                    cv2.IMREAD_COLOR
                )

                if image is None:
                    continue

                name = "Left Camera" if camera_id == 0 else "Right Camera"
                cv2.imshow(name, image)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    raise KeyboardInterrupt

    except ConnectionError:
        print("Godot disconnected. Waiting for reconnection...")
    except KeyboardInterrupt:
        break

cv2.destroyAllWindows()
server.close()