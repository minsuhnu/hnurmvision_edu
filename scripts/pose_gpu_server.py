"""Local CUDA inference for the existing Pose26 raw-head decoder. No ROS or training."""
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import struct
import time
import types

import numpy as np
import torch
from ultralytics import YOLO

REQUEST = struct.Struct('!4sI')
RESPONSE = struct.Struct('!4sId')
INPUT_BYTES = 3*640*640*4


def raw_head(self, features):
    heads = self.one2one if self.end2end else self.one2many
    return tuple(torch.cat((heads['box_head'][i](x), heads['cls_head'][i](x).sigmoid(),
                            heads['kpts_head'][i](heads['pose_head'][i](x))), dim=1)
                 for i, x in enumerate(features))


def receive_exact(connection, size):
    data = bytearray(size)
    view = memoryview(data)
    offset = 0
    while offset < size:
        count = connection.recv_into(view[offset:])
        if not count:
            raise EOFError
        offset += count
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--socket', type=Path, required=True)
    parser.add_argument('--ready', type=Path, required=True)
    args = parser.parse_args()
    if args.socket.exists() or args.ready.exists():
        raise FileExistsError('Use a new private runtime directory')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; GPU mode does not silently fall back to CPU')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    model = YOLO(str(args.weights)).model.cpu().float().eval()
    model.fuse()
    head = model.model[-1]
    assert type(head).__name__ == 'Pose26' and head.reg_max == 1 and head.nc == 14 and tuple(head.kpt_shape) == (4, 2)
    head.forward = types.MethodType(raw_head, head)
    model = model.cuda().eval()
    with torch.inference_mode():
        for _ in range(3):
            model(torch.zeros(1, 3, 640, 640, device='cuda'))
    torch.cuda.synchronize()
    args.socket.parent.mkdir(parents=True, exist_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    timings = deque(maxlen=100)
    requests = 0
    def terminate(_signal, _frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        server.bind(str(args.socket))
        os.chmod(args.socket, 0o600)
        server.listen(1)
        args.ready.write_text(json.dumps(dict(device=torch.cuda.get_device_name(0), dtype='float32',
            weights_sha256=hashlib.sha256(args.weights.read_bytes()).hexdigest(), protocol='RMP1/RMR1',
            heads=[[1, 26, s, s] for s in [80, 40, 20]], pid=os.getpid()), indent=2))
        print('CUDA Pose worker ready', args.ready.read_text(), flush=True)
        while True:
            connection, _ = server.accept()
            with connection:
                while True:
                    try:
                        magic, size = REQUEST.unpack(receive_exact(connection, REQUEST.size))
                        if magic != b'RMP1' or size != INPUT_BYTES:
                            raise ValueError('Invalid fixed 640 float32 input')
                        payload = receive_exact(connection, size)
                        started = time.perf_counter()
                        tensor = torch.from_numpy(np.frombuffer(payload, dtype='<f4').reshape(1, 3, 640, 640)).to('cuda')
                        with torch.inference_mode():
                            outputs = model(tensor)
                            result = [x.detach().float().cpu().numpy().astype('<f4', copy=False) for x in outputs]
                        milliseconds = (time.perf_counter()-started)*1000
                        connection.sendall(RESPONSE.pack(b'RMR1', 0, milliseconds))
                        for output in result:
                            connection.sendall(output.tobytes())
                        timings.append(milliseconds)
                        requests += 1
                        if requests % 100 == 0:
                            print(json.dumps(dict(requests=requests, last_ms=milliseconds,
                                                  median_ms=float(np.median(timings)))), flush=True)
                    except (EOFError, BrokenPipeError, ConnectionResetError):
                        break
                    except Exception as exc:
                        message = str(exc).encode()[:4096]
                        connection.sendall(RESPONSE.pack(b'RMR1', 1, 0.0)+struct.pack('!I', len(message))+message)
                        print('Inference error:', repr(exc), flush=True)
                        break
    finally:
        server.close()
        args.socket.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
