"""Run the existing simulator; X11 input only selects and starts its normal score UI."""
import argparse
import json
import hashlib
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'submission/tools'))
import desktop
from run_lesson04_acceptance import focus, wait_window, request_close


def snapshot(xid, path):
    import cv2
    import numpy as np
    Gst = desktop.Gst
    pipe = Gst.parse_launch(f'ximagesrc xid={xid} use-damage=false show-pointer=false ! '
                           'videoconvert ! video/x-raw,format=BGR ! appsink name=sink sync=false')
    pipe.set_state(Gst.State.PLAYING)
    try:
        sample = pipe.get_by_name('sink').emit('try-pull-sample', 3*Gst.SECOND)
        if sample is None:
            raise RuntimeError('X11 screenshot timed out')
        caps = sample.get_caps().get_structure(0)
        w, h = caps.get_value('width'), caps.get_value('height')
        buf = sample.get_buffer()
        ok, mapped = buf.map(Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError('cannot map screenshot')
        try:
            pixels = np.frombuffer(mapped.data, np.uint8).reshape(h, -1)[:, :w*3].reshape(h,w,3)
            if not cv2.imwrite(str(path), pixels):
                raise RuntimeError('cannot save screenshot')
        finally:
            buf.unmap(mapped)
    finally:
        pipe.set_state(Gst.State.NULL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--tracking-config', type=Path, default=ROOT/'configs/lesson05_tracking.yaml')
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('--seconds must be finite and positive')
    tracking_config = args.tracking_config.resolve()
    required = [tracking_config, ROOT/'configs/simulation.yaml',
                ROOT/'simulation/simulator/config.toml',
                ROOT/'simulation/simulator/target/release/daedalus', ROOT/'build-week05/training']
    for path in required:
        if not path.is_file():
            parser.error(f'required file is missing: {path}')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    # 保留有效跟踪参数和输入哈希，防止日后配置变化使结果无法追溯。
    (out/'tracking_config_snapshot.yaml').write_bytes(tracking_config.read_bytes())
    provenance = {
        'tracking_config': str(tracking_config),
        'inputs_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in required},
        'invocation': sys.argv,
    }
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2))
    (out/'config.toml').write_bytes((ROOT/'simulation/simulator/config.toml').read_bytes())
    (out/'simulation.yaml').write_text((ROOT/'configs/simulation.yaml').read_text().replace(
        'allow_fire: false', 'allow_fire: true'))
    env = os.environ.copy()
    env.update(RM_DETECTOR='opencv', RM_TRACKING_CONFIG=str(tracking_config),
               RM_AIM_LOG=str(out/'aim.jsonl'), RM_TIMING_LOG=str(out/'timing.jsonl'),
               TRAINING_HIT_LOG=str(out/'hits.jsonl'))
    children, streams, actions = [], [], []
    simwin = None

    def launch(cmd, log, cwd=ROOT):
        stream = (out/log).open('x')
        streams.append(stream)
        p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT)
        children.append(p)
        return p

    def key(name):
        focus(simwin)
        desktop.key(name)
        actions.append({'key':name, 'monotonic':time.monotonic()})

    try:
        sim = launch([str(ROOT/'simulation/simulator/target/release/daedalus')], 'simulator.log', out)
        simwin = wait_window('daedalus', 30)
        training = launch([str(ROOT/'build-week05/training'), str(out/'simulation.yaml')], 'training.log')
        trainwin = wait_window('RoboMaster training', 30)
        time.sleep(3)
        key('F8')
        for index, name in enumerate(['static_target', 'moving_target']):
            if index:
                key('F6')
                time.sleep(2)
            recorder = None
            stop = out/(name+'.stop')
            if args.record:
                recorder = launch(['/usr/bin/python3', str(ROOT/'submission/tools/record_lesson04_window.py'),
                                   str(simwin), str(out/(name+'.mp4')), str(stop)], name+'_record.log')
            key('F7')
            start = time.monotonic()
            while time.monotonic()-start < args.seconds:
                if sim.poll() is not None or training.poll() is not None:
                    raise RuntimeError('simulator or training process exited')
                if recorder and recorder.poll() is not None:
                    raise RuntimeError('recorder exited before the scene finished')
                time.sleep(1)
            key('F7')
            time.sleep(.3)
            snapshot(simwin, out/(name+'.png'))
            snapshot(trainwin, out/(name+'_detection.png'))
            key('F9')
            deadline = time.monotonic()+5
            while not (out/'training_scores.json').exists():
                if time.monotonic()>deadline:
                    raise RuntimeError('simulator did not export score')
                time.sleep(.05)
            (out/'training_scores.json').rename(out/(name+'_scores.json'))
            if recorder:
                stop.touch()
                recorder.wait(timeout=10)
                if recorder.returncode:
                    raise RuntimeError('recording failed')
            print(name, (out/(name+'_scores.json')).read_text(), flush=True)
        training.send_signal(signal.SIGINT)
        training.wait(timeout=10)
        request_close(simwin)
        sim.wait(timeout=30)
        (out/'run.json').write_text(json.dumps({'actions':actions,
            'simulator_returncode':sim.returncode,'training_returncode':training.returncode,
            'seconds_per_scene':args.seconds,'recorded':args.record,
            'tracking_config':str(tracking_config)}, indent=2))
    finally:
        for p in reversed(children):
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
                try:
                    p.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    p.terminate()
                    p.wait(timeout=8)
        for stream in streams:
            stream.close()


if __name__ == '__main__':
    main()
