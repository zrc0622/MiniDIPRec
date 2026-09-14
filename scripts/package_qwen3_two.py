#!/usr/bin/env python3
"""Package one stopped Qwen3 two-dataset results2 suite, without weights."""
import argparse
import fcntl
from pathlib import Path
import re
import tarfile
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.six_utils import read_json
from scripts.qwen3_two_utils import SNAPSHOTS


def package(repo, name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
        raise ValueError('Specify a simple run name')
    root = Path(repo) / 'results2' / name
    if not root.is_dir() or root.is_symlink():
        raise ValueError('Expected an existing results2 suite')
    with (root / '.five.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Suite is running; package after completion or interruption') from exc
        for path in root.rglob('*'):
            if path.is_symlink() or not (path.is_file() or path.is_dir()):
                raise ValueError(f'Nonportable artifact: {path}')
            if path.suffix in ('.safetensors', '.pt', '.pth', '.bin'):
                raise ValueError(f'Weights must stay outside results2: {path}')
        config = read_json(root / 'suite_config.json')
        if config['snapshot_steps'] != SNAPSHOTS or len(config['variants']) != 2:
            raise ValueError('Expected a Qwen3 two-dataset suite')
        status = read_json(root / 'status.json') if (root / 'status.json').exists() else []
        complete = sum(r['sft_complete'] and r['baseline_complete'] and r['rl_complete']
                       and r['evaluated_steps'] == SNAPSHOTS for r in status)
        output = root.parent / f'{name}.tar.gz'
        temporary = output.with_suffix('.tmp')
        with tarfile.open(temporary, 'w:gz') as archive:
            archive.add(root, arcname=root.name)
        temporary.replace(output)
        print(f'{output} ({complete}/2 experiments fully evaluated)')
        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_name', nargs='?', default='qwen3_1p7b_two350')
    package(Path(__file__).resolve().parents[1], parser.parse_args().run_name)
