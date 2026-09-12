#!/usr/bin/env python3
"""Package only one stopped results2 suite; exclude model/optimizer state."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import tarfile


def package(repo, run_name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run_name):
        raise ValueError('Specify a simple run directory name')
    root = Path(repo) / 'results2' / run_name
    if not root.is_dir() or root.is_symlink():
        raise ValueError('Specify an existing results2 suite')
    with (root / '.five.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Suite is running; package after it completes or stops to avoid partial logs') from exc
        for path in root.rglob('*'):
            if path.is_symlink() or not (path.is_file() or path.is_dir()):
                raise ValueError(f'Nonportable artifact: {path}')
            if path.suffix in ('.safetensors', '.pt', '.pth', '.bin'):
                raise ValueError(f'Weights must stay outside results2: {path}')
        # Incomplete stopped suites remain packageable and clearly labelled.
        status = json.loads((root / 'status.json').read_text()) if (root / 'status.json').exists() else []
        complete = sum(r['rl_complete'] and len(r['evaluated_steps']) == 4 for r in status)
        output = root.parent / f'{run_name}.tar.gz'
        temporary = output.with_suffix('.tmp')
        with tarfile.open(temporary, 'w:gz') as archive:
            archive.add(root, arcname=root.name)
        temporary.replace(output)
        print(f'{output} ({complete}/5 experiments fully evaluated)')
        return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_name', nargs='?', default='minionerec_five350')
    package(Path(__file__).resolve().parents[1], p.parse_args().run_name)
