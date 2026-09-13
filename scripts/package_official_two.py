#!/usr/bin/env python3
"""Package a stopped direct official suite, without weights or old results."""
import argparse
import fcntl
from pathlib import Path
import re
import tarfile


def package(repo,name):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',name):raise ValueError('Use a simple run name')
    root=Path(repo)/'results2'/name
    if not (root/'setup_complete.json').is_file() or root.is_symlink():raise ValueError('Missing official suite')
    with (root/'.official.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise ValueError('Suite is running; package after it stops') from exc
        for p in root.rglob('*'):
            if p.is_symlink() or not (p.is_file() or p.is_dir()) or p.suffix in ('.pt','.bin','.safetensors','.pth'):
                raise ValueError(f'Nonportable result or model weight: {p}')
        output=root.parent/(name+'.tar.gz');tmp=output.with_suffix('.tmp')
        with tarfile.open(tmp,'w:gz') as archive:archive.add(root,arcname=root.name)
        tmp.replace(output)
    print(output)
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run_name',nargs='?',default='official_qwen25_two700')
    package(Path(__file__).resolve().parents[1],p.parse_args().run_name)
