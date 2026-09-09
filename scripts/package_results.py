#!/usr/bin/env python3
"""Pack a portable run and reject symlinks or accidental checkpoint inclusion."""
import argparse
from pathlib import Path
import tarfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('run_name')
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = repo / 'results' / args.run_name
    if root.parent != repo / 'results' or not root.is_dir():
        raise ValueError('Specify an existing run directory name')
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError(f'Portable results must not contain symlinks: {path}')
        if path.suffix in ('.safetensors', '.pt', '.bin'):
            raise ValueError(f'Move model/checkpoint weights outside results before packaging: {path}')
    output = root.parent / f'{args.run_name}.tar.gz'
    with tarfile.open(output, 'w:gz') as archive:
        archive.add(root, arcname=root.name)
    print(output)


if __name__ == '__main__':
    main()
