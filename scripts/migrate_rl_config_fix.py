#!/usr/bin/env python3
"""Accept only the dtype/config-copy fix in an existing run, preserving provenance.

Run with training stopped. No checkpoints, data, metrics or run configuration are
modified. The original source manifest and both trainer versions are archived.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tempfile


TRAINER = 'minionerec_trainer.py'
BEFORE = b'        model_init_kwargs = args.model_init_kwargs or {}\n'
AFTER = b'        model_init_kwargs = dict(args.model_init_kwargs or {})\n'
FIX = 'rl_config_copy_v1'


def digest(content):
    return hashlib.sha256(content).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()


def atomic_write(path, content):
    tmp = path.with_name(path.name + '.migration-tmp')
    tmp.write_bytes(content)
    tmp.replace(path)


def migrate(repo, root, dry_run=False):
    repo, root = Path(repo), Path(root)
    files = list((repo / 'reproduction').glob('*.py')) + [repo / name for name in (
        'data.py', TRAINER, 'requirements-reproduction.txt', 'config/zero2_opt.yaml', 'scripts/reproduce.sh')]
    current = {str(path.relative_to(repo)): path.read_bytes() for path in files}
    new_hashes = {name: digest(content) for name, content in current.items()}
    manifest = root / 'source_sha256.json'
    old_manifest = manifest.read_bytes()
    old_hashes = json.loads(old_manifest)
    if old_hashes.keys() != new_hashes.keys():
        raise ValueError('Source file list changed; this migration only accepts the RL config-copy fix')
    changed = {name for name in old_hashes if old_hashes[name] != new_hashes[name]}
    archive = root / 'source_migrations' / FIX
    if not changed:
        for name, expected in new_hashes.items():
            if digest((root / 'source' / name).read_bytes()) != expected:
                raise ValueError(f'Source snapshot changed: {name}')
        print('Source already matches; no migration needed.')
        return
    if changed != {TRAINER}:
        raise ValueError(f'Unrelated source changes refused: {sorted(changed)}')
    patched = current[TRAINER]
    if patched.count(AFTER) != 1:
        raise ValueError('Current trainer does not contain exactly one config-copy fix')
    original = patched.replace(AFTER, BEFORE, 1)
    if digest(original) != old_hashes[TRAINER]:
        raise ValueError('Trainer differs beyond the config-copy fix; migration refused')
    # Validate every original snapshot; allow retry after the snapshot was updated
    # but before the manifest commit, only with the verified archive below.
    for name, expected in old_hashes.items():
        actual = digest((root / 'source' / name).read_bytes())
        if actual != expected and not (
                name == TRAINER and archive.is_dir() and actual == new_hashes[TRAINER]):
            raise ValueError(f'Source snapshot changed: {name}')
    if archive.exists():
        record = json.loads((archive / 'record.json').read_text())
        if (record['before_sha256'] != old_hashes or record['after_sha256'] != new_hashes
                or json.loads((archive / 'source_sha256.before.json').read_bytes()) != old_hashes
                or (archive / 'trainer.before.py').read_bytes() != original
                or (archive / 'trainer.after.py').read_bytes() != patched):
            raise ValueError('Existing migration archive is inconsistent')
    print(f'{root}: verified one-line RL config-copy fix; completed stages will be retained.')
    if dry_run:
        print('Dry run: no files changed.')
        return
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.migration-', dir=archive.parent))
        try:
            (staging / 'source_sha256.before.json').write_bytes(old_manifest)
            (staging / 'trainer.before.py').write_bytes(original)
            (staging / 'trainer.after.py').write_bytes(patched)
            tool_source = Path(__file__).read_bytes()
            (staging / 'migration_tool.py').write_bytes(tool_source)
            (staging / 'record.json').write_bytes(json_bytes({
                'fix': FIX, 'time': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'reason': 'Copy model_init_kwargs before dtype conversion; preserve serializable config',
                'before_sha256': old_hashes, 'after_sha256': new_hashes,
                'migration_tool_sha256': digest(tool_source),
                'preserved_artifacts': 'All data, training/evaluation results, run config and checkpoints',
            }))
            staging.rename(archive)
        except BaseException:
            shutil.rmtree(staging)
            raise
    atomic_write(root / 'source' / TRAINER, patched)
    atomic_write(manifest, json_bytes(new_hashes))
    print(f'Migration complete. Original source retained in {archive}. Continue with --resume.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = repo / 'results' / args.run_name
    if root.parent != repo / 'results' or not root.is_dir() or args.run_name in ('.', '..'):
        raise ValueError('Specify an existing run directory name')
    migrate(repo, root, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
