#!/usr/bin/env python3
"""Change RL micro batch with effective batch 1024; run only with training stopped.

Archive run/checkpoint JSON metadata before changing it. Model, optimizer,
scheduler, reference and RNG state files remain untouched. The saved Trainer batch
must also change: Transformers restores it over the command-line micro batch.
"""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.migrate_rl_config_fix import atomic_write, digest, json_bytes, migrate as migrate_source


def migrate_batch(repo, root, micro_batch, dry_run=False):
    repo, root = Path(repo), Path(root)
    if micro_batch < 16 or micro_batch % 16 or 256 % micro_batch:
        raise ValueError('RL micro batch must be a multiple of 16 dividing 256')
    config_path = root / 'run_config.json'
    before = config_path.read_bytes()
    config = json.loads(before)
    old_micro = config['rl_micro_batch']
    if (config['world_size'] != 4 or config['effective_batch'] != 1024
            or config['rl_generations'] != 16 or old_micro < 16
            or old_micro % 16 or 256 % old_micro
            or config['rl_accumulation'] != 256 // old_micro):
        raise ValueError('Expected four-GPU G16 recipe with effective batch 1024')
    archive_parent = root / 'config_migrations'
    # Finish an interrupted migration first; the run config is committed last.
    pending = []
    if archive_parent.exists():
        for path in archive_parent.glob('rl_batch_*'):
            if path.is_dir() and not (path / 'complete.json').exists():
                pending.append(path)
    if len(pending) > 1:
        raise ValueError('Multiple pending migrations; inspect config_migrations')
    if pending:
        archive = pending[0]
        record = json.loads((archive / 'record.json').read_text())
        if record['after']['rl_micro_batch'] != micro_batch:
            raise ValueError('Finish the pending migration with its original micro batch first')
        if config not in (record['before'], record['after']):
            raise ValueError('Run configuration changed during pending migration')
        changes = []
        for i, entry in enumerate(record['changes']):
            path = Path(entry['path'])
            original = (archive / f'{i}.before.json').read_bytes()
            updated = (archive / f'{i}.after.json').read_bytes()
            if digest(original) != entry['before_sha256'] or digest(updated) != entry['after_sha256']:
                raise ValueError('Migration backup hash mismatch')
            if path.read_bytes() not in (original, updated):
                raise ValueError(f'Metadata changed since migration started: {path}')
            changes.append((path, original, updated))
    else:
        if old_micro == micro_batch:
            print(f'RL micro batch is already {micro_batch}; accumulation {256 // micro_batch}.')
            return
        checkpoint_root = Path(config['checkpoint_root'])
        if not checkpoint_root.is_dir():
            raise ValueError(f'Checkpoint root unavailable: {checkpoint_root}; run on the training server')
        updated_config = dict(config, rl_micro_batch=micro_batch, rl_accumulation=256 // micro_batch)
        changes, checkpoints = [], []
        for category in ('Office_Products', 'Industrial_and_Scientific'):
            artifact = root / category / 'rl'
            if (artifact / 'rl.complete.json').exists():
                continue
            output = checkpoint_root / category / 'rl'
            for checkpoint in sorted(output.glob('checkpoint-*')):
                state_path = checkpoint / 'trainer_state.json'
                if not state_path.is_file():
                    continue
                state_bytes = state_path.read_bytes()
                state = json.loads(state_bytes)
                if state.get('train_batch_size') != old_micro:
                    raise ValueError(f'Unexpected saved train_batch_size: {state_path}')
                new_state = dict(state, train_batch_size=micro_batch)
                changes.append((state_path, state_bytes, json_bytes(new_state)))
                checkpoints.append({'category': category, 'path': str(checkpoint),
                    'step': state['global_step'], 'epoch': state.get('epoch'),
                    'has_reference': (checkpoint / 'reference_model.pt').is_file()})
            # The next training process will overwrite training_args.json, so
            # preserve this pre-migration record alongside the run/checkpoint JSON.
            training_args = artifact / 'training_args.json'
            if training_args.is_file():
                content = training_args.read_bytes()
                recorded = json.loads(content)
                if (recorded['per_device_train_batch_size'] != old_micro
                        or recorded['gradient_accumulation_steps'] != config['rl_accumulation']):
                    raise ValueError(f'RL training arguments disagree with run config: {training_args}')
                # Keep this file unchanged until a new training process records its arguments.
                changes.append((training_args, content, content))
        changes.append((config_path, before, json_bytes(updated_config)))
        record = {'time': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'reason': 'User-requested throughput adaptation; effective candidate batch remains 1024',
            'before': config, 'after': updated_config, 'checkpoints': checkpoints,
            'changes': [{'path': str(path.resolve()), 'before_sha256': digest(original),
                         'after_sha256': digest(updated)} for path, original, updated in changes],
            'resume_note': 'Resume latest complete checkpoint; unsaved updates are replayed. '
                           'Micro grouping, padding and stochastic generation can change; not bitwise identical.'}
        archive = None
    # Validate code too; also accept/record the earlier one-line dtype fix if needed.
    migrate_source(repo, root, dry_run=True)
    old = record['before']['rl_micro_batch']
    print(f'RL micro {old} -> {micro_batch}; accumulation {256 // old} -> {256 // micro_batch}; '
          'effective batch = 4 * micro * accumulation = 1024.')
    for checkpoint in record['checkpoints']:
        print(f'  {checkpoint["path"]}: step {checkpoint["step"]}, reference={checkpoint["has_reference"]}')
    if dry_run:
        print('Dry run: no files changed.')
        return
    migrate_source(repo, root)
    if archive is None:
        archive_parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix='.rl-batch-', dir=archive_parent))
        try:
            for i, (_, original, updated) in enumerate(changes):
                (staging / f'{i}.before.json').write_bytes(original)
                (staging / f'{i}.after.json').write_bytes(updated)
            (staging / 'record.json').write_bytes(json_bytes(record))
            (staging / 'migration_tool.py').write_bytes(Path(__file__).read_bytes())
            (staging / 'migrate_rl_config_fix.py').write_bytes(
                Path(__file__).with_name('migrate_rl_config_fix.py').read_bytes())
            archive = archive_parent / f'rl_batch_{old}_to_{micro_batch}_{staging.name.removeprefix(".rl-batch-")}'
            staging.rename(archive)
        except BaseException:
            shutil.rmtree(staging)
            raise
    for path, original, updated in changes:
        if path.read_bytes() == original and original != updated:
            atomic_write(path, updated)
    atomic_write(archive / 'complete.json', json_bytes({'completed': True}))
    print(f'Migration complete; metadata backups: {archive}')
    print(f'Continue the original command with --rl-micro-batch {micro_batch} --resume.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--rl-micro-batch', type=int, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = repo / 'results' / args.run_name
    if root.parent != repo / 'results' or not root.is_dir() or args.run_name in ('.', '..'):
        raise ValueError('Specify an existing run name')
    migrate_batch(repo, root, args.rl_micro_batch, args.dry_run)


if __name__ == '__main__':
    main()
