#!/usr/bin/env python3
"""Fresh RL from SFT, bounded updates with the full schedule, then validation only."""
import argparse
import fcntl
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproduction.prepare import CATEGORIES, prepare, sha256, write_json
from reproduction.run import latest_checkpoint
from scripts import evaluate_checkpoints as evaluation
from scripts.start_rl_from_sft import create_run

REPO = Path(__file__).resolve().parents[1]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name', required=True)
    p.add_argument('--source-run', help='Required for a new run; selected SFT only is imported')
    p.add_argument('--dataset', choices=CATEGORIES)
    p.add_argument('--gpus', help='Exactly four GPU IDs; default: source run')
    p.add_argument('--rl-micro-batch', type=int)
    p.add_argument('--learning-rate', type=float, help='Default 1e-5 (official); use 5e-6 for the proposed LR check')
    p.add_argument('--beta', type=float, help='KL coefficient; default 0.001 (official)')
    p.add_argument('--do-sample', action=argparse.BooleanOptionalAction, default=None,
                   help='RL beam candidate sampling; default enabled. Use --no-do-sample for deterministic beams')
    p.add_argument('--stop-after-steps', type=int, help='Optimizer updates, default 350; full scheduler retained')
    p.add_argument('--snapshot-steps', type=int, nargs='+', help='Default 50 100 175, within budget; final step always saved')
    p.add_argument('--resume', action='store_true', help='Use stored settings and resume this short run only')
    p.add_argument('--stage', choices=['all', 'prepare', 'train', 'eval'], default='all')
    return p


def validate_short(config):
    stop, snapshots = config['stop_after_steps'], config['snapshot_steps']
    if stop <= 0 or not snapshots or any(s <= 0 or s > stop for s in snapshots) or stop not in snapshots:
        raise ValueError('Snapshots must be positive, within budget and include the final stop step')
    if len(snapshots) != len(set(snapshots)):
        raise ValueError('Duplicate snapshot steps')
    lr, micro = config['learning_rate'], config['rl_micro_batch']
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError('Learning rate must be positive and finite')
    beta = config.get('beta', 0.001)
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError('Beta must be positive and finite (reference KL remains enabled)')
    if not isinstance(config.get('do_sample', True), bool):
        raise ValueError('do_sample must be a boolean')
    if micro < 16 or micro % 16 or 256 % micro:
        raise ValueError('RL micro batch must be a multiple of G16 dividing 256')
    devices = config['gpus'].split(',')
    if len(devices) != 4 or len(set(devices)) != 4 or any(not re.fullmatch(r'\d+|GPU-[A-Za-z0-9-]+', d) for d in devices):
        raise ValueError('Specify exactly four distinct GPU indices or UUIDs')


def setup_run(repo, args):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_name):
        raise ValueError('Run name must be a simple directory name')
    root = repo / 'results' / args.run_name
    keys = ('source_run', 'dataset', 'gpus', 'rl_micro_batch', 'learning_rate', 'beta', 'do_sample',
            'stop_after_steps', 'snapshot_steps')
    if args.resume:
        config = evaluation.read_json(root / 'run_config.json')
        short = config.get('rl_short')
        if short is None:
            raise ValueError('This is not a short RL run; choose a new run name with --source-run')
        # Old short runs still face their strict source check; defaults are only
        # for interpreting their stored recipe, never a source-check bypass.
        short = {'beta': 0.001, 'do_sample': True, **short}
        for key in keys:
            value = getattr(args, key)
            if key == 'snapshot_steps' and value is not None:
                value = sorted(set(value) | {short['stop_after_steps']})
            if value is not None and value != short[key]:
                raise ValueError(f'Cannot change {key} on resume; use a new run-name')
    else:
        if not args.source_run or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.source_run):
            raise ValueError('A simple --source-run is required to import completed SFT')
        source = evaluation.read_json(repo / 'results' / args.source_run / 'run_config.json')
        if source.get('max_steps', -1) != -1:
            raise ValueError('Import a full-recipe SFT run, not a max-steps smoke test')
        stop = args.stop_after_steps if args.stop_after_steps is not None else 350
        snapshots = args.snapshot_steps if args.snapshot_steps is not None else [s for s in (50, 100, 175) if s <= stop]
        if len(snapshots) != len(set(snapshots)):
            raise ValueError('Duplicate snapshot steps')
        short = {'source_run': args.source_run, 'dataset': args.dataset or CATEGORIES[0],
                 'gpus': args.gpus if args.gpus is not None else source['gpus'],
                 'rl_micro_batch': args.rl_micro_batch if args.rl_micro_batch is not None else 16,
                 'learning_rate': args.learning_rate if args.learning_rate is not None else 1e-5,
                 'beta': args.beta if args.beta is not None else 0.001,
                 'do_sample': args.do_sample if args.do_sample is not None else True,
                 'stop_after_steps': stop, 'snapshot_steps': sorted(set(snapshots) | {stop}),
                 'schedule': 'original 2 epochs, cosine, warmup_ratio=0.03; max_steps=-1',
                 'evaluation': 'valid only; beam50; no final test or new selection rule'}
        validate_short(short)
        create_run(repo, args.source_run, args.run_name, short['dataset'],
                   short['rl_micro_batch'], short_config=short)
        config = evaluation.read_json(root / 'run_config.json')
    validate_short(short)
    return root, config, short


def verify_inputs(repo, root, config, short):
    for manifest, folder in [('source_sha256.json', 'source'), ('short_source_sha256.json', 'short_source')]:
        for name, digest in evaluation.read_json(root / manifest).items():
            if sha256(repo / name) != digest or sha256(root / folder / name) != digest:
                raise ValueError(f'Source changed: {name}; restore matching code or use a new run-name')
    cat = root / short['dataset']
    prepare(repo / 'data/Amazon', cat / 'data', short['dataset'])
    if evaluation.read_json(cat / 'lengths.json')['data_audit_sha256'] != sha256(cat / 'data/audit.json'):
        raise ValueError('Prepared data/lengths disagree')
    imported = evaluation.read_json(root / 'sft_import/record.json')
    model = Path(config['checkpoint_root']) / short['dataset'] / 'sft/selected_model'
    if str(model) != imported['copied_model']:
        raise ValueError('Imported SFT path changed')
    for name, digest in imported['model_sha256'].items():
        if sha256(model / name) != digest:
            raise ValueError(f'Imported SFT weights/tokenizer changed: {name}')
    for name, digest in imported['copied_artifacts_sha256'].items():
        if name == 'lengths.json' or name.startswith(('data/', 'tokenizer/')):
            if sha256(cat / name) != digest:
                raise ValueError(f'Imported artifact changed: {name}')


def training_command(repo, root, config, short, resume=None):
    cat = root / short['dataset']
    weights = Path(config['checkpoint_root']) / short['dataset']
    command = [sys.executable, '-m', 'accelerate.commands.launch', '--config_file',
               str(repo / 'config/zero2_opt.yaml'), '--num_processes', '4', '--main_process_port', '29503',
               '-m', 'scripts.train_rl_short', '--model', str(weights / 'sft/selected_model'),
               '--data', str(cat / 'data'), '--category', short['dataset'], '--output', str(weights / 'rl'),
               '--artifacts', str(cat / 'rl'), '--lengths', str(cat / 'lengths.json'),
               '--micro-batch', str(short['rl_micro_batch']), '--learning-rate', str(short['learning_rate']),
               '--beta', str(short.get('beta', 0.001)),
               '--do-sample' if short.get('do_sample', True) else '--no-do-sample',
               '--stop-after-steps', str(short['stop_after_steps']),
               '--snapshot-steps', *map(str, short['snapshot_steps'])]
    if resume:
        command += ['--resume', str(resume)]
    return command


def finish_training(root, config, short):
    """Recover metadata even if interrupted just after saving the budget checkpoint."""
    artifacts = root / short['dataset'] / 'rl'
    weights = Path(config['checkpoint_root']) / short['dataset'] / 'rl'
    final = weights / f'checkpoint-{short["stop_after_steps"]}'
    for step in short['snapshot_steps']:
        path = weights / f'checkpoint-{step}'
        evaluation.model_files(path, step)
        if not (path / 'reference_model.pt').is_file():
            raise ValueError(f'Incomplete RL checkpoint: {path}')
    state = evaluation.read_json(final / 'trainer_state.json')
    schedule = evaluation.read_json(artifacts / 'schedule.json')
    if state['max_steps'] != schedule['planned_steps'] or state['max_steps'] < short['stop_after_steps']:
        raise ValueError('Saved checkpoint scheduler disagrees with full schedule')
    best = state.get('best_model_checkpoint')
    training = {'parent_model': str(weights.parent / 'sft/selected_model'),
                'final_checkpoint': str(final), 'final_step': state['global_step'],
                'selected_step': int(Path(best).name.split('-')[-1]) if best else state['global_step'],
                'best_checkpoint': best, 'best_metric': state.get('best_metric'),
                'selection': 'Original validation reward selection recorded only; all requested steps evaluated on valid',
                'training_args': evaluation.read_json(artifacts / 'training_args.json'), 'short_run': schedule}
    shutil.copyfile(final / 'trainer_state.json', artifacts / 'trainer_state.json')
    write_json(artifacts / 'training.json', training)
    write_json(artifacts / 'short.complete.json', {'final_step': state['global_step'],
               'planned_steps': state['max_steps'], 'final_checkpoint': str(final)})


def validation_plan(repo, root, config, short):
    # Reevaluate SFT as well: its imported metric metadata retains its source path.
    # Five evaluations cost ~30 min on the measured server and guarantee one setup.
    args = evaluation.parser().parse_args(['--run-name', config['run_name'], '--dataset', short['dataset'],
        '--gpus', short['gpus'], '--steps', *map(str, short['snapshot_steps']),
        '--diagnostic-name', 'short_validation', '--reevaluate-selected'])
    return evaluation.build_plan(repo, args)


def run(repo, args, launcher=evaluation.launch):
    root, config, short = setup_run(repo, args)
    with (root / '.short.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('Another short run process is already active') from exc
        verify_inputs(repo, root, config, short)
        artifacts = root / short['dataset'] / 'rl'
        artifacts.mkdir(exist_ok=True)
        weights = Path(config['checkpoint_root']) / short['dataset'] / 'rl'
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=short['gpus'], TOKENIZERS_PARALLELISM='false',
                   WANDB_MODE='disabled', NCCL_IB_DISABLE='1', HF_HUB_OFFLINE='1',
                   HF_DATASETS_CACHE=str(root / '.cache'))
        env['PYTHONPATH'] = str(repo) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        command = training_command(repo, root, config, short)
        print(shlex.join(command), flush=True)
        if args.stage == 'prepare':
            return
        if args.stage in ('train', 'all') and not (artifacts / 'short.complete.json').exists():
            resume = latest_checkpoint(weights, require_reference=True)
            if resume and int(Path(resume).name.split('-')[-1]) > short['stop_after_steps']:
                raise ValueError('Existing checkpoint exceeds the stop budget')
            final_saved = resume and int(Path(resume).name.split('-')[-1]) == short['stop_after_steps']
            if not final_saved:
                if weights.exists() and any(weights.iterdir()) and not resume:
                    raise ValueError('Nonempty RL directory without a resumable checkpoint')
                check = ('import torch, transformers, deepspeed; '
                         'assert torch.cuda.is_available() and torch.cuda.device_count()==4, "Exactly four CUDA GPUs required"; '
                         'print(torch.__version__, transformers.__version__, deepspeed.__version__)')
                launcher([sys.executable, '-c', check], env, repo, root, root / 'environment.log')
                launcher([sys.executable, '-m', 'pip', 'freeze'], env, repo, root, root / 'environment.log')
                launcher(training_command(repo, root, config, short, resume), env, repo, root, artifacts / 'train.log')
            finish_training(root, config, short)
        if args.stage in ('eval', 'all'):
            if not (artifacts / 'short.complete.json').is_file():
                raise ValueError('Complete the short training stage before evaluation')
            plan = validation_plan(repo, root, config, short)
            evaluation.execute_plan(repo, plan, launcher)
            # The unchanged evaluator's generic wording assumes beam sampling.
            # Record the actual G16 mode for these explicitly varied experiments.
            summary = Path(plan['output']) / 'summary.md'
            summary.write_text(summary.read_text().replace('训练期间 G16 采样奖励',
                f"训练期间 G16 {'采样' if short.get('do_sample', True) else '确定性 beam'}奖励"))
            for extension in ('md', 'csv', 'json'):
                shutil.copyfile(Path(plan['output']) / f'summary.{extension}', root / f'summary.{extension}')
            print(f'Validation only: {root / "summary.md"}', flush=True)


if __name__ == '__main__':
    run(REPO, parser().parse_args())
