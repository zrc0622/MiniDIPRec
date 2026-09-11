#!/usr/bin/env python3
"""Serial A/B/C short RL from one selected SFT; full schedule and validation only."""
import argparse
import csv
import fcntl
import os
from pathlib import Path
import re
import shlex
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproduction.prepare import CATEGORIES, sha256, write_json
from scripts import run_rl_short as short

REPO = Path(__file__).resolve().parents[1]
VARIANTS = [('A', 0.04, True), ('B', 0.001, False), ('C', 0.04, False)]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name', required=True, help='Suite name; children are <name>_A, _B, _C')
    p.add_argument('--source-run', default='qwen3_h50_seed42_rl16_fixed')
    p.add_argument('--dataset', choices=CATEGORIES, default=CATEGORIES[0])
    p.add_argument('--gpus', default='0,1,2,3')
    p.add_argument('--rl-micro-batch', type=int, default=16)
    p.add_argument('--stop-after-steps', type=int, default=175,
                   help='Updates per experiment, default175; original full scheduler retained')
    p.add_argument('--snapshot-steps', type=int, nargs='+',
                   help='Default50/100 within budget plus final step')
    p.add_argument('--stage', choices=['all', 'prepare', 'train', 'eval'], default='all')
    return p


def settings(args):
    for name in (args.run_name, args.source_run):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
            raise ValueError('Use simple run names')
    steps = args.snapshot_steps if args.snapshot_steps is not None else [s for s in (50, 100) if s <= args.stop_after_steps]
    if len(steps) != len(set(steps)):
        raise ValueError('Duplicate snapshot steps')
    config = {'run_name': args.run_name, 'source_run': args.source_run, 'dataset': args.dataset,
              'gpus': args.gpus, 'rl_micro_batch': args.rl_micro_batch, 'learning_rate': 1e-5,
              'stop_after_steps': args.stop_after_steps,
              'snapshot_steps': sorted(set(steps) | {args.stop_after_steps}),
              'variants': [{'label': label, 'beta': beta, 'do_sample': sample,
                            'run_name': f'{args.run_name}_{label}'} for label, beta, sample in VARIANTS],
              'evaluation': 'valid only; deterministic beam50; no automatic test or model selection',
              'schedule': 'original full 2 epochs; early stop, not max_steps override'}
    short.validate_short(config)
    if args.source_run == args.run_name or args.source_run in [v['run_name'] for v in config['variants']]:
        raise ValueError('Source, suite and experiment run names must be distinct')
    return config


def child_command(repo, config, variant, stage, resume):
    command = [sys.executable, str(repo / 'scripts/run_rl_short.py'),
               '--run-name', variant['run_name'], '--source-run', config['source_run'],
               '--dataset', config['dataset'], '--gpus', config['gpus'],
               '--rl-micro-batch', str(config['rl_micro_batch']), '--learning-rate', '1e-5',
               '--beta', str(variant['beta']), '--do-sample' if variant['do_sample'] else '--no-do-sample',
               '--stop-after-steps', str(config['stop_after_steps']), '--snapshot-steps',
               *map(str, config['snapshot_steps']), '--stage', stage]
    if resume:
        command.append('--resume')
    return command


def check_shared_sft(repo, root, config):
    """Check all three imported model/data copies before spending GPU time."""
    identity = None
    for variant in config['variants']:
        child = repo / 'results' / variant['run_name']
        record = short.evaluation.read_json(child / 'sft_import/record.json')
        current = {'selected_step': record['selected_step'], 'model_sha256': record['model_sha256'],
                   'data_sha256': sha256(child / config['dataset'] / 'data/audit.json'),
                   'lengths_sha256': sha256(child / config['dataset'] / 'lengths.json')}
        if identity is not None and identity != current:
            raise ValueError('A/B/C do not share identical selected SFT/data/lengths')
        identity = current
    saved = root / 'shared_sft.json'
    if saved.exists() and short.evaluation.read_json(saved) != identity:
        raise ValueError('Suite SFT identity changed')
    write_json(saved, identity)


def collect(repo, root, config, variant):
    """Portable ordinary-file snapshot, including partial logs on failure; no weights."""
    source = repo / 'results' / variant['run_name']
    if not source.exists():
        return
    destination = root / config['dataset'] / variant['label']
    copied = {}
    for path in sorted(source.rglob('*')):
        relative = path.relative_to(source)
        if '.cache' in relative.parts or path.name.endswith('.lock'):
            continue
        if path.is_symlink() or path.suffix in ('.pt', '.bin', '.safetensors'):
            raise ValueError(f'Results must be regular artifacts without weights: {path}')
        if path.is_file():
            digest = sha256(path)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or sha256(target) != digest:
                shutil.copyfile(path, target)
            copied[str(relative)] = digest
    write_json(destination / 'suite_copy.json', {'source_run': variant['run_name'], 'sha256': copied,
               'purpose': 'Portable analysis copy; resume using the suite resume.sh at its top level'})


def summarize(root, config):
    rows = []
    for variant in config['variants']:
        report = root / config['dataset'] / variant['label'] / 'summary.json'
        if not report.is_file():
            continue
        for row in short.evaluation.read_json(report):
            rows.append({'experiment': variant['label'], 'beta': variant['beta'],
                         'do_sample': variant['do_sample'], 'learning_rate': config['learning_rate'], **row})
    write_json(root / 'summary.json', rows)
    columns = ['experiment', 'beta', 'do_sample', 'checkpoint', 'rl_step',
               'HR@5', 'HR@10', 'NDCG@5', 'NDCG@10', 'delta_HR@10', 'delta_NDCG@10']
    with (root / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = ['A/B/C 验证集对比（HR=Recall；delta为相对本组同一SFT的绝对差值）。', '',
             '三组LR=1e-5、G16、有效候选batch1024、完整2epoch调度；仅提前停止。',
             'beta/do_sample作用于训练及原G16验证奖励；排名评估始终为确定性beam50。',
             '不进行测试集评价，不自动选模型；短跑只用于早期筛查。', '',
             '| ' + ' | '.join(columns) + ' |', '|' + '|'.join(['---'] * len(columns)) + '|']
    for row in rows:
        lines.append('| ' + ' | '.join(f'{row[k]:.6f}' if isinstance(row[k], float) else str(row[k]) for k in columns) + ' |')
    lines += ['', f'已汇总 {len(rows)} 项；每组预期 SFT + {len(config["snapshot_steps"])} 个RL快照。',
              f'完整日志/预测/配置实体副本位于 {config["dataset"]}/A、B、C；checkpoint路径见各组记录。']
    (root / 'summary.md').write_text('\n'.join(lines) + '\n')


def run(repo, args, launcher=short.evaluation.launch):
    config = settings(args)
    root = repo / 'results' / config['run_name']
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.abc.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('This A/B/C suite is already running') from exc
        path = root / 'abc_config.json'
        hashes = {name: sha256(repo / name) for name in ('scripts/run_rl_abc.py', 'scripts/package_results.py')}
        if path.exists():
            if short.evaluation.read_json(path) != config:
                raise ValueError('Suite settings changed; use the saved resume.sh or a new run-name')
            if short.evaluation.read_json(root / 'abc_source_sha256.json') != hashes:
                raise ValueError('Suite source changed; restore matching code or use a new run-name')
            for name, digest in hashes.items():
                if sha256(root / 'source' / name) != digest:
                    raise ValueError(f'Suite source snapshot changed: {name}')
        else:
            if any(p.name != '.abc.lock' for p in root.iterdir()):
                raise ValueError('Nonempty suite directory without configuration')
            for variant in config['variants']:
                if (repo / 'results' / variant['run_name']).exists():
                    raise ValueError(f'Experiment already exists: {variant["run_name"]}; choose a new suite name')
            for name in hashes:
                target = root / 'source' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(repo / name, target)
            write_json(root / 'abc_source_sha256.json', hashes)
            write_json(path, config)
        resume = [sys.executable, str(repo / 'scripts/run_rl_abc.py'), '--run-name', config['run_name'],
                  '--source-run', config['source_run'], '--dataset', config['dataset'], '--gpus', config['gpus'],
                  '--rl-micro-batch', str(config['rl_micro_batch']), '--stop-after-steps', str(config['stop_after_steps']),
                  '--snapshot-steps', *map(str, config['snapshot_steps'])]
        (root / 'resume.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd ' + shlex.quote(str(repo)) + '\n' + shlex.join(resume) + '\n')
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=config['gpus'], TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1')
        # Prepare all copies first; each worker also checks the pinned source/data.
        for variant in config['variants']:
            exists = (repo / 'results' / variant['run_name'] / 'run_config.json').is_file()
            if args.stage == 'eval' and not exists:
                raise ValueError('Prepare/train the suite before using --stage eval')
            launcher(child_command(repo, config, variant, 'prepare', exists), env, repo, root,
                     root / f'{variant["label"]}.log')
        check_shared_sft(repo, root, config)
        for variant in config['variants']:
            print(f'=== Experiment {variant["label"]}: beta={variant["beta"]}, do_sample={variant["do_sample"]} ===', flush=True)
            try:
                if args.stage != 'prepare':
                    launcher(child_command(repo, config, variant, args.stage, True), env, repo, root,
                             root / f'{variant["label"]}.log')
            finally:
                collect(repo, root, config, variant)
                summarize(root, config)
        print(f'A/B/C {args.stage} finished: {root / "summary.md"}', flush=True)


if __name__ == '__main__':
    run(REPO, parser().parse_args())
