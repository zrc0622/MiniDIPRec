#!/usr/bin/env python3
"""Serial Qwen3-1.7B/current MiniOneRec Office+Industrial: fresh SFT then RL350."""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproduction.prepare import CATEGORIES, sha256, write_json
from scripts.five_utils import prepare_official, resumable_checkpoint
from scripts.run_rl_five import evaluate, source_files as old_sources
from scripts.run_rl_six import recover_sft_export
from scripts.six_utils import file_manifest, launch, pin_json, read_json, verify_files
from scripts.qwen3_two_utils import DEFAULT_MODEL, SNAPSHOTS, finish_rl, variants

REPO = Path(__file__).resolve().parents[1]


def source_files(repo):
    return sorted(set(old_sources(repo) + ['scripts/run_qwen3_two.py',
        'scripts/train_qwen3_two.py', 'scripts/qwen3_two_utils.py', 'scripts/package_qwen3_two.py']))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name', default='qwen3_1p7b_two350')
    p.add_argument('--checkpoint-root', default='checkpoints', help='Weight parent; outside results/results2')
    p.add_argument('--model', default=DEFAULT_MODEL, help='Qwen/Qwen3-1.7B or its unmodified local copy')
    p.add_argument('--gpus', default='0,1,2,3')
    p.add_argument('--sft-micro-batch', type=int, default=2)
    p.add_argument('--eval-batch-size', type=int, default=1)
    p.add_argument('--offline', action='store_true')
    p.add_argument('--stage', choices=['all', 'prepare', 'train', 'eval'], default='all')
    p.add_argument('--experiments', type=int, nargs='+', choices=[1, 2], default=[1, 2])
    p.add_argument('--dry-run', action='store_true', help='Print plan without writes/model downloads')
    return p


def settings(repo, args):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_name):
        raise ValueError('Use a simple run name')
    devices = args.gpus.split(',')
    if len(devices) != 4 or len(set(devices)) != 4 or any(not re.fullmatch(r'\d+|GPU-[A-Za-z0-9-]+', d) for d in devices):
        raise ValueError('Exactly four distinct GPU IDs required')
    if args.sft_micro_batch <= 0 or 256 % args.sft_micro_batch or args.eval_batch_size <= 0:
        raise ValueError('SFT micro batch must divide256; evaluation batch must be positive')
    model = Path(args.model).expanduser()
    if args.model != DEFAULT_MODEL and not model.is_dir():
        raise ValueError('Use Qwen/Qwen3-1.7B or an existing local copy')
    if model.is_dir():
        model_name = str(model.resolve())
        if model.name in ('selected_model', 'final_checkpoint') or model.name.startswith('checkpoint-'):
            raise ValueError('Use the original pretrained model, not a recommendation checkpoint')
    else:
        model_name = DEFAULT_MODEL
    weights = Path(args.checkpoint_root).expanduser()
    weights = ((weights if weights.is_absolute() else repo / weights) / args.run_name).resolve()
    for results in [(repo / 'results').resolve(), (repo / 'results2').resolve()]:
        if weights == results or results in weights.parents or weights in results.parents:
            raise ValueError('Checkpoint root must be separate from results/results2')
    root = repo / 'results2' / args.run_name
    config = dict(run_name=args.run_name, checkpoint_root=str(weights), model=model_name,
        gpus=args.gpus, sft_micro_batch=args.sft_micro_batch, eval_batch_size=args.eval_batch_size,
        offline=args.offline, variants=variants(), stop_after_steps=350, snapshot_steps=SNAPSHOTS,
        seed=42, evaluation='valid only; deterministic beam50; SFT-aligned shared prompt; own SFT baseline',
        artifacts='results2; shared data/source; gzip predictions; 256KiB console +64KiB warning tails')
    if (root / 'suite_config.json').exists() and read_json(root / 'suite_config.json') != config:
        raise ValueError('Suite settings changed; use saved resume.sh or a new run name')
    return root, config


def setup(repo, root, config):
    """Atomically materialize data/source only; no dependency on historical runs."""
    shared = root / 'shared'
    hashes = {name: sha256(repo / name) for name in source_files(repo)}
    weights = Path(config['checkpoint_root'])
    if not (root / 'suite_config.json').exists():
        if any(p.name != '.five.lock' for p in root.iterdir()) or weights.exists():
            raise ValueError('Nonempty destination without suite identity; use a new run name')
        pin_json(root / 'suite_config.json', config)
    if not shared.exists():
        with tempfile.TemporaryDirectory(prefix='.qwen3-two-', dir=root.parent) as tmp:
            stage = Path(tmp) / 'shared'
            for category in CATEGORIES:
                prepare_official(repo / 'data/Amazon', stage / category / 'data', category)
            for name in hashes:
                target = stage / 'source' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(repo / name, target)
            write_json(stage / 'manifest.json', dict(artifacts=file_manifest(stage), source_sha256=hashes))
            stage.rename(shared)
    manifest = read_json(shared / 'manifest.json')
    verify_files(shared, manifest['artifacts'])
    verify_files(repo, manifest['source_sha256'])
    if hashes != manifest['source_sha256']:
        raise ValueError('Suite source inventory changed')
    if weights.exists() and not (weights / 'suite_identity.json').exists() and any(weights.iterdir()):
        raise ValueError('Nonempty weights directory without matching suite identity')
    pin_json(weights / 'suite_identity.json', config)


def environment(repo, config):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=config['gpus'], TOKENIZERS_PARALLELISM='false', WANDB_MODE='disabled',
               NCCL_IB_DISABLE='1', HF_HUB_OFFLINE='1' if config['offline'] else '0',
               HF_DATASETS_OFFLINE='1' if config['offline'] else '0', HF_HUB_DISABLE_PROGRESS_BARS='1')
    env['PYTHONPATH'] = str(repo) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    return env


def worker_command(repo, spec, stage, resume=None):
    if stage == 'rl':
        command = [sys.executable, '-m', 'accelerate.commands.launch', '--config_file',
            str(repo / 'config/zero2_opt.yaml'), '--num_processes', '4', '--main_process_port', '29503']
    elif stage == 'sft':
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4']
    else:
        command = [sys.executable]
    command += ['-m', 'scripts.train_qwen3_two', '--spec', str(spec), '--stage', stage]
    if resume:
        command += ['--resume', str(resume)]
    return command


def baseline_path(root, variant):
    return root / 'baselines' / variant['label']


def summarize(root, config):
    rows, status = [], []
    for variant in config['variants']:
        exp = root / 'experiments' / variant['label']
        baseline = baseline_path(root, variant)
        points = [step for step in SNAPSHOTS if (exp / f'validation/checkpoint-{step}/complete.json').exists()]
        status.append(dict(experiment=variant['label'],
                           sft_complete=(exp / 'sft/complete.json').exists(),
                           baseline_complete=(baseline / 'complete.json').exists(),
                           rl_complete=(exp / 'rl/complete.json').exists(), evaluated_steps=points))
        if not (baseline / 'complete.json').exists():
            continue
        base = read_json(baseline / 'valid.metrics.json')
        for step in points:
            score = read_json(exp / f'validation/checkpoint-{step}/valid.metrics.json')
            row = dict(experiment=variant['label'], dataset=variant['category'], model=config['model'],
                step=step, prompt_groups_seen=step * 64, sft_HR10=base['HR@10'], sft_NDCG10=base['NDCG@10'])
            row.update({key: score[key] for key in ('samples', 'HR@5', 'HR@10', 'NDCG@5', 'NDCG@10', 'HR@50', 'NDCG@50')})
            row.update(delta_HR10=score['HR@10']-base['HR@10'], delta_NDCG10=score['NDCG@10']-base['NDCG@10'],
                       delta_HR50=score['HR@50']-base['HR@50'])
            rows.append(row)
    write_json(root / 'summary.json', rows)
    write_json(root / 'status.json', status)
    if rows:
        with (root / 'summary.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    columns = ['experiment', 'step', 'HR@10', 'NDCG@10', 'delta_HR10', 'delta_NDCG10', 'HR@50']
    lines = ['# Qwen3-1.7B 双数据集 RL350 验证结果', '',
             '当前 MiniOneRec H10 配方；每组与自己的 SFT 比较；保留两轮学习率计划，仅训练前350步。',
             'Office / Industrial 各自重新 SFT；不继承旧模型或上一组RL；统一SFT/RL/评测提示词。', '',
             '| ' + ' | '.join(columns) + ' |', '|' + '|'.join(['---'] * len(columns)) + '|']
    lines += ['| ' + ' | '.join(f'{r[k]:.6f}' if isinstance(r[k], float) else str(r[k]) for k in columns) + ' |' for r in rows]
    if not rows:
        lines += ['', '尚无完成的 RL 验证结果。状态见 status.json。']
    (root / 'summary.md').write_text('\n'.join(lines) + '\n')


def run(repo, args, launcher=launch):
    repo = Path(repo).resolve()
    root, config = settings(repo, args)
    selected = [config['variants'][i-1] for i in sorted(set(args.experiments))]
    if args.dry_run:
        print(json.dumps(dict(results=str(root), checkpoints=config['checkpoint_root'], variants=selected,
            sft='each dataset independently starts from '+config['model'], rl_updates=350,
            snapshots=SNAPSHOTS, gpus=config['gpus'], evaluations=6 * len(selected), groups_per_run=22400), indent=2))
        return
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.five.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('This Qwen3 two-dataset suite is already running') from exc
        setup(repo, root, config)
        command = [sys.executable, str(repo / 'scripts/run_qwen3_two.py'), '--run-name', config['run_name'],
            '--checkpoint-root', str(Path(config['checkpoint_root']).parent), '--model', config['model'],
            '--gpus', config['gpus'], '--sft-micro-batch', str(config['sft_micro_batch']),
            '--eval-batch-size', str(config['eval_batch_size'])]
        if config['offline']:
            command += ['--offline']
        (root / 'resume.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd ' + shlex.quote(str(repo)) + '\n' + shlex.join(command) + '\n')
        summarize(root, config)
        if args.stage == 'prepare':
            return
        env = environment(repo, config)
        weights = Path(config['checkpoint_root'])
        specs = []
        # Scan all selected model/data combinations before spending time on training.
        for variant in selected:
            exp = root / 'experiments' / variant['label']; exp.mkdir(parents=True, exist_ok=True)
            data = root / 'shared' / variant['category'] / 'data'
            model = config['model']
            lengths = exp / 'lengths.json'
            spec = dict(model=model, data=str(data), category=variant['category'], variant=variant,
                lengths=str(lengths), artifacts=str(exp / 'sft'), output=str(weights / variant['label'] / 'sft'),
                sft_micro_batch=config['sft_micro_batch'])
            spec_path = exp / 'sft_spec.json'; pin_json(spec_path, spec)
            if not lengths.exists():
                if args.stage == 'eval':
                    raise ValueError('Run preflight/training before --stage eval')
                launcher(worker_command(repo, spec_path, 'preflight'), env, repo, root, exp / 'preflight.tail.log')
            pin_json(lengths.parent / 'lengths_sha256.json', {'sha256': sha256(lengths)})
            specs.append((variant, spec, spec_path))
        if not (root / 'environment.json').exists():
            probe = ('import json,torch,transformers,trl,accelerate,deepspeed,bitsandbytes; '
                     'from pathlib import Path; '
                     'assert torch.cuda.is_available() and torch.cuda.device_count()==4; '
                     'd={x.__name__:x.__version__ for x in (torch,transformers,trl,accelerate,deepspeed,bitsandbytes)}; '
                     'd["gpus"]=[torch.cuda.get_device_name(i) for i in range(4)]; '
                     'Path(__import__("sys").argv[1]).write_text(json.dumps(d,indent=2))')
            launcher([sys.executable, '-c', probe, str(root / 'environment.json')], env, repo, root, root / 'environment.tail.log')
        for variant, sft_spec, sft_spec_path in specs:
            exp = root / 'experiments' / variant['label']
            data, lengths = Path(sft_spec['data']), Path(sft_spec['lengths'])
            print(f'[qwen3-two] {variant["label"]}: {args.stage}', flush=True)
            try:
                done = exp / 'sft/complete.json'
                if not done.exists() and (exp / 'sft/fit_complete.json').exists():
                    recover_sft_export(sft_spec)
                if not done.exists():
                    if args.stage == 'eval':
                        raise ValueError('Complete SFT before --stage eval')
                    output = Path(sft_spec['output'])
                    resume = resumable_checkpoint(output, 'sft')
                    if output.exists() and any(not p.name.startswith('.incomplete-') for p in output.iterdir()) and not resume:
                        raise ValueError('Nonempty SFT output without resumable checkpoint')
                    launcher(worker_command(repo, sft_spec_path, 'sft', resume), env, repo, root, exp / 'sft/train.tail.log')
                model = Path(sft_spec['output']) / 'selected_model'
                record = read_json(done)
                if record['model'] != str(model):
                    raise ValueError('SFT export path changed')
                verify_files(model, record['model_sha256'])
                if args.stage in ('all', 'eval'):
                    evaluate(repo, root, config, variant, model, data, lengths, baseline_path(root, variant), env, launcher)
                rl_spec = dict(model=str(model), data=str(data), category=variant['category'], variant=variant,
                    artifacts=str(exp / 'rl'), output=str(weights / variant['label'] / 'rl'), lengths=str(lengths))
                spec_path = exp / 'rl_spec.json'; pin_json(spec_path, rl_spec)
                output, artifacts = Path(rl_spec['output']), Path(rl_spec['artifacts']); artifacts.mkdir(exist_ok=True)
                done = artifacts / 'complete.json'
                if done.exists():
                    record = read_json(done)
                    if record['final_step'] != 350 or record['parent_model'] != str(model):
                        raise ValueError('Completed RL provenance changed')
                    verify_files(output, record['checkpoint_files'])
                elif args.stage in ('all', 'train'):
                    resume = resumable_checkpoint(output, 'rl')
                    step = int(Path(resume).name.split('-')[-1]) if resume else 0
                    if step > 350:
                        raise ValueError('Existing RL exceeds350 steps')
                    if step != 350:
                        if output.exists() and any(not p.name.startswith('.incomplete-') for p in output.iterdir()) and not resume:
                            raise ValueError('Nonempty RL output without resumable checkpoint')
                        launcher(worker_command(repo, spec_path, 'rl', resume), env, repo, root, artifacts / 'train.tail.log')
                    finish_rl(rl_spec)
                else:
                    raise ValueError('Complete RL before --stage eval')
                if args.stage in ('all', 'eval'):
                    for step in SNAPSHOTS:
                        evaluate(repo, root, config, variant, output / f'checkpoint-{step}', data, lengths,
                                 exp / f'validation/checkpoint-{step}', env, launcher, step)
            finally:
                summarize(root, config)
        print(f'[qwen3-two] Finished {args.stage}: {root / "summary.md"}', flush=True)


if __name__ == '__main__':
    run(REPO, parser().parse_args())
