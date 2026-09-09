"""Serial Office SFT/eval/RL/eval then Industrial, on the same four devices."""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from reproduction.prepare import CATEGORIES, prepare, write_json, sha256

REPO = Path(__file__).resolve().parents[1]


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--run-name', required=True)
    p.add_argument('--gpus', default='0,1,2,3')
    p.add_argument('--model', default='Qwen/Qwen3-0.6B')
    p.add_argument('--checkpoint-root', default='checkpoints')
    p.add_argument('--dataset', choices=['all', *CATEGORIES], default='all')
    p.add_argument('--stage', choices=['all', 'prepare', 'preflight', 'sft', 'eval-sft', 'rl', 'eval-rl', 'summary'], default='all')
    p.add_argument('--sft-micro-batch', type=int, default=4)
    p.add_argument('--rl-micro-batch', type=int, default=16)
    p.add_argument('--eval-batch-size', type=int, default=2)
    p.add_argument('--resume', action='store_true', help='skip completed stages; resume latest incomplete training checkpoint')
    p.add_argument('--dry-run', action='store_true', help='emit commands; do not train/evaluate or load model')
    p.add_argument('--max-steps', type=int, default=-1, help='GPU smoke only; never mix with a full run')
    return p


def latest_checkpoint(path, require_reference=False):
    candidates = [p for p in path.glob('checkpoint-*') if (p / 'trainer_state.json').is_file()
                  and (not require_reference or (p / 'reference_model.pt').is_file())]
    return str(max(candidates, key=lambda p: int(p.name.split('-')[-1]))) if candidates else None


def summarize(root):
    rows = []
    for category in CATEGORIES:
        for stage in ('sft', 'rl'):
            path = root / category / stage / 'test.metrics.json'
            train = root / category / stage / 'training.json'
            if path.exists():
                data = json.loads(path.read_text())
                record = json.loads(train.read_text())
                rows.append({'dataset': category, 'stage': 'SFT' if stage == 'sft' else 'SFT+RL',
                    'selected_checkpoint': record['best_checkpoint'],
                    'selected_step': record['selected_step'],
                    **{k: data[k] for k in ('samples', 'HR@5', 'Recall@5', 'HR@10', 'Recall@10', 'NDCG@5', 'NDCG@10')}})
    columns = ['dataset', 'stage', 'samples', 'HR@5', 'Recall@5', 'HR@10', 'Recall@10', 'NDCG@5', 'NDCG@10', 'selected_checkpoint', 'selected_step']
    with (root / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    write_json(root / 'summary.json', rows)
    lines = ['| ' + ' | '.join(columns[:9]) + ' |', '|' + '|'.join(['---'] * 9) + '|']
    for row in rows:
        lines.append('| ' + ' | '.join(f'{row[k]:.6f}' if isinstance(row[k], float) else str(row[k]) for k in columns[:9]) + ' |')
    if not rows:
        lines.append('\nNo training/evaluation metrics yet. No improvement is assumed.')
    (root / 'summary.md').write_text('\n'.join(lines) + '\n')


def main():
    args = parser().parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_name):
        raise ValueError('run-name must be a simple directory name')
    devices = args.gpus.split(',')
    if len(devices) != 4 or len(set(devices)) != 4 or any(not re.fullmatch(r'\d+|GPU-[A-Za-z0-9-]+', d) for d in devices):
        raise ValueError('Specify exactly four distinct GPU indices or UUIDs')
    for micro in (args.sft_micro_batch, args.rl_micro_batch):
        if micro <= 0 or 1024 % (4 * micro):
            raise ValueError('Micro batches must divide 256; effective batch stays 1024')
    if args.rl_micro_batch % 16:
        raise ValueError('RL local micro batch must be divisible by 16')
    if args.eval_batch_size < 1:
        raise ValueError('Evaluation batch must be positive')
    os.chdir(REPO)
    root = REPO / 'results' / args.run_name
    checkpoint_root = Path(args.checkpoint_root).resolve() / args.run_name
    if checkpoint_root == root or root in checkpoint_root.parents:
        raise ValueError('Checkpoints must be stored outside results/<run_name>')
    root.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if k not in ('stage', 'dataset', 'resume', 'dry_run')}
    config.update(checkpoint_root=str(checkpoint_root), recipe='official-5f4f733-qwen3-history50-v1', world_size=4,
        effective_batch=1024, sft_accumulation=256 // args.sft_micro_batch,
        rl_accumulation=256 // args.rl_micro_batch, rl_generations=16)
    cfg_path = root / 'run_config.json'
    if cfg_path.exists() and json.loads(cfg_path.read_text()) != config:
        raise ValueError('Run configuration changed. Use a new run-name.')
    write_json(cfg_path, config)
    invocation = {'time': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'argv': sys.argv, 'arguments': vars(args)}
    with (root / 'invocations.jsonl').open('a') as f:
        f.write(json.dumps(invocation) + '\n')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.gpus, TOKENIZERS_PARALLELISM='false', WANDB_MODE='disabled', NCCL_IB_DISABLE='1')
    env['PYTHONPATH'] = str(REPO) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env['HF_DATASETS_CACHE'] = str(root / '.cache')
    source_files = list((REPO / 'reproduction').glob('*.py')) + [REPO / 'data.py', REPO / 'minionerec_trainer.py', REPO / 'requirements-reproduction.txt', REPO / 'config/zero2_opt.yaml', REPO / 'scripts/reproduce.sh']
    source_hashes = {str(p.relative_to(REPO)): sha256(p) for p in source_files}
    source_dir = root / 'source'
    if (root / 'source_sha256.json').exists():
        if json.loads((root / 'source_sha256.json').read_text()) != source_hashes:
            raise ValueError('Implementation changed since this run began; use a new run-name')
    else:
        for p in source_files:
            target = source_dir / p.relative_to(REPO)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
        write_json(root / 'source_sha256.json', source_hashes)
    def launch(command, log):
        log.parent.mkdir(parents=True, exist_ok=True)
        entry = {'time': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'argv': command,
            'cwd': str(REPO), 'env': {'CUDA_VISIBLE_DEVICES': args.gpus}, 'dry_run': args.dry_run}
        with (root / 'commands.jsonl').open('a') as f:
            f.write(json.dumps(entry) + '\n')
        with (root / 'commands.sh').open('a') as f:
            f.write('CUDA_VISIBLE_DEVICES=' + shlex.quote(args.gpus) + ' ' + shlex.join(command) + '\n')
        print(shlex.join(command), flush=True)
        if args.dry_run:
            return
        with log.open('a') as f:
            proc = subprocess.Popen(command, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                for line in proc.stdout:
                    print(line, end='', flush=True)
                    f.write(line)
                    f.flush()
                code = proc.wait()
            except BaseException:
                proc.terminate()
                proc.wait()
                raise
        if code:
            raise RuntimeError(f'Command exited {code}; see {log}')
    if not args.dry_run and args.stage not in ('prepare', 'summary'):
        # Run in a child after applying CUDA_VISIBLE_DEVICES, before any CUDA import.
        launch([sys.executable, '-c', 'import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==4, "Exactly four CUDA GPUs required"; print(torch.__version__); print([torch.cuda.get_device_name(i) for i in range(4)])'], root / 'environment.log') if args.stage != 'preflight' else None
        launch([sys.executable, '-m', 'pip', 'freeze'], root / 'environment.log')
    torchrun = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4']
    # Preserve official RL ZeRO-2 backend, not a replacement TRL trainer.
    accelerate = [sys.executable, '-m', 'accelerate.commands.launch', '--config_file', 'config/zero2_opt.yaml', '--num_processes', '4', '--main_process_port', '29503']
    datasets = CATEGORIES if args.dataset == 'all' else [args.dataset]
    for category in datasets:
        cat = root / category
        data = cat / 'data'
        if args.stage == 'summary':
            continue
        report = prepare(REPO / 'data/Amazon', data, category)
        print(f'{category}: history recovery verified; {report["stats"]["users"]} users', flush=True)
        if args.stage == 'prepare':
            continue
        lengths = cat / 'lengths.json'
        if lengths.exists():
            measured = json.loads(lengths.read_text())
            if measured['model'] != args.model or measured['data_audit_sha256'] != sha256(data / 'audit.json'):
                raise ValueError('Token preflight model/data mismatch; use a new run-name')
        if not lengths.exists():
            launch([sys.executable, '-m', 'reproduction.preflight', '--model', args.model, '--data', str(data),
                    '--category', category, '--output', str(lengths)], cat / 'preflight.log')
        if args.stage == 'preflight':
            continue
        stages = ('sft', 'eval-sft', 'rl', 'eval-rl') if args.stage == 'all' else [args.stage]
        for stage in stages:
            name = stage.removeprefix('eval-')
            artifacts = cat / name
            output = checkpoint_root / category / name
            done = artifacts / (stage + '.complete.json')
            if done.exists():
                if not args.resume:
                    raise ValueError(f'{stage} already complete; use --resume or a new run-name')
                continue
            if not stage.startswith('eval-'):
                parent = args.model if name == 'sft' else str(checkpoint_root / category / 'sft' / 'selected_model')
                if name == 'rl' and not args.dry_run and not (cat / 'sft' / 'sft.complete.json').exists():
                    raise ValueError('RL requires completed SFT for this category/run')
                resume = latest_checkpoint(output, require_reference=name == 'rl') if args.resume else None
                if output.exists() and any(output.iterdir()) and not resume:
                    raise ValueError(f'Nonempty checkpoint directory without a resumable checkpoint: {output}')
                launcher = torchrun if name == 'sft' else accelerate
                command = launcher + ['-m', 'reproduction.train', '--stage', name, '--model', parent,
                    '--data', str(data), '--category', category, '--output', str(output), '--artifacts', str(artifacts),
                    '--lengths', str(lengths), '--micro-batch', str(args.sft_micro_batch if name == 'sft' else args.rl_micro_batch), '--max-steps', str(args.max_steps)]
                if resume:
                    command += ['--resume', resume]
                launch(command, artifacts / 'train.log')
            else:
                if not args.dry_run and not (artifacts / (name + '.complete.json')).exists():
                    raise ValueError(f'Complete {name} before evaluation')
                model = str(output / 'selected_model')
                for split in ('valid', 'test'):
                    launch(torchrun + ['-m', 'reproduction.evaluate', '--model', model, '--data', str(data),
                        '--category', category, '--output', str(artifacts), '--split', split,
                        '--batch-size', str(args.eval_batch_size), '--lengths', str(lengths)], artifacts / f'eval-{split}.log')
            if not args.dry_run:
                write_json(done, {'stage': stage, 'completed_at': datetime.datetime.now(datetime.timezone.utc).isoformat()})
            summarize(root)
    summarize(root)


if __name__ == '__main__':
    main()
