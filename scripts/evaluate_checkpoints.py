#!/usr/bin/env python3
"""Validate saved checkpoints serially; never train, select a model, or read test results."""
import argparse
import ast
import csv
import datetime
import fcntl
import itertools
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproduction.evaluate import metrics
from reproduction.prepare import CATEGORIES, sha256, write_json

REPO = Path(__file__).resolve().parents[1]
METRICS = ('HR@5', 'Recall@5', 'HR@10', 'Recall@10', 'NDCG@5', 'NDCG@10')
EVAL_SOURCE = ('reproduction/__init__.py', 'reproduction/evaluate.py',
               'reproduction/contracts.py', 'reproduction/prepare.py')


def read_json(path):
    return json.loads(Path(path).read_text())


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name', required=True)
    p.add_argument('--dataset', choices=CATEGORIES, default=CATEGORIES[0])
    p.add_argument('--steps', type=int, nargs='+', default=[175, 350, 1400, 1575])
    p.add_argument('--gpus', help='Exactly four IDs; default: original run config')
    p.add_argument('--checkpoint-root', help='Parent of <run-name>; default: original run config')
    p.add_argument('--batch-size', type=int, help='Per-GPU evaluation batch; default: original run config')
    p.add_argument('--diagnostic-name', default='checkpoint_validation')
    p.add_argument('--reevaluate-selected', action='store_true',
                   help='Reevaluate SFT selected_model and the selected RL checkpoint instead of copying existing valid results')
    p.add_argument('--dry-run', action='store_true', help='Audit results and print plan; no writes, CUDA imports or model loads')
    return p


def evaluation_command(model, data, category, output, lengths, batch_size):
    return [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4',
            '-m', 'reproduction.evaluate', '--model', str(model), '--data', str(data),
            '--category', category, '--output', str(output), '--split', 'valid',
            '--batch-size', str(batch_size), '--lengths', str(lengths)]


def audit_predictions(directory, data, category):
    """Check full validation coverage, row labels, legal ranked candidates and metric values."""
    report = read_json(directory / 'valid.metrics.json')
    conf = report['generation_config']
    expected = {'num_beams': 50, 'num_return_sequences': 50, 'do_sample': False,
                'temperature': 1.0, 'length_penalty': 0.0, 'use_cache': True}
    if any(conf.get(k) != v for k, v in expected.items()):
        raise ValueError(f'Unexpected generation settings: {directory}')
    if report['split'] != 'valid' or report['invalid_candidates'] != 0:
        raise ValueError(f'Not a clean validation evaluation: {directory}')
    catalog = {}
    for item, sid in read_json(data / f'{category}.index.json').items():
        catalog.setdefault(''.join(sid), []).append(item)
    metric_rows = []
    with (data / 'valid.csv').open() as stream, (directory / 'valid.predictions.jsonl').open() as predictions:
        for source, line in itertools.zip_longest(csv.DictReader(stream), predictions):
            if source is None or line is None:
                raise ValueError(f'Validation coverage mismatch: {directory}')
            row = json.loads(line)
            expected_row = {'sample_id': source['sample_id'], 'user_id': source['user_id'],
                            'target_position': int(source['target_position']),
                            'history_item_id': ast.literal_eval(source['history_item_id']),
                            'target_item_id': source['item_id'], 'target_sid': source['item_sid']}
            if any(row.get(k) != v for k, v in expected_row.items()):
                raise ValueError(f'Validation row/label mismatch: {source["sample_id"]}')
            candidates, scores = row['predictions'], row['scores']
            if (len(candidates) != 50 or len(set(candidates)) != 50
                    or any(s not in catalog for s in candidates)
                    or row['predicted_item_id_groups'] != [catalog[s] for s in candidates]
                    or len(scores) != 50 or not all(math.isfinite(s) for s in scores)
                    or any(a < b - 1e-6 for a, b in zip(scores, scores[1:]))):
                raise ValueError(f'Invalid candidates/scores: {source["sample_id"]}')
            metric_rows.append({'target_sid': row['target_sid'], 'predictions': candidates})
    computed = metrics(metric_rows)
    if any(not math.isclose(report[k], value, rel_tol=0, abs_tol=1e-12)
           for k, value in computed.items()):
        raise ValueError(f'Metrics do not match predictions: {directory}')
    return report


def result_hashes(directory):
    return {name: sha256(directory / name) for name in ('valid.metrics.json', 'valid.predictions.jsonl')}


def model_files(model, step=None):
    """Only inference weights/config/tokenizer are needed; no optimizer or RL reference."""
    required = ['config.json', 'tokenizer_config.json', 'tokenizer.json']
    if (model / 'model.safetensors').is_file():
        required.append('model.safetensors')
    elif (model / 'pytorch_model.bin').is_file():
        required.append('pytorch_model.bin')
    else:
        index = next((name for name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json')
                      if (model / name).is_file()), None)
        if index is None:
            raise ValueError(f'Missing inference weights: {model}; choose saved steps with --steps')
        shards = sorted(set(read_json(model / index)['weight_map'].values()))
        if not shards or any(Path(name).name != name for name in shards):
            raise ValueError(f'Invalid weight shard index: {model / index}')
        required += [index, *shards]
    if step is not None:
        required.append('trainer_state.json')
        if not (model / 'trainer_state.json').is_file() or read_json(model / 'trainer_state.json')['global_step'] != step:
            raise ValueError(f'Checkpoint global_step does not match requested step {step}: {model}')
    for name in ('generation_config.json', 'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt'):
        if (model / name).exists():
            required.append(name)
    for name in required:
        if not (model / name).is_file() or (model / name).stat().st_size == 0:
            raise ValueError(f'Missing/empty model or tokenizer file: {model / name}')
    return sorted(required)


def build_plan(repo, args):
    for name in (args.run_name, args.diagnostic_name):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
            raise ValueError('Run/diagnostic names must be simple directory names')
    if not args.steps or any(step <= 0 for step in args.steps) or len(set(args.steps)) != len(args.steps):
        raise ValueError('Specify distinct positive checkpoint steps')
    root = repo / 'results' / args.run_name
    cat = root / args.dataset
    config = read_json(root / 'run_config.json')
    gpus = args.gpus if args.gpus is not None else config['gpus']
    devices = gpus.split(',')
    if len(devices) != 4 or len(set(devices)) != 4 or any(not re.fullmatch(r'\d+|GPU-[A-Za-z0-9-]+', d) for d in devices):
        raise ValueError('Specify exactly four distinct GPU indices or UUIDs')
    batch = args.batch_size if args.batch_size is not None else config['eval_batch_size']
    if batch < 1:
        raise ValueError('Evaluation batch must be positive')
    weights = (Path(args.checkpoint_root).resolve() / args.run_name
               if args.checkpoint_root else Path(config['checkpoint_root']))
    if not weights.is_absolute():
        weights = (repo / weights).resolve()
    data, lengths = cat / 'data', cat / 'lengths.json'
    audit = read_json(data / 'audit.json')
    if (audit['category'] != args.dataset or audit['history_limit'] != 50
            or read_json(lengths)['data_audit_sha256'] != sha256(data / 'audit.json')):
        raise ValueError('Data/length audit mismatch')
    # Deliberately read only valid.csv and its catalog; never test.csv/results.
    for name in ('valid.csv', f'{args.dataset}.index.json'):
        if sha256(data / name) != audit['generated_sha256'][name]:
            raise ValueError(f'Prepared validation data changed: {name}')
    recorded = read_json(root / 'source_sha256.json')
    for name in EVAL_SOURCE:
        if sha256(repo / name) != recorded[name] or sha256(root / 'source' / name) != recorded[name]:
            raise ValueError(f'Evaluation source differs from this run: {name}; use the matching fixed code')
    if 'use_model_defaults=False' not in (repo / 'reproduction/evaluate.py').read_text():
        raise ValueError('Qwen3 generation-default fix is required; old sampled results cannot be reused')
    output = cat / 'diagnostics' / args.diagnostic_name
    training = {stage: read_json(cat / stage / 'training.json') for stage in ('sft', 'rl')}
    jobs = []
    reuse_conf = None
    for stage, step in [('sft', 0), *[('rl', step) for step in sorted(args.steps)]]:
        selected = stage == 'sft' or step == training['rl']['selected_step']
        label = 'sft' if stage == 'sft' else f'checkpoint-{step}'
        model = weights / args.dataset / stage / ('selected_model' if stage == 'sft' else label)
        source = cat / stage
        reuse = (selected and not args.reevaluate_selected
                 and (source / f'eval-{stage}.complete.json').is_file()
                 and all((source / name).is_file() for name in ('valid.metrics.json', 'valid.predictions.jsonl')))
        reuse_hashes = None
        if reuse:
            report = audit_predictions(source, data, args.dataset)
            if reuse_conf is not None and report['generation_config'] != reuse_conf:
                raise ValueError('Existing SFT/RL validation generation settings differ')
            reuse_conf = report['generation_config']
            # Imported SFT training.json may refer to its original run; the evaluated
            # export must still be the current category/stage selected_model.
            recorded_model = Path(config['checkpoint_root']) / args.dataset / stage / 'selected_model'
            if Path(report['model']) != recorded_model:
                raise ValueError(f'Unexpected selected-model evaluation path: {report["model"]}')
            reuse_hashes = result_hashes(source)
        jobs.append({'label': label, 'stage': stage, 'rl_step': step,
                     'source_step': training['sft']['selected_step'] if stage == 'sft' else step,
                     'selected_by_training': selected, 'model': str(model),
                     'reuse_from': str(source) if reuse else None, 'reuse_sha256': reuse_hashes,
                     'command': evaluation_command(model, data, args.dataset, output / label, lengths, batch)})
    inputs = [root / 'run_config.json', root / 'source_sha256.json', lengths, data / 'audit.json',
              data / 'valid.csv', data / f'{args.dataset}.index.json',
              *[cat / stage / 'training.json' for stage in ('sft', 'rl')]]
    if (cat / 'rl/trainer_state.json').is_file():
        inputs.append(cat / 'rl/trainer_state.json')
    return {'run_name': args.run_name, 'dataset': args.dataset, 'gpus': gpus, 'world_size': 4,
            'batch_size': batch, 'split': 'valid', 'output': str(output), 'data': str(data),
            'jobs': jobs, 'input_sha256': {str(p.relative_to(root)): sha256(p) for p in inputs},
            'source_sha256': {name: sha256(repo / name) for name in (*EVAL_SOURCE,
                'scripts/evaluate_checkpoints.py', 'scripts/evaluate_checkpoints.sh')},
            'purpose': 'validation diagnostics only; no change to model selection or final test report'}


def launch(command, env, repo, output, log):
    entry = {'time': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'argv': command,
             'cwd': str(repo), 'env': {k: env[k] for k in ('CUDA_VISIBLE_DEVICES', 'HF_HUB_OFFLINE', 'TOKENIZERS_PARALLELISM')}}
    with (output / 'commands.jsonl').open('a') as f:
        f.write(json.dumps(entry) + '\n')
    with (output / 'commands.sh').open('a') as f:
        f.write('env ' + ' '.join(f'{k}={shlex.quote(v)}' for k, v in entry['env'].items())
                + ' ' + shlex.join(command) + '\n')
    print(shlex.join(command), flush=True)
    with log.open('a') as f:
        proc = subprocess.Popen(command, cwd=repo, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
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
        raise RuntimeError(f'Command exited {code}; see {log}; rerun the same command to retry this checkpoint')


def summarize(plan):
    output = Path(plan['output'])
    state = Path(plan['data']).parent / 'rl/trainer_state.json'
    rewards = {r['step']: r['eval_reward'] for r in read_json(state).get('log_history', [])
               if 'eval_reward' in r} if state.is_file() else {}
    rows = []
    for job in plan['jobs']:
        directory = output / job['label']
        if not (directory / 'complete.json').is_file():
            continue
        report = read_json(directory / 'valid.metrics.json')
        rows.append({'checkpoint': job['label'], 'rl_step': job['rl_step'], 'source_step': job['source_step'],
                     'source': ('reused_selected_export' if job['reuse_from'] else
                                'evaluated_selected_export' if job['stage'] == 'sft' else 'evaluated_checkpoint'),
                     'selected_by_training': job['selected_by_training'],
                     'eval_reward': rewards.get(job['rl_step'], '') if job['stage'] == 'rl' else '',
                     'samples': report['samples'], **{key: report[key] for key in METRICS},
                     **{f'delta_{key}': report[key] - (rows[0][key] if rows else report[key]) for key in METRICS},
                     'model': report['model']})
    write_json(output / 'summary.json', rows)
    if not rows:
        return
    with (output / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    columns = ('checkpoint', 'rl_step', 'eval_reward', 'HR@5', 'HR@10', 'NDCG@5', 'NDCG@10', 'delta_HR@10', 'delta_NDCG@10')
    lines = ['验证集 checkpoint 对比；SFT 为 RL step 0，source_step 记录其 SFT 训练步数。',
             'HR=Recall；delta 为相对 SFT 的绝对差值。eval_reward 是训练期间 G16 采样奖励，与 beam50 指标不同。',
             '复用的 RL 结果来自 training.json 所记选中步数的 selected_model 导出；未独立比对导出与 checkpoint 权重。',
             '这里只诊断验证集，不更改原 selected_model、选模规则或测试集汇总。', '',
             '| ' + ' | '.join(columns) + ' |', '|' + '|'.join(['---'] * len(columns)) + '|']
    for row in rows:
        lines.append('| ' + ' | '.join(f'{row[k]:.6f}' if isinstance(row[k], float) else str(row[k]) for k in columns) + ' |')
    lines += ['', f'已完成 {len(rows)}/{len(plan["jobs"])} 项；同一命令重跑会校验并跳过已完成项。']
    (output / 'summary.md').write_text('\n'.join(lines) + '\n')


def execute_plan(repo, plan, launcher=launch):
    output, data = Path(plan['output']), Path(plan['data'])
    output.mkdir(parents=True, exist_ok=True)
    # Kernel lock is released even after interruption; the file may remain safely.
    with (output / '.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f'Another checkpoint evaluation is running in {output}') from exc
        config = output / 'config.json'
        if config.exists() and read_json(config) != plan:
            raise ValueError('Diagnostic inputs/config changed; use a different --diagnostic-name')
        if not config.exists() and any(p.name != '.lock' for p in output.iterdir()):
            raise ValueError(f'Nonempty diagnostic directory without config: {output}')
        # Fail before any GPU work when a requested checkpoint has been deleted/incompletely saved.
        pending_files = {}
        reference_conf = None
        for job in plan['jobs']:
            target = output / job['label']
            if (target / 'complete.json').is_file():
                if read_json(target / 'complete.json')['sha256'] != result_hashes(target):
                    raise ValueError(f'Completed diagnostic was modified: {target}')
                report = audit_predictions(target, data, plan['dataset'])
                if reference_conf is not None and report['generation_config'] != reference_conf:
                    raise ValueError('Generation settings differ between checkpoints')
                reference_conf = report['generation_config']
            elif not job['reuse_from']:
                pending_files[job['label']] = model_files(Path(job['model']), job['rl_step'] or None)
        write_json(config, plan)
        for name in plan['source_sha256']:
            destination = output / 'source' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repo / name, destination)
        root = repo / 'results' / plan['run_name']
        for name in plan['input_sha256']:
            destination = output / 'inputs' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, destination)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=plan['gpus'], TOKENIZERS_PARALLELISM='false',
                   WANDB_MODE='disabled', NCCL_IB_DISABLE='1', HF_HUB_OFFLINE='1')
        env['PYTHONPATH'] = str(repo) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
        environment_checked = False
        for job in plan['jobs']:
            target = output / job['label']
            target.mkdir(exist_ok=True)
            if (target / 'complete.json').is_file():
                print(f'Skip verified {job["label"]}', flush=True)
                continue
            if job['reuse_from']:
                source = Path(job['reuse_from'])
                if result_hashes(source) != job['reuse_sha256']:
                    raise ValueError(f'Reused results changed during execution: {source}')
                for name in job['reuse_sha256']:
                    shutil.copyfile(source / name, target / name)
                provenance = {'reuse_from': str(source), 'sha256': job['reuse_sha256'],
                              'source_step': job['source_step'],
                              'weight_verification': 'step association from training.json; export/checkpoint weights not compared'}
                print(f'Reuse validation: {job["label"]} <- {source}', flush=True)
            else:
                model = Path(job['model'])
                provenance = {'model': str(model), 'source_step': job['source_step'],
                              'model_sha256': {name: sha256(model / name) for name in pending_files[job['label']]}}
                old = target / 'provenance.json'
                if old.exists() and read_json(old) != provenance:
                    raise ValueError(f'Checkpoint changed since interrupted evaluation: {model}')
                write_json(old, provenance)
                if not environment_checked:
                    check = 'import torch; print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda); assert torch.cuda.is_available() and torch.cuda.device_count()==4, "Exactly four CUDA GPUs required"'
                    if reference_conf is not None:
                        version = reference_conf['transformers_version']
                        check += f'; import transformers; assert transformers.__version__ == {version!r}, "Use the same Transformers version as the baseline"'
                    launcher([sys.executable, '-c', check], env, repo, output, output / 'environment.log')
                    launcher([sys.executable, '-m', 'pip', 'freeze'], env, repo, output, output / 'environment.log')
                    environment_checked = True
                launcher(job['command'], env, repo, output, target / 'eval-valid.log')
            report = audit_predictions(target, data, plan['dataset'])
            if not job['reuse_from'] and Path(report['model']) != Path(job['model']):
                raise ValueError(f'Evaluation model does not match the requested checkpoint: {target}')
            if reference_conf is not None and report['generation_config'] != reference_conf:
                raise ValueError('Generation settings differ between checkpoints; results not marked complete')
            reference_conf = report['generation_config']
            write_json(target / 'provenance.json', provenance)
            write_json(target / 'complete.json', {'sha256': result_hashes(target), 'split': 'valid',
                       'completed_at': datetime.datetime.now(datetime.timezone.utc).isoformat()})
            summarize(plan)
        summarize(plan)
        print(f'Validation summary: {output / "summary.md"}', flush=True)


def main():
    args = parser().parse_args()
    plan = build_plan(REPO, args)
    if args.dry_run:
        print(f'DRY RUN: validation only, GPUs={plan["gpus"]}, output={plan["output"]}')
        for job in plan['jobs']:
            if job['reuse_from']:
                print(f'{job["label"]}: reuse verified validation from {job["reuse_from"]}')
            elif (Path(plan['output']) / job['label'] / 'complete.json').is_file():
                print(f'{job["label"]}: existing diagnostic; real run will verify before skipping')
            else:
                try:
                    model_files(Path(job['model']), job['rl_step'] or None)
                except ValueError as exc:
                    print(f'UNAVAILABLE LOCALLY: {exc}')
                print(f'CUDA_VISIBLE_DEVICES={plan["gpus"]} ' + shlex.join(job['command']))
        return
    execute_plan(REPO, plan)


if __name__ == '__main__':
    main()
