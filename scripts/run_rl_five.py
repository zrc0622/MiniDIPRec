#!/usr/bin/env python3
"""Five serial sub-1B experiments, RL350, bounded artifacts under results2."""
import argparse
import ast
import csv
import fcntl
import gzip
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproduction.prepare import CATEGORIES, sha256, write_json
from scripts import evaluate_checkpoints as evaluation
from scripts.five_utils import SNAPSHOTS, prepare_official, resumable_checkpoint, variants
from scripts.run_rl_six import finish_rl, recover_sft_export
from scripts.six_utils import (audit_compact, compress_predictions, file_manifest, launch,
                               pin_json, read_json, verify_files)

REPO = Path(__file__).resolve().parents[1]


def source_files(repo):
    names = ['data.py', 'minionerec_trainer.py', 'requirements-reproduction.txt', 'config/zero2_opt.yaml',
             'scripts/run_rl_five.py', 'scripts/train_rl_five.py', 'scripts/five_utils.py',
             'scripts/run_rl_six.py', 'scripts/train_rl_six.py', 'scripts/six_utils.py',
             'scripts/evaluate_checkpoints.py', 'scripts/start_rl_from_sft.py',
             'scripts/migrate_rl_config_fix.py', 'scripts/package_results2.py']
    names += [str(p.relative_to(repo)) for p in (repo / 'reproduction').glob('*.py')]
    names += [str(p.relative_to(repo)) for p in (repo / 'reproduction/upstream_minionerec').iterdir() if p.is_file()]
    return sorted(set(names))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name', default='minionerec_five350')
    p.add_argument('--source-suite', default='results/qwen3_six350', help='Previous six-suite directory (not run name)')
    p.add_argument('--source-sft', help='Moved source SFT10 selected_model; hashes must match old complete.json')
    p.add_argument('--checkpoint-root', default='checkpoints', help='Parent for NEW suite weights, outside results/results2')
    p.add_argument('--qwen3-model', default='Qwen/Qwen3-0.6B', help='Original base ID or local directory')
    p.add_argument('--qwen25-model', default='Qwen/Qwen2.5-0.5B', help='Base, not Instruct; ID or local directory')
    p.add_argument('--gpus', default='0,1,2,3')
    p.add_argument('--sft-micro-batch', type=int, default=4)
    p.add_argument('--eval-batch-size', type=int, default=2)
    p.add_argument('--offline', action='store_true', help='Require all original models/tokenizers already cached')
    p.add_argument('--stage', choices=['all', 'prepare', 'train', 'eval'], default='all')
    p.add_argument('--experiments', type=int, nargs='+', choices=range(1, 6), default=list(range(1, 6)))
    p.add_argument('--dry-run', action='store_true', help='Print plan only; no writes, weights or source suite required')
    return p


def absolute(repo, value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else repo / path).resolve()


def settings(repo, args):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.run_name):
        raise ValueError('Use a simple run name')
    devices = args.gpus.split(',')
    if len(devices) != 4 or len(set(devices)) != 4 or any(not re.fullmatch(r'\d+|GPU-[A-Za-z0-9-]+', d) for d in devices):
        raise ValueError('Exactly four distinct GPU IDs required')
    if args.sft_micro_batch <= 0 or 256 % args.sft_micro_batch or args.eval_batch_size <= 0:
        raise ValueError('SFT micro batch must divide256; evaluation batch must be positive')
    root = repo / 'results2' / args.run_name
    weights = absolute(repo, args.checkpoint_root) / args.run_name
    for results in ((repo / 'results').resolve(), (repo / 'results2').resolve()):
        if weights == results or results in weights.parents or weights in results.parents:
            raise ValueError('Checkpoint root must be separate from results and results2')
    for model in (args.qwen3_model, args.qwen25_model):
        if 'instruct' in model.lower() or Path(model).name in ('selected_model', 'final_checkpoint'):
            raise ValueError('Fresh official SFT requires original Qwen bases, not Instruct or selected SFT')
    config = dict(run_name=args.run_name, source_suite=str(absolute(repo, args.source_suite)),
        source_sft=str(absolute(repo, args.source_sft)) if args.source_sft else None,
        checkpoint_root=str(weights), qwen3_model=args.qwen3_model, qwen25_model=args.qwen25_model,
        gpus=args.gpus, sft_micro_batch=args.sft_micro_batch, eval_batch_size=args.eval_batch_size,
        offline=args.offline, variants=variants(), stop_after_steps=350, snapshot_steps=SNAPSHOTS,
        seed=42, evaluation='valid only; deterministic beam50; own SFT baseline',
        artifacts='results2; shared data/source; gzip predictions; 256KiB console tail +64KiB warnings per process')
    if (root / 'suite_config.json').exists() and read_json(root / 'suite_config.json') != config:
        raise ValueError('Suite settings changed; use saved resume.sh or a new run name')
    return root, config


def source_sft(config):
    source = Path(config['source_suite'])
    suite = read_json(source / 'suite_config.json')
    old = read_json(source / 'experiments/06_history10/sft/complete.json')
    if suite['dataset'] != CATEGORIES[0] or old['selected_step'] != 378:
        raise ValueError('Expected completed Office history10 SFT step378')
    model = Path(config['source_sft'] or old['model'])
    evaluation.model_files(model)
    verify_files(model, old['model_sha256'])
    data = source / 'shared/history10/data'
    audit = read_json(data / 'audit.json')
    lengths = read_json(source / 'experiments/06_history10/lengths.json')
    if audit['history_limit'] != 10 or audit['category'] != CATEGORIES[0] or \
            lengths['data_audit_sha256'] != sha256(data / 'audit.json'):
        raise ValueError('Source SFT10 data/preflight mismatch')
    verify_files(data, audit['generated_sha256'])
    return model, data, old


def compare_office(imported, official):
    """Same labels/history rows across imported H10 and original Office baseline."""
    for split in ('train', 'valid'):
        with (imported / f'{split}.csv').open() as a, (official / f'{split}.csv').open() as b:
            rows_a, rows_b = list(csv.DictReader(a)), list(csv.DictReader(b))
        if len(rows_a) != len(rows_b):
            raise ValueError('Imported/official Office row count differs')
        for old, new in zip(rows_a, rows_b):
            if old.keys() != new.keys():
                raise ValueError('Imported/official Office schema differs')
            for key in old:
                x, y = old[key], new[key]
                if key.startswith('history_item_'):
                    x, y = ast.literal_eval(x), ast.literal_eval(y)
                if x != y:
                    raise ValueError(f'Imported/official Office row differs: {split}/{key}')
    for name in (f'{CATEGORIES[0]}.index.json', f'{CATEGORIES[0]}.item.json', 'info.txt'):
        if sha256(imported / name) != sha256(official / name):
            raise ValueError(f'Imported/official Office catalog differs: {name}')


def setup(repo, root, config):
    weights = Path(config['checkpoint_root'])
    pending = root / 'setup_pending.json'
    if pending.exists() and not (root / 'suite_config.json').exists():
        record = read_json(pending)
        if record['config'] != config:
            raise ValueError('Interrupted setup settings changed')
        shared = root / 'shared' if (root / 'shared').exists() else Path(record['shared'])
        staged_weights = weights if weights.exists() else Path(record['weights'])
        manifest = read_json(shared / 'manifest.json')
        verify_files(shared, manifest['artifacts'])
        verify_files(staged_weights / 'imported_sft10', manifest['model_sha256'])
        if not weights.exists():
            staged_weights.rename(weights)
        if not (root / 'shared').exists():
            shared.rename(root / 'shared')
        pin_json(root / 'suite_config.json', config)
        pending.unlink()
    if (root / 'suite_config.json').exists():
        manifest = read_json(root / 'shared/manifest.json')
        verify_files(root / 'shared', manifest['artifacts'])
        verify_files(weights / 'imported_sft10', manifest['model_sha256'])
        verify_files(repo, manifest['source_sha256'])
        return
    if any(p.name != '.five.lock' for p in root.iterdir()) or weights.exists():
        raise ValueError('Nonempty destination without complete setup; use a new run name')
    model, data, record = source_sft(config)
    source = Path(config['source_suite'])
    # Previous six-suite sources used by runs1/2 must still be identical.
    verify_files(repo, read_json(source / 'shared/manifest.json')['source_sha256'])
    weights.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.five-prepare-', dir=root.parent))
    weight_stage = Path(tempfile.mkdtemp(prefix='.five-weights-', dir=weights.parent))
    try:
        shared = staging / 'shared'
        shutil.copytree(data, shared / 'imported_office/data')
        write_json(shared / 'source_sft_complete.json', record)
        write_json(shared / 'source_suite_config.json', read_json(source / 'suite_config.json'))
        for category in CATEGORIES:
            prepare_official(repo / 'data/Amazon', shared / category / 'data', category)
        compare_office(shared / 'imported_office/data', shared / CATEGORIES[0] / 'data')
        model_hashes = record['model_sha256']
        shutil.copytree(model, weight_stage / 'imported_sft10')
        verify_files(weight_stage / 'imported_sft10', model_hashes)
        hashes = {}
        for name in source_files(repo):
            target = shared / 'source' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(repo / name, target)
            hashes[name] = sha256(target)
        write_json(shared / 'manifest.json', dict(artifacts=file_manifest(shared), source_sha256=hashes,
            model_sha256=model_hashes, source_model=str(model), selected_step=record['selected_step']))
        write_json(pending, dict(config=config, shared=str(shared), weights=str(weight_stage)))
        weight_stage.rename(weights)
        shared.rename(root / 'shared')
        pin_json(root / 'suite_config.json', config)
        pending.unlink()
    finally:
        if not pending.exists():
            for temporary in (staging, weight_stage):
                if temporary.exists():
                    shutil.rmtree(temporary)


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
    command += ['-m', 'scripts.train_rl_five', '--spec', str(spec), '--stage', stage]
    if resume:
        command += ['--resume', str(resume)]
    return command


def evaluate(repo, root, config, variant, model, data, lengths, target, env, launcher, step=None):
    target.mkdir(parents=True, exist_ok=True)
    plan = dict(model=str(model), category=variant['category'], data=str(data), step=step,
                lengths_sha256=sha256(lengths), data_sha256=sha256(data / 'audit.json'),
                batch_size=config['eval_batch_size'], split='valid')
    pin_json(target / 'job.json', plan)
    done = target / 'complete.json'
    if done.exists():
        verify_files(target, read_json(done)['sha256'])
        return audit_compact(target, data, variant['category'])
    files = evaluation.model_files(model, step)
    provenance = dict(model=str(model), step=step, model_sha256={name: sha256(model / name) for name in files})
    pin_json(target / 'provenance.json', provenance)
    if not (target / 'valid.predictions.jsonl.gz').exists():
        raw, metrics = target / 'valid.predictions.jsonl', target / 'valid.metrics.json'
        ready = False
        if raw.exists() and metrics.exists():
            try:
                evaluation.audit_predictions(target, data, variant['category'])
                ready = True
            except (ValueError, KeyError, TypeError, json.JSONDecodeError):
                pass  # Interrupted inference; rerun only this suite-owned incomplete job.
        if not ready:
            launcher(evaluation.evaluation_command(model, data, variant['category'], target, lengths,
                config['eval_batch_size']), env, repo, root, target / 'eval.tail.log')
        evaluation.audit_predictions(target, data, variant['category'])
        compress_predictions(target)
    report = audit_compact(target, data, variant['category'])
    if report['model'] != str(model):
        raise ValueError('Evaluation model identity mismatch')
    hits, gain = 0, 0.
    with gzip.open(target / 'valid.predictions.jsonl.gz', 'rt') as f:
        for line in f:
            row = json.loads(line)
            if row['target_sid'] in row['predictions']:
                rank = row['predictions'].index(row['target_sid']) + 1
                hits += 1
                gain += 1 / math.log2(rank + 1)
    report.update({'HR@50': hits / report['samples'], 'NDCG@50': gain / report['samples']})
    write_json(target / 'valid.metrics.json', report)
    verify_files(model, provenance['model_sha256'])
    write_json(done, {'sha256': {name: sha256(target / name) for name in
        ('valid.metrics.json', 'valid.predictions.jsonl.gz', 'job.json', 'provenance.json')}})
    return report


def baseline_path(root, variant):
    return root / 'baselines' / ('imported_sft10' if variant['recipe'] == 'current' else variant['label'])


def summarize(root, config):
    rows, status = [], []
    for variant in config['variants']:
        exp = root / 'experiments' / variant['label']
        baseline = baseline_path(root, variant)
        points = [step for step in SNAPSHOTS if (exp / f'validation/checkpoint-{step}/complete.json').exists()]
        status.append(dict(experiment=variant['label'],
                           sft_complete=(root / 'shared/source_sft_complete.json').exists() if variant['recipe'] == 'current'
                           else (exp / 'sft/complete.json').exists(),
                           baseline_complete=(baseline / 'complete.json').exists(),
                           rl_complete=(exp / 'rl/complete.json').exists(), evaluated_steps=points))
        if not (baseline / 'complete.json').exists():
            continue
        base = read_json(baseline / 'valid.metrics.json')
        for step in points:
            score = read_json(exp / f'validation/checkpoint-{step}/valid.metrics.json')
            row = dict(experiment=variant['label'], dataset=variant['category'], model=variant['family'],
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
    lines = ['# 五组 RL350 验证结果', '',
             '每组与自己的 SFT 比较；官方配方只完成两轮调度的前350步，不代表完整复现。',
             '01/02复用旧SFT10；03/04/05各自重新SFT；所有RL独立初始化。', '',
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
            sft='01/02 import completedSFT10;03/04/05 freshSFT', rl_updates=350, groups_per_run=22400), indent=2))
        return
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.five.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('This five-experiment suite is already running') from exc
        setup(repo, root, config)
        command = [sys.executable, str(repo / 'scripts/run_rl_five.py'), '--run-name', config['run_name'],
            '--source-suite', config['source_suite'], '--checkpoint-root', str(Path(config['checkpoint_root']).parent),
            '--qwen3-model', config['qwen3_model'], '--qwen25-model', config['qwen25_model'],
            '--gpus', config['gpus'], '--sft-micro-batch', str(config['sft_micro_batch']),
            '--eval-batch-size', str(config['eval_batch_size'])]
        if config['source_sft']:
            command += ['--source-sft', config['source_sft']]
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
            imported = variant['recipe'] == 'current'
            data = root / 'shared' / ('imported_office' if imported else variant['category']) / 'data'
            model = str(weights / 'imported_sft10') if imported else config['qwen3_model' if variant['family']=='qwen3' else 'qwen25_model']
            lengths = root / 'baselines/imported_sft10/lengths.json' if imported else exp / 'lengths.json'
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
            print(f'[five] {variant["label"]}: {args.stage}', flush=True)
            try:
                if variant['recipe'] == 'current':
                    model = weights / 'imported_sft10'
                else:
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
        print(f'[five] Finished {args.stage}: {root / "summary.md"}', flush=True)


if __name__ == '__main__':
    run(REPO, parser().parse_args())
