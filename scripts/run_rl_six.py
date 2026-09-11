#!/usr/bin/env python3
"""Six serial Ranking experiments, each RL350; shared inputs and compressed results."""
import argparse
import csv
import fcntl
import os
from pathlib import Path
import re
import shlex
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproduction.prepare import CATEGORIES, sha256, write_json
from reproduction.run import latest_checkpoint
from scripts.start_rl_from_sft import check_source
from scripts import evaluate_checkpoints as evaluation
from scripts.six_utils import (SNAPSHOTS, variants, read_json, pin_json, file_manifest, verify_files,
                               write_history10, launch, compress_predictions, audit_compact)

REPO = Path(__file__).resolve().parents[1]
SUITE_FILES = ('scripts/run_rl_six.py', 'scripts/train_rl_six.py', 'scripts/six_utils.py',
               'scripts/evaluate_checkpoints.py', 'scripts/start_rl_from_sft.py',
               'scripts/migrate_rl_config_fix.py', 'scripts/package_results.py')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name', required=True)
    p.add_argument('--source-run', default='qwen3_h50_seed42_rl16_fixed')
    p.add_argument('--dataset', choices=CATEGORIES, default=CATEGORIES[0])
    p.add_argument('--gpus', default='0,1,2,3')
    p.add_argument('--checkpoint-root', help='Parent for this suite weights, default source checkpoint parent')
    p.add_argument('--base-model', help='Original Qwen3-0.6B base or cached ID for history10 SFT; never the selected SFT')
    p.add_argument('--sft-micro-batch', type=int)
    p.add_argument('--eval-batch-size', type=int)
    p.add_argument('--stage', choices=['all', 'prepare', 'train', 'eval'], default='all')
    p.add_argument('--experiments', type=int, nargs='+', choices=range(1, 7), default=list(range(1, 7)),
                   help='Optional subset, always executed in numeric order; default all six')
    p.add_argument('--dry-run', action='store_true', help='Print recipes/budgets only; no writes or model access')
    return p


def settings(repo, args):
    for name in (args.run_name, args.source_run):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
            raise ValueError('Use simple run names')
    if args.run_name == args.source_run:
        raise ValueError('Suite and source run must differ')
    devices = args.gpus.split(',')
    if len(devices) != 4 or len(set(devices)) != 4 or any(not re.fullmatch(r'\d+|GPU-[A-Za-z0-9-]+', x) for x in devices):
        raise ValueError('Exactly four distinct GPU IDs required')
    root = repo / 'results' / args.run_name
    # A prepared suite no longer depends on original source artifacts being present.
    saved = root / 'suite_config.json'
    if saved.exists():
        source_config = read_json(root / 'shared/source_run_config.json')
    else:
        source_config = read_json(repo / 'results' / args.source_run / 'run_config.json')
    if any(source_config.get(k) != v for k, v in [('world_size', 4), ('effective_batch', 1024), ('rl_generations', 16), ('max_steps', -1)]):
        raise ValueError('Source must be the completed four-GPU G16/full-recipe SFT run')
    weights = (Path(args.checkpoint_root or Path(source_config['checkpoint_root']).parent).expanduser())
    if not weights.is_absolute():
        weights = repo / weights
    weights = (weights / args.run_name).resolve()
    results = (repo / 'results').resolve()
    if weights == results or results in weights.parents or weights in results.parents:
        raise ValueError('Checkpoint root must be separate from results')
    config = dict(run_name=args.run_name, source_run=args.source_run, dataset=args.dataset, gpus=args.gpus,
                  checkpoint_root=str(weights), base_model=args.base_model or source_config['model'],
                  sft_micro_batch=args.sft_micro_batch if args.sft_micro_batch is not None else source_config['sft_micro_batch'],
                  eval_batch_size=args.eval_batch_size if args.eval_batch_size is not None else source_config['eval_batch_size'],
                  variants=variants(), stop_after_steps=350, snapshot_steps=SNAPSHOTS,
                  seed=42, evaluation='valid only, deterministic beam50; compare to own SFT',
                  artifacts='one shared dataset per history, gzip predictions, bounded console tails, all scalar metrics')
    micro = config['sft_micro_batch']
    if micro <= 0 or 256 % micro or config['eval_batch_size'] <= 0:
        raise ValueError('Invalid SFT micro or eval batch')
    if saved.exists() and read_json(saved) != config:
        raise ValueError('Suite settings changed; use saved resume.sh or a new run name')
    return root, config, source_config


def source_model(repo, config, source_config):
    source = repo / 'results' / config['source_run']
    cat = source / config['dataset']
    model = Path(source_config['checkpoint_root']) / config['dataset'] / 'sft/selected_model'
    if not model.is_absolute():
        model = repo / model
    for name in ('sft.complete.json', 'training.json'):
        if not (cat / 'sft' / name).is_file():
            raise ValueError(f'Completed source SFT required: {cat / "sft" / name}')
    training = read_json(cat / 'sft/training.json')
    evaluation.model_files(model)
    hashes = file_manifest(model)
    if (source / 'sft_import/record.json').exists():
        imported = read_json(source / 'sft_import/record.json')
        if (Path(imported['copied_model']).resolve() != model.resolve()
                or imported['selected_step'] != training['selected_step'] or hashes != imported['model_sha256']):
            raise ValueError('Source SFT import identity mismatch')
    elif Path(training['final_checkpoint']).resolve() != model.resolve():
        raise ValueError('Source selected-model path mismatch')
    return model, hashes, training


def setup(repo, root, config, source_config):
    """Copy data/source once, SFT weights once outside results; atomic preparation."""
    weights = Path(config['checkpoint_root'])
    pending = root / 'setup_pending.json'
    if pending.exists() and not (root / 'suite_config.json').exists():
        transaction = read_json(pending)
        if transaction['config'] != config:
            raise ValueError('Interrupted setup settings changed')
        staged_shared, staged_weights = Path(transaction['shared']), Path(transaction['weights'])
        shared = root / 'shared' if (root / 'shared').exists() else staged_shared
        weight_source = weights if weights.exists() else staged_weights
        manifest = read_json(shared / 'manifest.json')
        verify_files(shared, manifest['artifacts'])
        verify_files(weight_source / 'shared_sft', manifest['model_sha256'])
        if not weights.exists():
            staged_weights.rename(weights)
        if not (root / 'shared').exists():
            staged_shared.rename(root / 'shared')
        pin_json(root / 'suite_config.json', config)
        pending.unlink()
    if (root / 'suite_config.json').exists():
        manifest = read_json(root / 'shared/manifest.json')
        verify_files(root / 'shared', manifest['artifacts'])
        verify_files(weights / 'shared_sft', manifest['model_sha256'])
        for name, digest in manifest['source_sha256'].items():
            if sha256(repo / name) != digest or sha256(root / 'shared/source' / name) != digest:
                raise ValueError(f'Suite source changed: {name}')
        return
    if any(p.name != '.six.lock' for p in root.iterdir()) or weights.exists():
        raise ValueError('Nonempty suite/checkpoint destination without complete setup')
    source = repo / 'results' / config['source_run']
    cat = source / config['dataset']
    current, fixes = check_source(repo, source)
    model, model_hashes, training = source_model(repo, config, source_config)
    audit = read_json(cat / 'data/audit.json')
    lengths = read_json(cat / 'lengths.json')
    if (audit['category'] != config['dataset'] or audit['history_limit'] != 50
            or lengths['data_audit_sha256'] != sha256(cat / 'data/audit.json')):
        raise ValueError('Source history50 data/preflight mismatch')
    names = ['train.csv', 'valid.csv', f'{config["dataset"]}.index.json', f'{config["dataset"]}.item.json', 'info.txt']
    for name in names:
        if sha256(cat / 'data' / name) != audit['generated_sha256'][name]:
            raise ValueError(f'Source prepared data changed: {name}')
    # Never use the already fine-tuned SFT as the base of the history10 SFT experiment.
    if Path(config['base_model']).expanduser().resolve() == model.resolve():
        raise ValueError('history10 requires the original pretrained base, not selected SFT weights')
    weights.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.six-prepare-', dir=root.parent))
    weight_stage = Path(tempfile.mkdtemp(prefix='.six-weights-', dir=weights.parent))
    try:
        shared = staging / 'shared'
        shared.mkdir()
        write_json(shared / 'source_run_config.json', source_config)
        write_json(shared / 'source_sft_training.json', training)
        write_json(shared / 'source_data_audit.json', audit)
        for history in (50, 10):
            dest = shared / f'history{history}'
            data = dest / 'data'
            data.mkdir(parents=True)
            stats = {}
            for name in names:
                if history == 10 and name in ('train.csv', 'valid.csv'):
                    stats[name] = write_history10(cat / 'data' / name, data / name)
                else:
                    shutil.copyfile(cat / 'data' / name, data / name)
            derived = dict(category=config['dataset'], history_limit=history,
                parent_audit_sha256=sha256(cat / 'data/audit.json'), generated_sha256=file_manifest(data),
                changes=stats, policy='same train/valid rows, labels and SID; no test data included')
            write_json(data / 'audit.json', derived)
            if history == 50:
                write_json(dest / 'lengths.json', dict(lengths, data_audit_sha256=sha256(data / 'audit.json')))
        for name in SUITE_FILES:
            current[name] = (repo / name).read_bytes()
        for name, content in current.items():
            dest = shared / 'source' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
        shutil.copytree(model, weight_stage / 'shared_sft')
        verify_files(weight_stage / 'shared_sft', model_hashes)
        write_json(shared / 'manifest.json', dict(artifacts=file_manifest(shared),
            source_sha256={name: sha256(repo / name) for name in current}, model_sha256=model_hashes,
            source_model=str(model), selected_step=training['selected_step'], accepted_import_fixes=fixes))
        write_json(pending, {'config': config, 'shared': str(shared), 'weights': str(weight_stage)})
        weight_stage.rename(weights)
        (staging / 'shared').rename(root / 'shared')
        pin_json(root / 'suite_config.json', config)
        pending.unlink()
    finally:
        # Only temporary directories created in this setup are removed, never old results.
        if staging.exists() and not pending.exists():
            shutil.rmtree(staging)
        if weight_stage.exists() and not pending.exists():
            shutil.rmtree(weight_stage)


def environment(repo, config):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=config['gpus'], TOKENIZERS_PARALLELISM='false', WANDB_MODE='disabled',
               NCCL_IB_DISABLE='1', HF_HUB_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    env['PYTHONPATH'] = str(repo) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    return env


def worker_command(repo, spec_path, stage, resume=None):
    if stage == 'rl':
        command = [sys.executable, '-m', 'accelerate.commands.launch', '--config_file',
                   str(repo / 'config/zero2_opt.yaml'), '--num_processes', '4', '--main_process_port', '29503']
    elif stage == 'sft':
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=4']
    else:
        command = [sys.executable]
    command += ['-m', 'scripts.train_rl_six', '--spec', str(spec_path), '--stage', stage]
    if resume:
        command += ['--resume', str(resume)]
    return command


def evaluation_job(repo, root, config, model, data, lengths, target, env, launcher, step=None):
    target.mkdir(parents=True, exist_ok=True)
    plan = dict(model=str(model), data=str(data), lengths_sha256=sha256(lengths),
                data_sha256=sha256(data / 'audit.json'), step=step, batch_size=config['eval_batch_size'], split='valid')
    pin_json(target / 'job.json', plan)
    done = target / 'complete.json'
    if done.exists():
        verify_files(target, read_json(done)['sha256'])
        return audit_compact(target, data, config['dataset'])
    files = evaluation.model_files(model, step)
    provenance = dict(model=str(model), model_sha256={name: sha256(model / name) for name in files}, step=step)
    pin_json(target / 'provenance.json', provenance)
    # Crash after successful evaluation/compression is finalized without a second inference.
    if (target / 'valid.predictions.jsonl.gz').is_file():
        report = audit_compact(target, data, config['dataset'])
    else:
        raw = target / 'valid.predictions.jsonl'
        if not (raw.is_file() and (target / 'valid.metrics.json').is_file()):
            launcher(evaluation.evaluation_command(model, data, config['dataset'], target, lengths,
                        config['eval_batch_size']), env, repo, root, target / 'eval.tail.log')
        report = evaluation.audit_predictions(target, data, config['dataset'])
        compress_predictions(target)
        audit_compact(target, data, config['dataset'])
    if report['model'] != str(model):
        raise ValueError('Evaluation reported a different model')
    verify_files(model, provenance['model_sha256'])
    write_json(done, {'sha256': {name: sha256(target / name) for name in
        ('valid.metrics.json', 'valid.predictions.jsonl.gz', 'provenance.json', 'job.json')}})
    return report


def finish_rl(spec):
    artifacts, output = Path(spec['artifacts']), Path(spec['output'])
    for step in SNAPSHOTS:
        checkpoint = output / f'checkpoint-{step}'
        evaluation.model_files(checkpoint, step)
        if not (checkpoint / 'reference_model.pt').is_file():
            raise ValueError(f'Missing RL reference checkpoint: {checkpoint}')
    state = read_json(output / 'checkpoint-350/trainer_state.json')
    schedule = read_json(artifacts / 'schedule.json')
    if state['global_step'] != 350 or state['max_steps'] != schedule['trainer_max_steps']:
        raise ValueError('Saved final step/full scheduler changed')
    if schedule['stop_after_steps'] != 350:
        raise ValueError('Every experiment must stop350 updates')
    write_json(artifacts / 'complete.json', dict(final_step=350, parent_model=spec['model'],
        final_checkpoint=str(output / 'checkpoint-350'), schedule=schedule,
        checkpoint_files={str((output / f'checkpoint-{s}' / name).relative_to(output)):
                          sha256(output / f'checkpoint-{s}' / name) for s in SNAPSHOTS
                          for name in evaluation.model_files(output / f'checkpoint-{s}', s) + ['reference_model.pt']},
        selection='No reward-based model selection; compare saved checkpoints on validation'))


def recover_sft_export(spec):
    """Recover a completed SFT fit whose final export was interrupted, without training."""
    artifacts, output = Path(spec['artifacts']), Path(spec['output'])
    record = read_json(artifacts / 'fit_complete.json')
    best = Path(record['best_checkpoint'])
    if best.parent.resolve() != output.resolve():
        raise ValueError('SFT best checkpoint is outside this experiment')
    verify_files(best, record['model_sha256'])
    selected = output / 'selected_model'
    selected.mkdir(exist_ok=True)
    for name in record['model_sha256']:
        shutil.copyfile(best / name, selected / name)
    verify_files(selected, record['model_sha256'])
    write_json(artifacts / 'complete.json', dict(record, model=str(selected),
        model_sha256=file_manifest(selected), selection='minimum validation loss, patience3'))


def summarize(root, config):
    rows = []
    for variant in config['variants']:
        baseline = root / f'baselines/history{variant["history_limit"]}/valid.metrics.json'
        if not baseline.exists():
            continue
        base = read_json(baseline)
        for step in SNAPSHOTS:
            target = root / 'experiments' / variant['label'] / 'validation' / f'checkpoint-{step}'
            if not (target / 'complete.json').exists():
                continue
            score = read_json(target / 'valid.metrics.json')
            candidates = 4 * variant['micro_batch'] * variant['accumulation']
            row = dict(experiment=variant['label'], step=step, history=variant['history_limit'],
                learning_rate=variant['learning_rate'], beta=variant['beta'], generations=variant['generations'],
                task_mix=variant['task_mix'], candidates_per_update=candidates,
                prompt_groups_seen=step * candidates // variant['generations'],
                candidates_seen=step * candidates, sft_HR10=base['HR@10'], sft_NDCG10=base['NDCG@10'])
            row.update({k: score[k] for k in ('samples', 'HR@5', 'HR@10', 'NDCG@5', 'NDCG@10')})
            row.update(delta_HR10=score['HR@10'] - base['HR@10'], delta_NDCG10=score['NDCG@10'] - base['NDCG@10'])
            rows.append(row)
    write_json(root / 'summary.json', rows)
    if rows:
        with (root / 'summary.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    columns = ['experiment', 'step', 'prompt_groups_seen', 'HR@10', 'NDCG@10', 'delta_HR10', 'delta_NDCG10']
    lines = ['Six Ranking experiments: all RL350 optimizer updates, validation only.', '',
             '01-05 use shared history50 SFT;06 uses independently trained history10 SFT.',
             '03 sees2800 prompt groups at350 vs22400 for other runs;05 uses twice as many candidates.',
             'No equal-exposure claim; no automatic test evaluation or model promotion.', '',
             '| ' + ' | '.join(columns) + ' |', '|' + '|'.join(['---'] * len(columns)) + '|']
    for row in rows:
        lines.append('| ' + ' | '.join(f'{row[k]:.6f}' if isinstance(row[k], float) else str(row[k]) for k in columns) + ' |')
    (root / 'summary.md').write_text('\n'.join(lines) + '\n')


def run(repo, args, launcher=launch):
    root, config, source_config = settings(repo, args)
    selected = [config['variants'][i - 1] for i in sorted(set(args.experiments))]
    if args.dry_run:
        for variant in selected:
            candidates = 4 * variant['micro_batch'] * variant['accumulation']
            print(variant, 'RLupdates=350', 'prompt_groups=', 350 * candidates // variant['generations'])
        print('06: original base -> history10 SFT(full10epochs/earlystop) -> own SFT valid -> RL350 -> valid50/100/175/350')
        print('No files written, weights read or GPU work launched.')
        return
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.six.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('This six-experiment suite is already running') from exc
        setup(repo, root, config, source_config)
        command = [sys.executable, str(repo / 'scripts/run_rl_six.py'), '--run-name', config['run_name'],
            '--source-run', config['source_run'], '--dataset', config['dataset'], '--gpus', config['gpus'],
            '--checkpoint-root', str(Path(config['checkpoint_root']).parent), '--base-model', config['base_model'],
            '--sft-micro-batch', str(config['sft_micro_batch']), '--eval-batch-size', str(config['eval_batch_size'])]
        (root / 'resume.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd ' + shlex.quote(str(repo)) + '\n' + shlex.join(command) + '\n')
        if args.stage == 'prepare':
            summarize(root, config)
            return
        env = environment(repo, config)
        if not (root / 'environment.json').exists():
            probe = ('import json,torch,transformers,trl,accelerate,deepspeed,bitsandbytes; '
                     'from pathlib import Path; '
                     'assert torch.cuda.is_available() and torch.cuda.device_count()==4; '
                     'd={x.__name__:x.__version__ for x in (torch,transformers,trl,accelerate,deepspeed,bitsandbytes)}; '
                     'd["gpus"]=[torch.cuda.get_device_name(i) for i in range(4)]; '
                     'Path(__import__("sys").argv[1]).write_text(json.dumps(d,indent=2))')
            launcher([sys.executable, '-c', probe, str(root / 'environment.json')], env, repo, root, root / 'environment.tail.log')
        for variant in selected:
            print(f'[six] {variant["label"]}: {args.stage}', flush=True)
            history = variant['history_limit']
            shared = root / 'shared' / f'history{history}'
            data, lengths = shared / 'data', shared / 'lengths.json'
            weights = Path(config['checkpoint_root'])
            model = weights / 'shared_sft'
            experiment = root / 'experiments' / variant['label']
            experiment.mkdir(parents=True, exist_ok=True)
            try:
                if history == 10:
                    # Derived measured lengths are mutable outputs, outside pinned shared inputs.
                    lengths = experiment / 'lengths.json'
                    sft_spec = dict(model=config['base_model'], data=str(data), category=config['dataset'],
                        artifacts=str(experiment / 'sft'), output=str(weights / variant['label'] / 'sft'),
                        lengths=str(lengths), sft_micro_batch=config['sft_micro_batch'])
                    sft_spec_file = experiment / 'sft_spec.json'
                    pin_json(sft_spec_file, sft_spec)
                    if not lengths.exists():
                        if args.stage == 'eval':
                            raise ValueError('Run history10 preflight/SFT before --stage eval')
                        launcher(worker_command(repo, sft_spec_file, 'preflight'), env, repo, root, experiment / 'preflight.tail.log')
                    pin_json(experiment / 'lengths_sha256.json', {'sha256': sha256(lengths)})
                    sft_done = experiment / 'sft/complete.json'
                    if not sft_done.exists() and (experiment / 'sft/fit_complete.json').exists():
                        recover_sft_export(sft_spec)
                    if not sft_done.exists():
                        if args.stage == 'eval':
                            raise ValueError('Complete history10 SFT before evaluation')
                        resume = latest_checkpoint(Path(sft_spec['output']))
                        if Path(sft_spec['output']).exists() and any(Path(sft_spec['output']).iterdir()) and not resume:
                            raise ValueError('Nonempty SFT output without resumable checkpoint')
                        launcher(worker_command(repo, sft_spec_file, 'sft', resume), env, repo, root, experiment / 'sft/train.tail.log')
                    record = read_json(sft_done)
                    model = Path(sft_spec['output']) / 'selected_model'
                    if record['model'] != str(model):
                        raise ValueError('history10 SFT export path changed')
                    verify_files(model, record['model_sha256'])
                if args.stage in ('all', 'eval'):
                    evaluation_job(repo, root, config, model, data, lengths,
                        root / f'baselines/history{history}', env, launcher)
                spec = dict(model=str(model), data=str(data), category=config['dataset'],
                    artifacts=str(experiment / 'rl'), output=str(weights / variant['label'] / 'rl'),
                    lengths=str(lengths), variant=variant)
                spec_file = experiment / 'rl_spec.json'
                pin_json(spec_file, spec)
                artifacts, output = Path(spec['artifacts']), Path(spec['output'])
                artifacts.mkdir(exist_ok=True)
                done = artifacts / 'complete.json'
                if done.exists():
                    record = read_json(done)
                    if record['final_step'] != 350 or record['parent_model'] != str(model):
                        raise ValueError('Completed RL provenance changed')
                    verify_files(output, record['checkpoint_files'])
                elif args.stage in ('all', 'train'):
                    resume = latest_checkpoint(output, require_reference=True)
                    step = int(Path(resume).name.split('-')[-1]) if resume else 0
                    if step > 350:
                        raise ValueError('Existing RL checkpoint exceeds350')
                    if step != 350:
                        if output.exists() and any(output.iterdir()) and not resume:
                            raise ValueError('Nonempty RL directory without resumable checkpoint')
                        launcher(worker_command(repo, spec_file, 'rl', resume), env, repo, root, artifacts / 'train.tail.log')
                    finish_rl(spec)
                else:
                    raise ValueError(f'Complete {variant["label"]} RL before --stage eval')
                if args.stage in ('all', 'eval'):
                    for step in SNAPSHOTS:
                        evaluation_job(repo, root, config, output / f'checkpoint-{step}', data, lengths,
                            experiment / 'validation' / f'checkpoint-{step}', env, launcher, step)
            finally:
                summarize(root, config)
        print(f'[six] Finished {args.stage}: {root / "summary.md"}', flush=True)


if __name__ == '__main__':
    run(REPO, parser().parse_args())
