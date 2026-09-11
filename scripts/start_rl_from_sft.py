#!/usr/bin/env python3
"""Create a separate run from completed SFT, then train/evaluate fresh RL.

Copies SFT artifacts, prepared data and selected model as ordinary files. Never
copies or resumes source RL checkpoints. The source run is not modified.
"""
import argparse
import datetime
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reproduction.prepare import CATEGORIES, prepare, sha256, write_json
from reproduction.run import summarize
from scripts.migrate_rl_config_fix import BEFORE, AFTER, digest


GENERATION_FLAG = b'                        use_model_defaults=False,\n'
EVAL_HELPER = b'''def generate_candidates(model, inputs, conf, processors):
    # Qwen's saved defaults include do_sample=True and temperature=0.6. Without
    # this flag, HF >=4.50 overwrites even explicit default-valued config fields.
    return model.generate(**inputs, generation_config=conf,
                          logits_processor=processors, use_model_defaults=False)


'''
EVAL_OLD = b'''            generated = model.generate(**inputs, generation_config=conf,
                logits_processor=LogitsProcessorList([trie.processor(width)]))'''
EVAL_NEW = b'''            generated = generate_candidates(model, inputs, conf,
                LogitsProcessorList([trie.processor(width)]))'''


def check_source(repo, source):
    """Allow only reviewed dtype/generation fixes when importing completed SFT."""
    files = list((repo / 'reproduction').glob('*.py')) + [repo / name for name in (
        'data.py', 'minionerec_trainer.py', 'requirements-reproduction.txt',
        'config/zero2_opt.yaml', 'scripts/reproduce.sh')]
    current = {str(path.relative_to(repo)): path.read_bytes() for path in files}
    recorded = json.loads((source / 'source_sha256.json').read_text())
    if current.keys() != recorded.keys():
        raise ValueError('Source file list changed beyond supported SFT import fixes')
    changes = []
    for name, content in current.items():
        if sha256(source / 'source' / name) != recorded[name]:
            raise ValueError(f'Original source snapshot changed: {name}')
        if digest(content) == recorded[name]:
            continue
        variants = []
        if name == 'minionerec_trainer.py':
            variants = [content.replace(AFTER, BEFORE, 1)]
            if content.count(GENERATION_FLAG) == 1:
                old_generation = content.replace(GENERATION_FLAG, b'', 1)
                variants.extend([old_generation, old_generation.replace(AFTER, BEFORE, 1)])
        elif name == 'reproduction/evaluate.py':
            if content.count(EVAL_HELPER) == 1 and content.count(EVAL_NEW) == 1:
                variants = [content.replace(EVAL_HELPER, b'', 1).replace(EVAL_NEW, EVAL_OLD, 1)]
        if not any(digest(variant) == recorded[name] for variant in variants):
            raise ValueError(f'Unrelated source change refused: {name}')
        changes.append(name)
    return current, changes


def copy_files(source, target):
    """Keep both model and portable results independent of the source directory."""
    if source.is_symlink() or any(path.is_symlink() for path in source.rglob('*')):
        raise ValueError(f'Expected ordinary files, found a symlink under {source}')
    shutil.copytree(source, target)


def launch_command(repo, config, category):
    return [
        sys.executable, '-m', 'reproduction.run', '--run-name', config['run_name'],
        '--dataset', category, '--gpus', config['gpus'], '--model', config['model'],
        '--checkpoint-root', str(Path(config['checkpoint_root']).parent),
        '--sft-micro-batch', str(config['sft_micro_batch']),
        '--rl-micro-batch', str(config['rl_micro_batch']),
        '--eval-batch-size', str(config['eval_batch_size']),
        '--max-steps', str(config['max_steps']), '--resume']


def create_run(repo, source_run, run_name, category, micro_batch=32, *, short_config=None):
    repo = Path(repo).resolve()
    for name in (source_run, run_name):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name):
            raise ValueError('Run names must be simple directory names')
    if source_run == run_name:
        raise ValueError('Use a distinct run name to retain the previous RL experiment')
    if category not in CATEGORIES:
        raise ValueError(f'Unsupported category: {category}')
    if micro_batch < 16 or micro_batch % 16 or 256 % micro_batch:
        raise ValueError('RL micro batch must be a multiple of 16 dividing 256')
    source = repo / 'results' / source_run
    root = repo / 'results' / run_name
    config = json.loads((source / 'run_config.json').read_text())
    if config['world_size'] != 4 or config['rl_generations'] != 16 or config['effective_batch'] != 1024:
        raise ValueError('Expected the four-GPU G16/effective1024 reproduction recipe')
    source_checkpoint_root = Path(config['checkpoint_root'])
    checkpoint_root = (source_checkpoint_root.parent / run_name).resolve()
    if root.exists() or checkpoint_root.exists():
        raise ValueError('Destination already exists; use its resume command or choose a new run name')
    if (checkpoint_root == root or root in checkpoint_root.parents
            or checkpoint_root == source or source in checkpoint_root.parents):
        raise ValueError('Checkpoint root must be outside the results directories')
    cat = source / category
    required_artifacts = ['sft.complete.json', 'eval-sft.complete.json', 'training.json',
                          'valid.metrics.json', 'valid.predictions.jsonl']
    if short_config is None:
        required_artifacts += ['test.metrics.json', 'test.predictions.jsonl']
    for name in required_artifacts:
        if not (cat / 'sft' / name).is_file():
            raise ValueError(f'Completed SFT and evaluation required: {cat / "sft" / name}')
    training = json.loads((cat / 'sft/training.json').read_text())
    model = source_checkpoint_root / category / 'sft/selected_model'
    imported = source / 'sft_import/record.json'
    record = json.loads(imported.read_text()) if imported.is_file() else {}
    if record.get('category') == category:
        # Imported SFT training.json intentionally retains its historical path.
        # Validate the independent local copy without requiring that path to exist.
        if (Path(record['copied_model']).resolve() != model.resolve()
                or record['selected_step'] != training['selected_step']):
            raise ValueError('Imported SFT model/step disagrees with provenance')
        actual = {str(path.relative_to(model)): sha256(path)
                  for path in model.rglob('*') if path.is_file()}
        if not actual or actual != record['model_sha256']:
            raise ValueError('Imported SFT model hashes disagree with provenance')
    elif Path(training['final_checkpoint']).resolve() != model.resolve():
        raise ValueError('SFT selected-model path disagrees with training.json')
    required = ('config.json', 'tokenizer_config.json', 'tokenizer.json')
    if any(not (model / name).is_file() for name in required):
        raise ValueError(f'SFT model/tokenizer files unavailable: {model}; run on the training server')
    weight_files = list(model.glob('*.safetensors')) + list(model.glob('pytorch_model*.bin'))
    if not weight_files:
        raise ValueError(f'Missing SFT model weights: {model}')
    for index in model.glob('*.index.json'):
        for shard in json.loads(index.read_text()).get('weight_map', {}).values():
            if not (model / shard).is_file():
                raise ValueError(f'Missing SFT model shard: {shard}')
    # Read-only checks: generated CSV integrity and exact code/fix compatibility.
    if not (cat / 'data/audit.json').is_file():
        raise ValueError('Missing prepared data audit')
    prepare(repo / 'data/Amazon', cat / 'data', category)
    lengths = json.loads((cat / 'lengths.json').read_text())
    if lengths['model'] != config['model'] or lengths['data_audit_sha256'] != sha256(cat / 'data/audit.json'):
        raise ValueError('Saved tokenizer preflight does not match data/model')
    current_source, changed_source = check_source(repo, source)
    # Older evaluator may have inherited Qwen's sampling defaults even though its
    # metrics JSON claimed deterministic decoding. Always rerun that evaluation.
    reevaluate = 'reproduction/evaluate.py' in changed_source
    config = dict(config, run_name=run_name, checkpoint_root=str(checkpoint_root),
                  rl_micro_batch=micro_batch, rl_accumulation=256 // micro_batch)
    if short_config is not None:
        # Extra key deliberately makes the normal all-stage runner reject this run.
        config['rl_short'] = short_config
        config['gpus'] = short_config['gpus']
    checkpoint_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.sft-import-', dir=root.parent))
    checkpoint_staging = Path(tempfile.mkdtemp(prefix='.sft-import-', dir=checkpoint_root.parent))
    try:
        dest_cat = staging / category
        dest_cat.mkdir()
        for name in ('data', 'sft', 'tokenizer'):
            if (cat / name).exists():
                copy_files(cat / name, dest_cat / name)
        for name in ('lengths.json', 'preflight.log'):
            if (cat / name).is_file():
                shutil.copyfile(cat / name, dest_cat / name)
        selected = checkpoint_staging / category / 'sft/selected_model'
        selected.parent.mkdir(parents=True)
        print(f'Copying selected SFT model from step {training["selected_step"]}: {model}', flush=True)
        copy_files(model, selected)
        model_hashes = {str(path.relative_to(selected)): sha256(path)
                        for path in selected.rglob('*') if path.is_file()}
        for name, expected in model_hashes.items():
            if sha256(model / name) != expected:
                raise ValueError(f'SFT model copy mismatch: {name}')
        copy_files(source / 'source', staging / 'source')
        shutil.copyfile(source / 'source_sha256.json', staging / 'source_sha256.json')
        if (source / 'source_migrations').is_dir():
            copy_files(source / 'source_migrations', staging / 'source_migrations')
        origin = staging / 'sft_import'
        origin.mkdir()
        if (source / 'sft_import').is_dir():
            copy_files(source / 'sft_import', origin / 'previous_import')
        shutil.copyfile(source / 'run_config.json', origin / 'source_run_config.json')
        shutil.copyfile(source / 'source_sha256.json', origin / 'source_sha256.json')
        copy_files(source / 'source', origin / 'source')
        for name in ('environment.log', 'commands.jsonl', 'invocations.jsonl'):
            if (source / name).is_file():
                shutil.copyfile(source / name, origin / name)
        # Original SFT records retain their actual historical training paths/args.
        write_json(origin / 'record.json', {
            'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'source_run': str(source), 'category': category,
            'source_model': str(model), 'copied_model': str(checkpoint_root / category / 'sft/selected_model'),
            'selected_step': training['selected_step'], 'model_sha256': model_hashes,
            'source_fixes': changed_source, 'sft_requires_reevaluation': reevaluate,
            'rl_initialization': 'fresh optimizer, scheduler, reference and step 0 from selected SFT model',
            'copied_artifacts_sha256': {str(path.relative_to(dest_cat)): sha256(path)
                                       for path in dest_cat.rglob('*') if path.is_file()},
        })
        shutil.copyfile(Path(__file__), origin / 'start_rl_from_sft.py')
        shutil.copyfile(Path(__file__).with_name('migrate_rl_config_fix.py'), origin / 'migrate_rl_config_fix.py')
        write_json(staging / 'run_config.json', config)
        for name, content in current_source.items():
            (staging / 'source' / name).write_bytes(content)
        write_json(staging / 'source_sha256.json', {name: digest(content) for name, content in current_source.items()})
        if reevaluate:
            archived_eval = origin / 'superseded_sft_evaluation'
            archived_eval.mkdir()
            for path in list((dest_cat / 'sft').iterdir()):
                if (path.name == 'eval-sft.complete.json' or path.name.startswith(('eval-', 'valid.', 'test.'))):
                    path.rename(archived_eval / path.name)
            write_json(archived_eval / 'reason.json', {
                'reason': 'Old HF generation config inherited model sampling defaults; rerun deterministic beam50',
                'use_for_sft_rl_comparison': False})
            print('Old SFT evaluation archived; rerunning deterministic beam50 before RL. SFT weights are retained.')
        if short_config is None:
            summarize(staging)
            command = launch_command(repo, config, category)
        else:
            # Retain imported test results only as historical provenance. A short
            # validation experiment must not present them as its own evaluation.
            historical = origin / 'historical_sft_test'
            for path in list((dest_cat / 'sft').iterdir()):
                if path.name.startswith('test.') or path.name == 'eval-test.log':
                    historical.mkdir(exist_ok=True)
                    path.rename(historical / path.name)
            command = [sys.executable, str(repo / 'scripts/run_rl_short.py'),
                       '--run-name', run_name, '--resume']
            (staging / 'summary.md').write_text('RL short experiment: validation only; no results yet.\n')
            names = ['scripts/run_rl_short.py', 'scripts/train_rl_short.py',
                     'scripts/start_rl_from_sft.py', 'scripts/migrate_rl_config_fix.py',
                     'scripts/evaluate_checkpoints.py', 'scripts/evaluate_checkpoints.sh']
            for name in names:
                target = staging / 'short_source' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(repo / name, target)
            write_json(staging / 'short_source_sha256.json', {name: sha256(repo / name) for name in names})
        (staging / 'resume.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
            + 'cd ' + shlex.quote(str(repo)) + '\n' + shlex.join(command) + '\n')
        checkpoint_staging.rename(checkpoint_root)
        try:
            staging.rename(root)
        except BaseException:
            checkpoint_root.rename(checkpoint_staging)
            raise
    except BaseException:
        shutil.rmtree(staging)
        shutil.rmtree(checkpoint_staging)
        raise
    print(f'Prepared {root}; only SFT is imported. New RL micro={micro_batch}, accumulation={256 // micro_batch}.')
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', required=True)
    parser.add_argument('--run-name', required=True)
    parser.add_argument('--dataset', choices=CATEGORIES, default='Office_Products')
    parser.add_argument('--rl-micro-batch', type=int, default=32)
    parser.add_argument('--prepare-only', action='store_true', help='copy/import artifacts without starting training')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    command = create_run(repo, args.source_run, args.run_name, args.dataset, args.rl_micro_batch)
    print(shlex.join(command), flush=True)
    if not args.prepare_only:
        subprocess.run(command, cwd=repo, check=True)


if __name__ == '__main__':
    main()
