"""Pinned upstream sources and storage helpers for direct official RL700 runs."""
import difflib
import json
from pathlib import Path
import shutil
import tempfile

from reproduction.prepare import sha256, write_json
from scripts.six_utils import file_manifest, read_json, verify_files

SNAPSHOTS = [50, 100, 175, 350, 500, 550, 700]
CATEGORIES = ['Office_Products', 'Industrial_and_Scientific']
SNAPSHOT = Path(__file__).resolve().parents[1] / 'reproduction/official_direct'
SUFFIX = '_5_2016-10-2018-11'


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Upstream patch anchor changed: {old[:80]}')
    return source.replace(old, new, 1)


def patched(name, source):
    """Only compatibility, unused allocation, logging and export; no RL math changes."""
    if name == 'data.py':
        source = replace_once(source, 'while t[0] == self.bos_id:', 'while t and t[0] == self.bos_id:')
        source = replace_once(source, 'while t[-1] == self.eos_id:', 'while t and t[-1] == self.eos_id:')
    if name == 'minionerec_trainer.py':
        source = replace_once(source, 'self.max_prompt_length = args.max_prompt_length',
                              'self.max_prompt_length = getattr(args, "max_prompt_length", None)')
    if name == 'sft.py':
        source = replace_once(source, 'save_total_limit=1,', 'save_total_limit=2,')
        source = replace_once(source, 'report_to=None,',
                              'report_to="none", disable_tqdm=True, restore_callback_states_from_checkpoint=True,')
    if name == 'rl.py':
        source = replace_once(source,
            'llm_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")\n    device = llm_model.device',
            '# Ranking closures never use this auxiliary model; avoid allocating it on every GPU.\n'
            '    llm_model = None if reward_type == "ranking" else AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")\n'
            '    device = None if llm_model is None else llm_model.device')
        source = replace_once(source, 'report_to="wandb",', 'report_to="none", disable_tqdm=True,')
    if name in ('sft.py', 'rl.py'):
        # Export once through the storage wrapper, not all ranks saving duplicated root weights.
        start = source.index('    trainer.save_model(output_dir)')
        end = source.index('if __name__ == ', start)
        source = source[:start] + '    _direct_finish(trainer, tokenizer, output_dir)\n\n\n' + source[end:]
    if name == 'evaluate.py':
        source = replace_once(source, '    os.environ["CUDA_VISIBLE_DEVICES"] = "0"',
                              '    # Keep the single GPU selected for this evaluation worker.')
    return source


def materialize(destination):
    """Write a versioned runnable copy plus exact diff; snapshot itself stays immutable."""
    destination = Path(destination)
    manifest = read_json(SNAPSHOT / 'manifest.json')
    if destination.exists():
        record = read_json(destination / 'direct_manifest.json')
        if record['upstream'] != manifest:
            raise ValueError('Pinned upstream version changed')
        verify_files(destination, record['files'])
        for name, entry in manifest['files'].items():
            original = SNAPSHOT / entry['snapshot']
            if sha256(original) != entry['sha256']:
                raise ValueError(f'Upstream snapshot changed: {name}')
            if (destination / name).read_text() != patched(name, original.read_text()):
                raise ValueError(f'Runtime patch changed: {name}')
        return record
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix='.official-source-', dir=destination.parent))
    diff = []
    try:
        for name, entry in manifest['files'].items():
            original = SNAPSHOT / entry['snapshot']
            if sha256(original) != entry['sha256']:
                raise ValueError(f'Upstream snapshot changed: {name}')
            before = original.read_text(); after = patched(name, before)
            target = tmp / name; target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(after)
            diff.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                           fromfile='upstream/' + name, tofile='runtime/' + name))
        (tmp / 'runtime.patch').write_text(''.join(diff))
        record = dict(upstream=manifest, files=file_manifest(tmp))
        write_json(tmp / 'direct_manifest.json', record)
        tmp.rename(destination)
        return record
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)


def committed_checkpoint(output, stage):
    """Keep incomplete saves out of Trainer discovery; never silently restart RL."""
    complete = []
    output = Path(output)
    for checkpoint in output.glob('checkpoint-*'):
        marker = checkpoint / 'direct_complete.json'
        if not marker.exists():
            target = output / ('.incomplete-' + checkpoint.name)
            n = 0
            while target.exists():
                n += 1; target = output / f'.incomplete-{checkpoint.name}-{n}'
            checkpoint.rename(target)
            continue
        record = read_json(marker)
        sizes = {str(p.relative_to(checkpoint)): p.stat().st_size for p in checkpoint.rglob('*')
                 if p.is_file() and p.name != marker.name}
        if record['files'] != sizes or record['stage'] != stage:
            raise ValueError(f'Committed checkpoint changed: {checkpoint}')
        state = read_json(checkpoint / 'trainer_state.json')
        if state['global_step'] != record['step']:
            raise ValueError('Checkpoint step mismatch')
        if stage == 'rl' and not (checkpoint / 'direct_reference.pt').exists():
            raise ValueError('Missing saved reference model')
        complete.append((record['step'], checkpoint))
    return str(max(complete)[1]) if complete else None


def model_hashes(model):
    from scripts.evaluate_checkpoints import model_files
    return {name: sha256(Path(model) / name) for name in model_files(Path(model))}


def recover_export(spec):
    """Completed SFT fit is recoverable without another training update."""
    artifacts, output = Path(spec['artifacts']), Path(spec['output'])
    fit = read_json(artifacts / 'fit_complete.json')
    best = Path(fit['best_checkpoint'])
    if best.parent.resolve() != output.resolve():
        raise ValueError('Best checkpoint is outside this SFT run')
    verify_files(best, fit['model_sha256'])
    selected = output / 'selected_model'; selected.mkdir(exist_ok=True)
    for name in fit['model_sha256']:
        shutil.copyfile(best / name, selected / name)
    write_json(artifacts / 'complete.json', dict(fit, model=str(selected), model_sha256=model_hashes(selected)))
