"""Contracts for the five-run follow-up; existing six-run inputs stay immutable."""
import ast
from collections import Counter
import csv
import functools
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import types

from reproduction.prepare import CATEGORIES, SUFFIX, sha256, write_json
from scripts.six_utils import file_manifest, read_json, verify_files

SNAPSHOTS = [50, 100, 175, 350]
UPSTREAM = Path(__file__).resolve().parents[1] / 'reproduction/upstream_minionerec'


def variants():
    base = dict(learning_rate=1e-5, beta=.04, generations=16, micro_batch=16,
                accumulation=16, history_limit=10, task_mix='original',
                optimizer='paged_adamw_32bit', do_sample=False,
                category=CATEGORIES[0], family='qwen3', recipe='current')
    changes = [('01_history10_restart', {}),
               ('02_history10_adamw', dict(optimizer='adamw_torch')),
               ('03_official_qwen3_office', dict(recipe='official', beta=.001, do_sample=True)),
               ('04_official_qwen25_office', dict(recipe='official', family='qwen2', beta=.001, do_sample=True)),
               ('05_official_qwen25_industrial', dict(recipe='official', family='qwen2', beta=.001,
                                                      do_sample=True, category=CATEGORIES[1]))]
    return [{**base, **change, 'label': label} for label, change in changes]


def schedule_spec(rows):
    horizon = math.ceil(rows / 64) * 2
    if horizon < 350:
        raise ValueError('Full two-epoch schedule must accommodate 350 updates')
    return dict(full_scheduler_steps=horizon, warmup_steps=math.ceil(horizon * .03),
                ref_model_sync_steps=512, eval_steps=math.ceil(horizon * .0999),
                candidates_per_update=1024, prompt_groups_per_update=64,
                stop_after_steps=350, snapshot_steps=SNAPSHOTS,
                prompt_group_budget=22400, candidate_budget=358400,
                schedule_basis='full two-epoch horizon, not compressed to 350 updates')


@functools.lru_cache(maxsize=1)
def official_data_module():
    """Execute pinned upstream data code with only literal parsing/empty-token guards.

    Official dictionary deduplication, Fusion SID lookup, prompts and task sampling
    remain intact. Do not silently substitute our item-preserving data classes.
    """
    manifest = read_json(UPSTREAM / 'manifest.json')
    verify_files(UPSTREAM, manifest['files'])
    source = (UPSTREAM / 'data.py.txt').read_text()
    source = source.replace('while t[0] ==', 'while t and t[0] ==')
    source = source.replace('while t[-1] ==', 'while t and t[-1] ==')
    tree = ast.parse(source)

    class LiteralParser(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == 'eval':
                node.func = ast.Attribute(ast.Name('_safe_ast', ast.Load()), 'literal_eval', ast.Load())
            return node

    module = types.ModuleType('_five_official_data')
    module.__dict__['_safe_ast'] = ast
    exec(compile(ast.fix_missing_locations(LiteralParser().visit(tree)), str(UPSTREAM / 'data.py.txt'), 'exec'), module.__dict__)
    return module


def sft_data(root, category, tokenizer, max_len, recipe):
    if recipe == 'current':
        from reproduction.datasets import sft_data as current
        return current(root, category, tokenizer, max_len)
    if recipe != 'official':
        raise ValueError('Unknown recipe')
    module = official_data_module()
    root = Path(root)
    common = dict(tokenizer=tokenizer, max_len=max_len, seed=42)
    meta = dict(item_file=str(root / f'{category}.item.json'), index_file=str(root / f'{category}.index.json'))
    return [module.SidSFTDataset(str(root / 'train.csv'), **common),
            module.SidItemFeatDataset(**meta, **common),
            module.FusionSeqRecDataset(str(root / 'train.csv'), **meta, **common)], \
        module.SidSFTDataset(str(root / 'valid.csv'), **common)


def rl_data(root, category, recipe):
    if recipe == 'current':
        from reproduction.datasets import rl_data as current
        return current(root, category)
    if recipe != 'official':
        raise ValueError('Unknown recipe')
    module = official_data_module()
    root = Path(root)
    train = [module.SidDataset(str(root / 'train.csv')),
             module.RLTitle2SidDataset(str(root / f'{category}.item.json'), str(root / f'{category}.index.json')),
             module.RLSeqTitle2SidDataset(str(root / 'train.csv'), sample=10000)]
    valid = [module.SidDataset(str(root / 'valid.csv'))]

    def rows(datasets, split):
        # Raw official RL prompts intentionally have no added SFT instruction.
        # Row-local targets avoid upstream train/validation lookup collisions.
        return [dict(prompt=row['prompt'], target=row['completion'], sample_id=f'{split}:{task}:{i}')
                for task, dataset in enumerate(datasets) for i, row in enumerate(dataset)]
    return rows(train, 'train'), rows(valid, 'valid')


def prepare_official(source, destination, category):
    """Preserve original train/valid history10 rows; add audit identifiers only."""
    source, destination = Path(source), Path(destination)
    inputs = {split: source / split / f'{category}{SUFFIX}.csv' for split in ('train', 'valid')}
    inputs.update(index=source / 'index' / f'{category}.index.json',
                  items=source / 'index' / f'{category}.item.json',
                  info=source / 'info' / f'{category}{SUFFIX}.txt')
    digests = {key: sha256(path) for key, path in inputs.items()}
    if destination.exists():
        audit = read_json(destination / 'audit.json')
        if audit['source_sha256'] != digests:
            raise ValueError('Official source data changed')
        verify_files(destination, audit['generated_sha256'])
        return audit
    indices, items = read_json(inputs['index']), read_json(inputs['items'])
    sid_map = {str(k): ''.join(v) for k, v in indices.items()}
    if set(sid_map) - set(items):
        raise ValueError('Missing catalog metadata')
    info = [line.split('\t')[0] for line in inputs['info'].read_text().splitlines()]
    if len(info) != len(indices) or set(info) != set(sid_map.values()):
        raise ValueError('Official catalog mismatch')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.official-data-', dir=destination.parent))
    trajectories, counts = {}, {}
    try:
        for split in ('train', 'valid'):
            with inputs[split].open(newline='') as src, (temporary / f'{split}.csv').open('w', newline='') as dst:
                reader = csv.DictReader(src)
                if {'sample_id', 'target_position'} & set(reader.fieldnames):
                    raise ValueError('Expected original upstream CSV, not derived data')
                writer = csv.DictWriter(dst, fieldnames=[*reader.fieldnames, 'sample_id', 'target_position'])
                writer.writeheader()
                count = 0
                for i, row in enumerate(reader):
                    ids, sids, titles = [ast.literal_eval(row[key]) for key in
                                        ('history_item_id', 'history_item_sid', 'history_item_title')]
                    if not 1 <= len(ids) <= 10 or not len(ids) == len(sids) == len(titles):
                        raise ValueError('Original histories must be aligned and <=10')
                    for item, sid, title in zip(ids + [int(row['item_id'])], sids + [row['item_sid']], titles + [row['item_title']]):
                        if sid_map[str(item)] != sid or items[str(item)]['title'] != title:
                            raise ValueError('Original item/SID/title mismatch')
                    past = trajectories.setdefault(row['user_id'], [])
                    if not past:
                        if len(ids) != 1:
                            raise ValueError('Missing initial user trajectory')
                        past.extend(ids)
                    if ids != past[-10:]:
                        raise ValueError('Original row order/trajectory mismatch')
                    writer.writerow(dict(row, sample_id=f'{split}:{i}', target_position=len(past)))
                    past.append(int(row['item_id']))
                    count += 1
                counts[split] = count
        for key, name in [('index', f'{category}.index.json'), ('items', f'{category}.item.json'), ('info', 'info.txt')]:
            shutil.copyfile(inputs[key], temporary / name)
        audit = dict(category=category, history_limit=10, source_sha256=digests,
                     source_paths={k: str(p.resolve()) for k, p in inputs.items()}, rows=counts,
                     generated_sha256=file_manifest(temporary),
                     policy='original history10 values/order/splits; audit identifiers added; no test read')
        write_json(temporary / 'audit.json', audit)
        temporary.rename(destination)
        return audit
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def model_contract(config, family, expanded_vocab):
    """Check supported sub-1B architecture and exact expanded parameter count on meta."""
    expected = {'qwen3': (1024, 28), 'qwen2': (896, 24)}
    if family not in expected or config.model_type != family or \
            (config.hidden_size, config.num_hidden_layers) != expected[family]:
        raise ValueError('Expected Qwen3-0.6B or Qwen2.5-0.5B architecture')
    import copy
    import torch
    from transformers import AutoModelForCausalLM
    cfg = copy.deepcopy(config)
    cfg.vocab_size = expanded_vocab
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(cfg)
    count = sum(p.numel() for p in model.parameters())
    if count >= 1_000_000_000:
        raise ValueError(f'Expanded model must remain below 1B parameters, got {count}')
    return dict(family=family, parameters_after_sid_extension=count, limit_exclusive=1_000_000_000)


def row_digest(rows):
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode()).hexdigest()


def checkpoint_manifest(checkpoint):
    return {str(p.relative_to(checkpoint)): p.stat().st_size for p in Path(checkpoint).rglob('*')
            if p.is_file() and p.name != 'five_complete.json'}


def resumable_checkpoint(output, stage):
    """Ignore interrupted saves; never resume half-written optimizer/reference state."""
    output = Path(output)
    complete, partial = [], []
    for checkpoint in output.glob('checkpoint-*'):
        marker = checkpoint / 'five_complete.json'
        if not marker.exists():
            partial.append(checkpoint)
            continue
        record = read_json(marker)
        if record['stage'] != stage or record['files'] != checkpoint_manifest(checkpoint):
            raise ValueError(f'Completed checkpoint changed: {checkpoint}')
        if read_json(checkpoint / 'trainer_state.json')['global_step'] != record['step']:
            raise ValueError('Checkpoint step mismatch')
        if stage == 'rl' and not (checkpoint / 'reference_model.pt').is_file():
            raise ValueError('Missing saved RL reference')
        complete.append((record['step'], checkpoint))
    # Preserve partial saves outside the Trainer checkpoint naming pattern.
    for checkpoint in partial:
        target = output / ('.incomplete-' + checkpoint.name)
        counter = 0
        while target.exists():
            counter += 1
            target = output / f'.incomplete-{checkpoint.name}-{counter}'
        checkpoint.rename(target)
    return str(max(complete)[1]) if complete else None
