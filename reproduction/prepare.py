"""Recover chronological histories from the shipped, globally ordered sliding windows.

No sorting by SID/item, graph inference, future-row lookup or target deduplication.
The official converter emits contiguous per-user windows (width 10), globally sorted
by target timestamp, then slices 80/10/10. We validate that exact export contract.
"""
import ast
import csv
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path

CATEGORIES = ('Office_Products', 'Industrial_and_Scientific')
SUFFIX = '_5_2016-10-2018-11'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def recover(splits, indices, items, history_limit=50):
    if history_limit != 50:
        raise ValueError('This reproduction requires history_limit=50')
    sid_map = {str(k): ''.join(v) for k, v in indices.items()}
    if set(sid_map) - set(items):
        raise ValueError('Missing item metadata')
    sequences, output, stats = {}, {}, {}
    for split in ('train', 'valid', 'test'):
        output[split] = []
        lengths, extended = Counter(), 0
        for row_id, row in enumerate(splits[split]):
            where = f'{split}:{row_id} user={row["user_id"]}'
            try:
                hist = ast.literal_eval(row['history_item_id'])
                sids = ast.literal_eval(row['history_item_sid'])
                titles = ast.literal_eval(row['history_item_title'])
                if not all(isinstance(x, list) for x in (hist, sids, titles)):
                    raise ValueError('History fields must be lists')
                if not (len(hist) == len(sids) == len(titles)) or not 1 <= len(hist) <= 10:
                    raise ValueError('Invalid original width or unaligned histories')
                target = int(row['item_id'])
                for item, sid, title in zip(hist + [target], sids + [row['item_sid']], titles + [row['item_title']]):
                    if sid_map[str(item)] != sid or items[str(item)]['title'] != title:
                        raise ValueError(f'item/SID/title mismatch: {item}')
                user = row['user_id']
                if user not in sequences:
                    if len(hist) != 1:
                        raise ValueError('Missing beginning of user trajectory; cannot reliably recover')
                    sequences[user] = list(hist)
                past = sequences[user]
                if len(hist) != min(10, len(past)) or hist != past[-10:]:
                    raise ValueError('Missing, reordered or inconsistent sliding window; cannot reliably recover')
                recovered = past[-history_limit:]
                new = dict(row)
                new.update(history_item_id=repr(recovered),
                           history_item_sid=repr([sid_map[str(i)] for i in recovered]),
                           history_item_title=repr([items[str(i)]['title'] for i in recovered]),
                           sample_id=f'{split}:{row_id}', target_position=str(len(past)))
                output[split].append(new)
                lengths[len(recovered)] += 1
                extended += len(recovered) > len(hist)
                # Only AFTER emitting the row does its target become history.
                past.append(target)
            except (ValueError, KeyError, TypeError, SyntaxError) as exc:
                raise ValueError(f'{where}: {exc}') from exc
        stats[split] = {'samples': len(output[split]), 'extended_samples': extended,
                        'history_lengths': dict(sorted(lengths.items()))}
    stats['catalog_items'] = len(sid_map)
    stats['unique_sids'] = len(set(sid_map.values()))
    stats['sid_collision_excess_items'] = len(sid_map) - len(set(sid_map.values()))
    stats['users'] = len(sequences)
    stats['max_interactions'] = max(map(len, sequences.values()))
    return output, stats


def prepare(source, destination, category):
    source, destination = Path(source), Path(destination)
    if category not in CATEGORIES:
        raise ValueError(category)
    inputs = {sp: source / sp / f'{category}{SUFFIX}.csv' for sp in ('train', 'valid', 'test')}
    inputs.update(index=source / 'index' / f'{category}.index.json',
                  items=source / 'index' / f'{category}.item.json',
                  info=source / 'info' / f'{category}{SUFFIX}.txt')
    hashes = {k: sha256(v) for k, v in inputs.items()}
    if destination.exists():
        report = json.loads((destination / 'audit.json').read_text())
        if report['source_sha256'] != hashes:
            raise ValueError('Source changed; use a new run name')
        for name, digest in report['generated_sha256'].items():
            if sha256(destination / name) != digest:
                raise ValueError(f'Generated data changed: {name}')
        return report
    splits = {}
    for sp in ('train', 'valid', 'test'):
        with inputs[sp].open(newline='') as f:
            splits[sp] = list(csv.DictReader(f))
    indices, items = (json.loads(inputs[k].read_text()) for k in ('index', 'items'))
    output, stats = recover(splits, indices, items)
    catalog = {''.join(s) for s in indices.values()}
    with inputs['info'].open() as f:
        info_sids = [line.split('\t')[0] for line in f]
    if len(info_sids) != len(indices) or set(info_sids) != catalog:
        raise ValueError('Info catalog and SID index disagree')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix='.prepare-', dir=destination.parent))
    try:
        for sp, rows in output.items():
            with (temp / f'{sp}.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        for key, name in [('index', f'{category}.index.json'), ('items', f'{category}.item.json'), ('info', 'info.txt')]:
            shutil.copyfile(inputs[key], temp / name)
        report = {'category': category, 'history_limit': 50, 'stats': stats,
                  'source_sha256': hashes, 'source_paths': {k: str(v.resolve()) for k, v in inputs.items()},
                  'ordering_contract': 'official timestamp-sorted export; train, valid, test in original row order',
                  'generated_sha256': {p.name: sha256(p) for p in temp.iterdir()}}
        write_json(temp / 'audit.json', report)
        temp.rename(destination)
    except BaseException:
        shutil.rmtree(temp)
        raise
    return report
