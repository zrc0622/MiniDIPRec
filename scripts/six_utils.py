"""Lightweight, auditable data/artifact helpers for the six-run RL suite."""
import ast
import codecs
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import re
import signal
import subprocess
import time

from reproduction.prepare import sha256, write_json

SNAPSHOTS = [50, 100, 175, 350]


def variants():
    base = dict(learning_rate=1e-5, beta=.04, generations=16, micro_batch=16,
                accumulation=16, history_limit=50, task_mix='original')
    changes = [('01_lr5e6', dict(learning_rate=5e-6)),
               ('02_beta01', dict(beta=.1)),
               ('03_batch128', dict(accumulation=2)),
               ('04_equal_tasks', dict(task_mix='equal')),
               ('05_g32', dict(generations=32, micro_batch=32)),
               ('06_history10', dict(history_limit=10))]
    return [dict(base, label=label, **change) for label, change in changes]


def read_json(path):
    return json.loads(Path(path).read_text())


def pin_json(path, value):
    path = Path(path)
    if path.exists() and read_json(path) != value:
        raise ValueError(f'Pinned configuration/provenance changed: {path}')
    if not path.exists():
        write_json(path, value)


def file_manifest(root):
    root = Path(root)
    files = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Symlink is not a portable artifact: {path}')
        if path.is_file():
            files[str(path.relative_to(root))] = sha256(path)
    return files


def verify_files(root, manifest):
    for name, digest in manifest.items():
        if sha256(Path(root) / name) != digest:
            raise ValueError(f'Artifact changed: {Path(root) / name}')


def write_history10(source, destination):
    """Cap only synchronized input history fields; never move rows/labels across splits."""
    with Path(source).open(newline='') as src, Path(destination).open('w', newline='') as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        count = changed = 0
        for row in reader:
            histories = [ast.literal_eval(row[k]) for k in
                         ('history_item_id', 'history_item_sid', 'history_item_title')]
            if len({len(x) for x in histories}) != 1:
                raise ValueError('Misaligned item/SID/title histories')
            changed += len(histories[0]) > 10
            for key, values in zip(('history_item_id', 'history_item_sid', 'history_item_title'), histories):
                row[key] = repr(values[-10:])
            writer.writerow(row)
            count += 1
    return {'rows': count, 'truncated_rows': changed}


def balance_rows(rows, seed=42):
    """Equal task exposure, fixed total rows, cycle without replacement within each task."""
    buckets = [[] for _ in range(3)]
    for row in rows:
        buckets[int(row['sample_id'].split(':')[1])].append(row)
    if not all(buckets):
        raise ValueError('All three RL task pools must be nonempty')
    rng = random.Random(seed)
    output, desired = [], [len(rows) // 3 + (i < len(rows) % 3) for i in range(3)]
    for task, (pool, count) in enumerate(zip(buckets, desired)):
        selected = []
        while len(selected) < count:
            order = list(range(len(pool)))
            rng.shuffle(order)
            selected.extend(order[:count - len(selected)])
        output.extend(dict(pool[i], sample_id=f'train:{task}:draw{j}:{pool[i]["sample_id"]}')
                      for j, i in enumerate(selected))
    return output


def schedule_spec(train_rows, variant):
    """Keep C's cosine/ref cadence in prompt-exposure units; user stop is always350."""
    import math
    candidates = 4 * variant['micro_batch'] * variant['accumulation']
    groups = candidates // variant['generations']
    if candidates % variant['generations'] or groups not in (8, 64):
        raise ValueError('Unexpected candidate/group contract')
    scale = 64 // groups
    base_horizon = math.ceil(train_rows / 64) * 2
    return dict(candidates_per_update=candidates, prompt_groups_per_update=groups,
                full_scheduler_steps=base_horizon * scale,
                warmup_steps=math.ceil(base_horizon * .03) * scale,
                ref_model_sync_steps=512 * scale,
                eval_steps=math.ceil(base_horizon * .0999) * scale,
                stop_after_steps=350, snapshot_steps=SNAPSHOTS,
                prompt_group_budget=groups * 350,
                candidate_budget=candidates * 350,
                schedule_basis='C full two-epoch horizon scaled by input groups; stop350 updates, not equal exposure')


def launch(command, env, repo, root, log):
    """Stream useful lines; retain bounded console tail and warning/error evidence only."""
    log, root = Path(log), Path(root)
    log.parent.mkdir(parents=True, exist_ok=True)
    record = {'argv': command, 'cwd': str(repo), 'started_at': time.time(),
              'gpus': env.get('CUDA_VISIBLE_DEVICES')}
    with (root / 'commands.jsonl').open('a') as f:
        f.write(json.dumps(record) + '\n')
    print('Running: ' + ' '.join(command), flush=True)
    proc = subprocess.Popen(command, cwd=repo, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, start_new_session=True)
    tail = log.read_bytes()[-262144:] if log.exists() else b''
    warnings = log.with_suffix('.warnings.log')
    warning_tail = warnings.read_bytes()[-65536:] if warnings.exists() else b''
    decoder = codecs.getincrementaldecoder('utf-8')('replace')
    pending, saved = '', time.monotonic()
    try:
        while True:
            chunk = os.read(proc.stdout.fileno(), 8192)
            if not chunk:
                break
            tail = (tail + chunk)[-262144:]
            lines = re.split(r'[\r\n]', pending + decoder.decode(chunk))
            pending = lines.pop()[-8192:]
            for line in lines:
                important = re.search(r'warning|error|exception|traceback|out of memory|nan|inf\b', line, re.I)
                if important:
                    warning_tail = (warning_tail + line.encode() + b'\n')[-65536:]
                if important or line.startswith(('[six]', 'PASS', 'Traceback')):
                    print(line, flush=True)
            if time.monotonic() - saved > 1:
                log.write_bytes(tail)
                warnings.write_bytes(warning_tail)
                saved = time.monotonic()
        code = proc.wait()
    except BaseException:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        raise
    finally:
        proc.stdout.close()
        log.write_bytes(tail)
        warnings.write_bytes(warning_tail)
        record.update(finished_at=time.time(), exit_code=proc.poll())
        with (root / 'commands.jsonl').open('a') as f:
            f.write(json.dumps(record) + '\n')
    if code:
        raise RuntimeError(f'Command exited {code}; see {log}')


def compress_predictions(directory):
    """Lossless gzip, verified before removing the suite-owned uncompressed duplicate."""
    directory = Path(directory)
    source = directory / 'valid.predictions.jsonl'
    target = directory / 'valid.predictions.jsonl.gz'
    if not source.exists():
        return target
    expected = sha256(source)
    temporary = target.with_suffix('.tmp')
    with source.open('rb') as src, temporary.open('wb') as dst:
        with gzip.GzipFile(filename='', mode='wb', fileobj=dst, mtime=0) as compressed:
            for chunk in iter(lambda: src.read(1024 * 1024), b''):
                compressed.write(chunk)
    with gzip.open(temporary, 'rb') as f:
        actual = hashlib.file_digest(f, 'sha256').hexdigest()
    if actual != expected:
        raise ValueError('Compressed prediction integrity failure')
    temporary.replace(target)
    source.unlink()
    return target


def audit_compact(directory, data, category):
    """Same row/candidate/score checks as old evaluator, directly on lossless gzip."""
    from reproduction.evaluate import metrics
    import itertools
    import math
    report = read_json(Path(directory) / 'valid.metrics.json')
    expected = dict(num_beams=50, num_return_sequences=50, do_sample=False,
                    temperature=1.0, length_penalty=0.0, use_cache=True)
    if any(report['generation_config'].get(k) != v for k, v in expected.items()):
        raise ValueError('Unexpected validation generation config')
    if report['split'] != 'valid' or report['invalid_candidates'] != 0:
        raise ValueError('Invalid validation report')
    catalog = {}
    for item, sid in read_json(Path(data) / f'{category}.index.json').items():
        catalog.setdefault(''.join(sid), []).append(item)
    evaluated = []
    with (Path(data) / 'valid.csv').open() as src, gzip.open(Path(directory) / 'valid.predictions.jsonl.gz', 'rt') as pred:
        for row, line in itertools.zip_longest(csv.DictReader(src), pred):
            if row is None or line is None:
                raise ValueError('Incomplete validation coverage')
            record = json.loads(line)
            labels = dict(sample_id=row['sample_id'], user_id=row['user_id'],
                          target_position=int(row['target_position']), history_item_id=ast.literal_eval(row['history_item_id']),
                          target_item_id=row['item_id'], target_sid=row['item_sid'])
            if any(record.get(k) != v for k, v in labels.items()):
                raise ValueError('Validation labels/order changed')
            p, scores = record['predictions'], record['scores']
            if (len(p) != 50 or len(set(p)) != 50 or any(x not in catalog for x in p)
                    or record['predicted_item_id_groups'] != [catalog[x] for x in p]
                    or len(scores) != 50 or not all(math.isfinite(x) for x in scores)
                    or any(a < b - 1e-6 for a, b in zip(scores, scores[1:]))):
                raise ValueError('Invalid ranked candidates')
            evaluated.append({'target_sid': row['item_sid'], 'predictions': p})
    computed = metrics(evaluated)
    if any(not math.isclose(report[k], v, rel_tol=0, abs_tol=1e-12) for k, v in computed.items()):
        raise ValueError('Validation metric mismatch')
    return report
