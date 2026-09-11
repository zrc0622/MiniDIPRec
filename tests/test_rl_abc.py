"""Serial variants, fresh initialization, strict retry and portable result copies."""
import ast
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest

from reproduction.prepare import sha256, write_json
from scripts import run_rl_abc as abc
from scripts import run_rl_short as short
from tests import test_rl_short as short_tests
from tests import test_evaluate_checkpoints as eval_tests


class ABCSuiteTests(unittest.TestCase):
    def fixture(self, tmp):
        repo, source, model, _ = short_tests.ShortRunTests().fixture(tmp)
        for name in ('run_rl_abc.py', 'package_results.py'):
            shutil.copyfile(abc.REPO / 'scripts' / name, repo / 'scripts' / name)
        args = abc.parser().parse_args(['--source-run', 'run', '--run-name', 'abc', '--gpus', '2,3,6,7'])
        return repo, source, args

    def launcher(self, calls, fail=None):
        def launch(command, env, repo, output, log):
            parsed = short.parser().parse_args(command[2:])
            calls.append((parsed.run_name, parsed.stage, parsed.resume))
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '2,3,6,7')
            self.assertEqual(parsed.gpus, '2,3,6,7')
            if parsed.stage == 'prepare':
                short.run(repo, parsed, lambda *a: self.fail('Prepare launched GPU'))
                return
            if parsed.run_name == fail:
                raise RuntimeError('simulated B failure')
            root = repo / 'results' / parsed.run_name
            config = json.loads((root / 'run_config.json').read_text())
            self.assertEqual(config['rl_short']['beta'], parsed.beta)
            self.assertEqual(config['rl_short']['do_sample'], parsed.do_sample)
            self.assertEqual(config['rl_accumulation'], 16)
            rows = []
            for step in [0, *parsed.snapshot_steps]:
                rows.append({'checkpoint': f'checkpoint-{step}' if step else 'sft', 'rl_step': step,
                    'source_step': step or 378, 'HR@5': .2, 'HR@10': .23, 'Recall@5': .2, 'Recall@10': .23,
                    'NDCG@5': .18, 'NDCG@10': .188, 'delta_HR@10': 0., 'delta_NDCG@10': 0.})
            write_json(root / 'summary.json', rows)
            log.write_text('simulated worker log\n')
        return launch

    def test_serial_shared_sft_and_portable_copy_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, source, args = self.fixture(tmp)
            original = {str(p): sha256(p) for p in source.rglob('*') if p.is_file()}
            calls = []
            abc.run(repo, args, self.launcher(calls))
            self.assertEqual(calls, [('abc_A', 'prepare', False), ('abc_B', 'prepare', False),
                ('abc_C', 'prepare', False), ('abc_A', 'all', True), ('abc_B', 'all', True), ('abc_C', 'all', True)])
            root = repo / 'results/abc'
            self.assertEqual(len(json.loads((root / 'summary.json').read_text())), 12)
            for label, beta, sample in abc.VARIANTS:
                copied = root / args.dataset / label
                record = json.loads((copied / 'suite_copy.json').read_text())
                self.assertTrue(all(sha256(copied / name) == value for name, value in record['sha256'].items()))
                conf = json.loads((copied / 'run_config.json').read_text())
                self.assertEqual(conf['rl_short']['beta'], beta)
                self.assertIs(conf['rl_short']['do_sample'], sample)
                self.assertEqual(conf['rl_short']['stop_after_steps'], 175)
                self.assertEqual(conf['rl_short']['snapshot_steps'], [50, 100, 175])
                self.assertTrue(Path(conf['checkpoint_root']).is_dir())
            self.assertFalse(any(p.is_symlink() for p in root.rglob('*')))
            self.assertEqual(original, {str(p): sha256(p) for p in source.rglob('*') if p.is_file()})
            # All children were imported up front; retry does not need the source.
            shutil.rmtree(source)
            calls.clear()
            abc.run(repo, args, self.launcher(calls))
            self.assertTrue(all(resume for _, _, resume in calls))
            args.stop_after_steps = 100
            with self.assertRaisesRegex(ValueError, 'Suite settings changed'):
                abc.run(repo, args, self.launcher(calls))
            args.stop_after_steps = 175
            (root / 'source/scripts/run_rl_abc.py').write_text('tamper')
            with self.assertRaisesRegex(ValueError, 'source snapshot changed'):
                abc.run(repo, args, self.launcher(calls))

    def test_failure_stops_before_c_and_retains_a_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, source, args = self.fixture(tmp)
            calls = []
            with self.assertRaisesRegex(RuntimeError, 'simulated B'):
                abc.run(repo, args, self.launcher(calls, fail='abc_B'))
            self.assertNotIn(('abc_C', 'all', True), calls)
            rows = json.loads((repo / 'results/abc/summary.json').read_text())
            self.assertEqual({r['experiment'] for r in rows}, {'A'})
            self.assertTrue((repo / 'results/abc/resume.sh').is_file())

    def test_reject_mixed_sft_before_any_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, source, args = self.fixture(tmp)
            args.stage = 'prepare'
            abc.run(repo, args, self.launcher([]))
            record = repo / 'results/abc_B/sft_import/record.json'
            value = json.loads(record.read_text())
            value['selected_step'] += 1
            write_json(record, value)
            calls = []
            args.stage = 'all'
            with self.assertRaisesRegex(ValueError, 'identical selected SFT'):
                abc.run(repo, args, self.launcher(calls))
            self.assertTrue(all(stage == 'prepare' for _, stage, _ in calls))

    def test_smaller_budget_and_invalid_options(self):
        args = abc.parser().parse_args(['--run-name', 'quick', '--stop-after-steps', '100'])
        self.assertEqual(abc.settings(args)['snapshot_steps'], [50, 100])
        for field, value in [('stop_after_steps', 0), ('rl_micro_batch', 24), ('snapshot_steps', [101]),
                             ('snapshot_steps', [50, 50]), ('gpus', '0,1,2,2')]:
            original = getattr(args, field)
            setattr(args, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                abc.settings(args)
            setattr(args, field, original)

    def test_full_orchestration_with_fake_gpu_and_real_packaging(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, source, args = self.fixture(tmp)
            calls = []
            def gpu(command, env, repo, output, log):
                calls.append(command)
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '2,3,6,7')
                log.parent.mkdir(parents=True, exist_ok=True)
                log.write_text('CPU fixture; GPU command simulated\n')
                if 'scripts.train_rl_short' in command:
                    weights = Path(command[command.index('--output') + 1])
                    artifacts = Path(command[command.index('--artifacts') + 1])
                    steps = list(map(int, command[command.index('--snapshot-steps') + 1:]))
                    self.assertNotIn('--resume', command)  # Fresh RL for every variant.
                    for step in steps:
                        path = weights / f'checkpoint-{step}'
                        eval_tests.CheckpointValidationTests().write_model(path, step)
                        write_json(path / 'trainer_state.json', {'global_step': step, 'max_steps': 1746,
                                   'best_model_checkpoint': None, 'log_history': []})
                        (path / 'reference_model.pt').write_bytes(b'fake reference')
                    write_json(artifacts / 'schedule.json', {'planned_steps': 1746})
                    write_json(artifacts / 'training_args.json', {'max_steps': -1,
                               'beta': float(command[command.index('--beta') + 1])})
                elif 'reproduction.evaluate' in command:
                    self.assertEqual(command[command.index('--split') + 1], 'valid')
                    data = Path(command[command.index('--data') + 1])
                    target = Path(command[command.index('--output') + 1])
                    category = command[command.index('--category') + 1]
                    catalog = {}
                    for item, sid in json.loads((data / f'{category}.index.json').read_text()).items():
                        catalog.setdefault(''.join(sid), []).append(item)
                    candidates = list(catalog)[:50]
                    predictions = []
                    with (data / 'valid.csv').open() as f:
                        for row in csv.DictReader(f):
                            predictions.append({'sample_id': row['sample_id'], 'user_id': row['user_id'],
                                'target_position': int(row['target_position']),
                                'history_item_id': ast.literal_eval(row['history_item_id']),
                                'target_item_id': row['item_id'], 'target_sid': row['item_sid'],
                                'predictions': candidates, 'predicted_item_id_groups': [catalog[s] for s in candidates],
                                'scores': [-i * .1 for i in range(50)]})
                    (target / 'valid.predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in predictions))
                    write_json(target / 'valid.metrics.json', {**short.evaluation.metrics(predictions),
                        'split': 'valid', 'invalid_candidates': 0, 'model': command[command.index('--model') + 1],
                        'generation_config': {'num_beams': 50, 'num_return_sequences': 50, 'do_sample': False,
                            'temperature': 1., 'length_penalty': 0., 'use_cache': True, 'transformers_version': '4.57.1'}})
            def child(command, env, repo, output, log):
                short.run(repo, short.parser().parse_args(command[2:]), gpu)
            abc.run(repo, args, child)
            jobs = [c for c in calls if 'scripts.train_rl_short' in c or 'reproduction.evaluate' in c]
            self.assertEqual(len(jobs), 15)  # train + SFT/50/100/175 for each variant
            for index, (_, beta, sample) in enumerate(abc.VARIANTS):
                train = jobs[index * 5]
                self.assertIn('scripts.train_rl_short', train)
                self.assertEqual(float(train[train.index('--beta') + 1]), beta)
                self.assertEqual('--do-sample' in train, sample)
                self.assertTrue(all('reproduction.evaluate' in c for c in jobs[index * 5 + 1:index * 5 + 5]))
            calls.clear()
            abc.run(repo, args, child)
            self.assertEqual(calls, [])  # Complete suite genuinely skips all GPU calls.
            root = repo / 'results/abc'
            subprocess.run([sys.executable, str(repo / 'scripts/package_results.py'), 'abc'], check=True,
                           capture_output=True, text=True)
            with tarfile.open(repo / 'results/abc.tar.gz') as archive:
                members = archive.getmembers()
                self.assertFalse(any(m.issym() or m.islnk() for m in members))
                names = [m.name for m in members]
                for label, _, _ in abc.VARIANTS:
                    self.assertIn(f'abc/{args.dataset}/{label}/{args.dataset}/diagnostics/short_validation/checkpoint-175/valid.predictions.jsonl', names)
                self.assertFalse(any(n.endswith(('.pt', '.bin', '.safetensors')) for n in names))
            # Standalone analysis copy survives removal of the working run folders.
            for label, _, _ in abc.VARIANTS:
                shutil.rmtree(repo / f'results/abc_{label}')
            self.assertEqual(len(json.loads((root / 'summary.json').read_text())), 12)


if __name__ == '__main__':
    unittest.main()
