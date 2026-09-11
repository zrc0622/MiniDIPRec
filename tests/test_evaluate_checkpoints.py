"""CPU tests for validation-only routing, reuse and artifact integrity."""
import csv
import fcntl
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from reproduction.evaluate import metrics
from reproduction.prepare import sha256, write_json
from scripts import evaluate_checkpoints as runner


class CheckpointValidationTests(unittest.TestCase):
    def fixture(self, temporary):
        repo = Path(temporary).resolve()
        category = 'Office_Products'
        root = repo / 'results/run'
        cat = root / category
        data = cat / 'data'
        data.mkdir(parents=True)
        weights = repo / 'checkpoints/run'
        for name in (*runner.EVAL_SOURCE, 'scripts/evaluate_checkpoints.py', 'scripts/evaluate_checkpoints.sh'):
            target = repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(runner.REPO / name, target)
            snapshot = root / 'source' / name
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(target, snapshot)
        write_json(root / 'source_sha256.json', {name: sha256(repo / name) for name in runner.EVAL_SOURCE})
        write_json(root / 'run_config.json', {'checkpoint_root': str(weights), 'gpus': '2,3,6,7', 'eval_batch_size': 2})
        catalog = {str(i): [f'<sid{i}>'] for i in range(60)}
        write_json(data / f'{category}.index.json', catalog)
        rows = [{'sample_id': f'valid:{i}', 'user_id': str(i), 'target_position': '1',
                 'history_item_id': '[59]', 'item_id': str(i), 'item_sid': f'<sid{i}>'} for i in range(2)]
        with (data / 'valid.csv').open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        # No test file is created: any accidental dependency on it fails.
        write_json(data / 'audit.json', {'category': category, 'history_limit': 50,
                   'generated_sha256': {name: sha256(data / name) for name in ('valid.csv', f'{category}.index.json')}})
        write_json(cat / 'lengths.json', {'data_audit_sha256': sha256(data / 'audit.json')})
        for stage, step in [('sft', 378), ('rl', 1400)]:
            folder = cat / stage
            folder.mkdir()
            write_json(folder / 'training.json', {'selected_step': step})
            write_json(folder / f'eval-{stage}.complete.json', {})
            model = weights / category / stage / 'selected_model'
            self.write_model(model)
            predictions = [{'sample_id': f'valid:{i}', 'user_id': str(i), 'target_position': 1,
                            'history_item_id': [59], 'target_item_id': str(i), 'target_sid': f'<sid{i}>',
                            'predictions': [f'<sid{j}>' for j in range(50)],
                            'predicted_item_id_groups': [[str(j)] for j in range(50)],
                            'scores': [-j * 0.1 for j in range(50)]} for i in range(2)]
            (folder / 'valid.predictions.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in predictions))
            write_json(folder / 'valid.metrics.json', {**metrics(predictions), 'split': 'valid',
                       'invalid_candidates': 0, 'model': str(model), 'generation_config': {
                           'num_beams': 50, 'num_return_sequences': 50, 'do_sample': False,
                           'temperature': 1.0, 'length_penalty': 0.0, 'use_cache': True,
                           'transformers_version': '4.57.1'}})
        for step in (175, 350, 1400, 1575):
            self.write_model(weights / category / 'rl' / f'checkpoint-{step}', step)
        write_json(cat / 'rl/trainer_state.json', {'log_history': [{'step': 175, 'eval_reward': 0.01}]})
        args = runner.parser().parse_args(['--run-name', 'run'])
        return repo, root, cat, weights, args

    def write_model(self, model, step=None):
        model.mkdir(parents=True)
        for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json'):
            write_json(model / name, {})
        (model / 'model.safetensors').write_bytes(b'fixture weights; never loaded')
        if step:
            write_json(model / 'trainer_state.json', {'global_step': step})

    def fake_launcher(self, cat, calls, fail_step=None):
        def launch(command, env, repo, output, log):
            calls.append((command, env.copy()))
            if 'reproduction.evaluate' not in command:
                return
            model = Path(command[command.index('--model') + 1])
            target = Path(command[command.index('--output') + 1])
            if model.name == f'checkpoint-{fail_step}':
                raise RuntimeError('Simulated GPU interruption')
            for name in ('valid.metrics.json', 'valid.predictions.jsonl'):
                shutil.copyfile(cat / 'rl' / name, target / name)
            report = runner.read_json(target / 'valid.metrics.json')
            report['model'] = str(model)
            write_json(target / 'valid.metrics.json', report)
        return launch

    def test_serial_valid_only_reuses_selected_and_resumes_without_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, root, cat, weights, args = self.fixture(tmp)
            original = {str(p): sha256(p) for p in root.rglob('*') if p.is_file()}
            plan = runner.build_plan(repo, args)
            calls = []
            runner.execute_plan(repo, plan, self.fake_launcher(cat, calls))
            evaluations = [cmd for cmd, env in calls if 'reproduction.evaluate' in cmd]
            self.assertEqual([Path(c[c.index('--model') + 1]).name for c in evaluations],
                             ['checkpoint-175', 'checkpoint-350', 'checkpoint-1575'])
            for cmd, env in calls:
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '2,3,6,7')
                self.assertEqual(env['HF_HUB_OFFLINE'], '1')
            for cmd in evaluations:
                self.assertIn('--nproc_per_node=4', cmd)
                self.assertEqual(cmd[cmd.index('--split') + 1], 'valid')
            output = Path(plan['output'])
            table = runner.read_json(output / 'summary.json')
            self.assertEqual(len(table), 5)
            self.assertEqual(table[0]['rl_step'], 0)
            self.assertEqual(table[0]['source_step'], 378)
            self.assertEqual(table[1]['eval_reward'], 0.01)
            self.assertEqual(table[3]['source'], 'reused_selected_export')
            for name, digest in original.items():
                self.assertEqual(sha256(name), digest)
            # Completed validation remains reusable even when original checkpoints are moved.
            shutil.rmtree(weights)
            runner.execute_plan(repo, runner.build_plan(repo, args), lambda *unused: self.fail('Unexpected launch'))
            self.assertFalse(any(p.is_symlink() or p.suffix == '.safetensors' for p in output.rglob('*')))

    def test_interrupted_checkpoint_restarts_but_keeps_completed_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, root, cat, weights, args = self.fixture(tmp)
            plan = runner.build_plan(repo, args)
            with self.assertRaisesRegex(RuntimeError, 'interruption'):
                runner.execute_plan(repo, plan, self.fake_launcher(cat, [], fail_step=350))
            output = Path(plan['output'])
            first = sha256(output / 'checkpoint-175/valid.predictions.jsonl')
            calls = []
            runner.execute_plan(repo, plan, self.fake_launcher(cat, calls))
            self.assertEqual(sha256(output / 'checkpoint-175/valid.predictions.jsonl'), first)
            self.assertEqual([Path(cmd[cmd.index('--model') + 1]).name for cmd, _ in calls
                              if 'reproduction.evaluate' in cmd], ['checkpoint-350', 'checkpoint-1575'])

    def test_bad_inputs_and_missing_checkpoints_fail_before_gpu(self):
        for problem in ('source', 'data', 'label', 'metrics', 'config', 'model_path', 'missing_model', 'wrong_step'):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                repo, root, cat, weights, args = self.fixture(tmp)
                if problem == 'source':
                    with (repo / 'reproduction/evaluate.py').open('a') as f:
                        f.write('\n# incompatible code\n')
                elif problem == 'data':
                    with (cat / 'data/valid.csv').open('a') as f:
                        f.write('\n')
                elif problem == 'label':
                    path = cat / 'sft/valid.predictions.jsonl'
                    path.write_text(path.read_text().replace('"target_sid": "<sid0>"', '"target_sid": "<sid1>"'))
                elif problem in ('metrics', 'config', 'model_path'):
                    path = cat / 'sft/valid.metrics.json'
                    report = runner.read_json(path)
                    if problem == 'metrics':
                        report['HR@10'] = 0.5
                    elif problem == 'config':
                        report['generation_config']['do_sample'] = True
                    else:
                        report['model'] = '/other/run/Office_Products/sft/selected_model'
                    write_json(path, report)
                elif problem == 'missing_model':
                    (weights / 'Office_Products/rl/checkpoint-350/model.safetensors').unlink()
                else:
                    write_json(weights / 'Office_Products/rl/checkpoint-350/trainer_state.json', {'global_step': 175})
                with self.assertRaises(ValueError):
                    plan = runner.build_plan(repo, args)
                    runner.execute_plan(repo, plan, lambda *unused: self.fail('Must fail before GPU launch'))

    def test_diagnostic_tampering_and_changed_settings_rejected(self):
        for problem in ('prediction', 'settings', 'interrupted_weights'):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                repo, root, cat, weights, args = self.fixture(tmp)
                plan = runner.build_plan(repo, args)
                if problem == 'interrupted_weights':
                    with self.assertRaises(RuntimeError):
                        runner.execute_plan(repo, plan, self.fake_launcher(cat, [], fail_step=350))
                    (weights / 'Office_Products/rl/checkpoint-350/model.safetensors').write_bytes(b'changed')
                else:
                    runner.execute_plan(repo, plan, self.fake_launcher(cat, []))
                    if problem == 'prediction':
                        with (Path(plan['output']) / 'checkpoint-175/valid.predictions.jsonl').open('a') as f:
                            f.write('{}\n')
                    else:
                        args.batch_size = 1
                        plan = runner.build_plan(repo, args)
                with self.assertRaises(ValueError):
                    runner.execute_plan(repo, plan, lambda *unused: self.fail('Unexpected launch'))

    def test_reevaluate_selected_and_relocated_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, root, cat, weights, args = self.fixture(tmp)
            args.reevaluate_selected = True
            args.checkpoint_root = str(repo / 'moved_checkpoints')
            shutil.move(weights.parent, args.checkpoint_root)
            plan = runner.build_plan(repo, args)
            self.assertTrue(all(job['reuse_from'] is None for job in plan['jobs']))
            calls = []
            runner.execute_plan(repo, plan, self.fake_launcher(cat, calls))
            self.assertEqual(sum('reproduction.evaluate' in cmd for cmd, _ in calls), 5)

    def test_four_gpu_and_step_constraints(self):
        for field, value in [('gpus', '0,1'), ('gpus', '0,1,2,2'), ('steps', [175, 175]),
                             ('steps', [0]), ('batch_size', 0)]:
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as tmp:
                repo, root, cat, weights, args = self.fixture(tmp)
                setattr(args, field, value)
                with self.assertRaises(ValueError):
                    runner.build_plan(repo, args)

    def test_concurrent_launch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, root, cat, weights, args = self.fixture(tmp)
            plan = runner.build_plan(repo, args)
            output = Path(plan['output'])
            output.mkdir(parents=True)
            with (output / '.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(ValueError, 'Another checkpoint'):
                    runner.execute_plan(repo, plan, lambda *unused: self.fail('Unexpected launch'))

    def test_sharded_weights_require_all_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / 'checkpoint-175'
            self.write_model(model, 175)
            (model / 'model.safetensors').unlink()
            write_json(model / 'model.safetensors.index.json', {'weight_map': {
                'embedding': 'model-00001-of-00002.safetensors', 'head': 'model-00002-of-00002.safetensors'}})
            (model / 'model-00001-of-00002.safetensors').write_bytes(b'first shard')
            with self.assertRaisesRegex(ValueError, 'Missing/empty'):
                runner.model_files(model, 175)
            (model / 'model-00002-of-00002.safetensors').write_bytes(b'second shard')
            self.assertIn('model-00002-of-00002.safetensors', runner.model_files(model, 175))


if __name__ == '__main__':
    unittest.main()
