"""Short-budget schedule parity, fresh SFT provenance, valid-only routing and resume."""
import ast
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from reproduction.prepare import CATEGORIES, sha256, write_json
from scripts import run_rl_short as runner
from tests import test_start_rl_from_sft as import_tests
from tests import test_evaluate_checkpoints as evaluation_tests
from tests.test_reproduction_runtime import HAS_RUNTIME


class ShortRunTests(unittest.TestCase):
    def fixture(self, temporary):
        repo, source, model = import_tests.FreshRLTests().fixture(temporary)
        names = [str(p.relative_to(runner.REPO)) for p in (runner.REPO / 'reproduction').glob('*.py')]
        names += ['data.py', 'minionerec_trainer.py', 'requirements-reproduction.txt',
                  'config/zero2_opt.yaml', 'scripts/reproduce.sh']
        for name in names:
            for base in (repo, source / 'source'):
                target = base / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(runner.REPO / name, target)
        write_json(source / 'source_sha256.json', {name: sha256(repo / name) for name in names})
        for name in ('run_rl_short.py', 'train_rl_short.py', 'start_rl_from_sft.py',
                     'migrate_rl_config_fix.py', 'evaluate_checkpoints.py', 'evaluate_checkpoints.sh'):
            shutil.copyfile(runner.REPO / 'scripts' / name, repo / 'scripts' / name)
        args = runner.parser().parse_args(['--source-run', 'run', '--run-name', 'short',
            '--gpus', '2,3,6,7', '--learning-rate', '5e-6', '--stage', 'prepare'])
        return repo, source, model, args

    def test_fresh_sft_copy_source_unchanged_and_resume_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, source, model, args = self.fixture(tmp)
            before = {str(p): sha256(p) for p in source.rglob('*') if p.is_file()}
            runner.run(repo, args, lambda *a: self.fail('Prepare must not launch GPUs'))
            root = repo / 'results/short'
            config = runner.evaluation.read_json(root / 'run_config.json')
            short = config['rl_short']
            self.assertEqual(short['snapshot_steps'], [50, 100, 175, 350])
            self.assertEqual(short['learning_rate'], 5e-6)
            self.assertEqual(config['rl_accumulation'], 16)
            self.assertEqual(config['gpus'], '2,3,6,7')
            self.assertFalse((root / CATEGORIES[0] / 'sft/test.metrics.json').exists())
            self.assertTrue((root / 'sft_import/historical_sft_test/test.metrics.json').is_file())
            self.assertFalse(any(p.is_symlink() for p in root.rglob('*')))
            self.assertEqual(before, {str(p): sha256(p) for p in source.rglob('*') if p.is_file()})
            command = runner.training_command(repo, root, config, short)
            self.assertNotIn('--max-steps', command)
            self.assertNotIn('--resume', command)
            self.assertEqual(Path(command[command.index('--model') + 1]).name, 'selected_model')
            self.assertIn('run_rl_short.py', (root / 'resume.sh').read_text())
            resume = runner.parser().parse_args(['--run-name', 'short', '--resume', '--stage', 'prepare'])
            shutil.rmtree(source)
            shutil.rmtree(model.parents[2])
            runner.run(repo, resume, lambda *a: self.fail('Prepare must not launch GPUs'))
            resume.learning_rate = 1e-5
            with self.assertRaisesRegex(ValueError, 'Cannot change learning_rate'):
                runner.setup_run(repo, resume)
            resume.learning_rate = None
            with (repo / 'scripts/train_rl_short.py').open('a') as f:
                f.write('# unexpected edit\n')
            with self.assertRaisesRegex(ValueError, 'Source changed'):
                runner.run(repo, resume)

    def test_invalid_budget_batch_and_rates_refused_before_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, source, model, args = self.fixture(tmp)
            for field, value in [('stop_after_steps', 0), ('snapshot_steps', [351]),
                                 ('snapshot_steps', [50, 50]), ('rl_micro_batch', 24),
                                 ('rl_micro_batch', 0), ('learning_rate', float('nan')),
                                 ('gpus', '0,1,2')]:
                original = getattr(args, field)
                setattr(args, field, value)
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    runner.setup_run(repo, args)
                self.assertFalse((repo / 'results/short').exists())
                setattr(args, field, original)

    def test_validation_routing_has_no_test_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            helper = evaluation_tests.CheckpointValidationTests()
            repo, root, cat, weights, _ = helper.fixture(tmp)
            config = runner.evaluation.read_json(root / 'run_config.json')
            config['run_name'] = 'run'
            short = {'dataset': cat.name, 'gpus': '2,3,6,7', 'snapshot_steps': [175, 350]}
            plan = runner.validation_plan(repo, root, config, short)
            self.assertEqual([job['rl_step'] for job in plan['jobs']], [0, 175, 350])
            self.assertTrue(all(job['reuse_from'] is None for job in plan['jobs']))
            calls = []
            runner.evaluation.execute_plan(repo, plan, helper.fake_launcher(cat, calls))
            commands = [cmd for cmd, env in calls if 'reproduction.evaluate' in cmd]
            self.assertEqual(len(commands), 3)
            self.assertTrue(all(cmd[cmd.index('--split') + 1] == 'valid' for cmd in commands))
            runner.evaluation.execute_plan(repo, plan, lambda *a: self.fail('Completed eval repeated'))

    def test_checkpoint_at_budget_finalizes_without_another_training_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, source, model, args = self.fixture(tmp)
            root, config, short = runner.setup_run(repo, args)
            cat = root / CATEGORIES[0]
            artifacts = cat / 'rl'
            artifacts.mkdir()
            weights = Path(config['checkpoint_root']) / CATEGORIES[0] / 'rl'
            helper = evaluation_tests.CheckpointValidationTests()
            for step in short['snapshot_steps']:
                checkpoint = weights / f'checkpoint-{step}'
                helper.write_model(checkpoint, step)
                write_json(checkpoint / 'trainer_state.json', {'global_step': step, 'max_steps': 1746,
                    'best_model_checkpoint': None, 'log_history': []})
                (checkpoint / 'reference_model.pt').write_bytes(b'reference fixture')
            write_json(artifacts / 'schedule.json', {'planned_steps': 1746})
            write_json(artifacts / 'training_args.json', {'max_steps': -1})
            resume = runner.parser().parse_args(['--run-name', 'short', '--resume', '--stage', 'train'])
            runner.run(repo, resume, lambda *a: self.fail('Budget completed; no GPU call expected'))
            self.assertEqual(runner.evaluation.read_json(artifacts / 'training.json')['final_step'], 350)
            runner.run(repo, resume, lambda *a: self.fail('Completed training repeated'))


@unittest.skipUnless(HAS_RUNTIME, 'Install the runtime for real Trainer lifecycle tests')
class ShortScheduleTests(unittest.TestCase):
    def test_official_config_parity(self):
        # Compare every explicitly configured field of the actual entrypoints.
        original = ast.parse((runner.REPO / 'reproduction/train.py').read_text())
        short = ast.parse((runner.REPO / 'scripts/train_rl_short.py').read_text())
        def fields(tree):
            call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                        and isinstance(n.func, ast.Name) and n.func.id == 'GRPOConfig')
            return {kw.arg: kw.value for kw in call.keywords}
        expected, actual = fields(original), fields(short)
        self.assertEqual(expected.keys(), actual.keys())
        adapted = {'output_dir', 'per_device_train_batch_size', 'per_device_eval_batch_size',
                   'gradient_accumulation_steps', 'learning_rate', 'max_steps'}
        for key in expected.keys() - adapted:
            self.assertEqual(ast.dump(expected[key]), ast.dump(actual[key]), key)
        self.assertEqual(ast.literal_eval(actual['max_steps']), -1)

    def test_real_rl_schedule_prefix_and_interrupted_resume(self):
        import torch
        from datasets import Dataset
        from trl import GRPOConfig
        from transformers import TrainerCallback, set_seed
        from minionerec_trainer import ReReTrainer
        from reproduction.contracts import ranking_rewards
        from scripts.train_rl_short import StopAfterUpdates
        from tests.test_reproduction_runtime import RuntimeTests
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / 'model'
            base.mkdir()
            tok, _, sids, info = RuntimeTests().fixture(base)
            rows = [{'prompt': '### Response:\n', 'target': sids[i] + '\n', 'sample_id': str(i)} for i in range(4)]

            def reward(prompts, completions, target, **kwargs):
                return ranking_rewards(completions, target)[0]

            class Capture(TrainerCallback):
                def __init__(self):
                    self.rates = []
                def on_step_begin(self, args, state, control, optimizer=None, **kwargs):
                    self.rates.append((state.global_step, optimizer.param_groups[0]['lr']))

            class Interrupt(TrainerCallback):
                def on_save(self, args, state, control, **kwargs):
                    if state.global_step == 1:
                        raise RuntimeError('simulated interruption after checkpoint')

            class Snapshots(TrainerCallback):
                def on_step_end(self, args, state, control, **kwargs):
                    if state.global_step in (1, 3):
                        control.should_save = True

            def make(name, budget=None, interrupt=False):
                set_seed(42)
                output = Path(tmp) / name
                conf = GRPOConfig(output_dir=str(output), use_cpu=True, bf16=False, fp16=False,
                    per_device_train_batch_size=16, per_device_eval_batch_size=16, num_generations=16,
                    gradient_accumulation_steps=2, num_train_epochs=2, max_steps=-1,
                    learning_rate=5e-6, warmup_ratio=.03, lr_scheduler_type='cosine',
                    optim='adamw_torch', gradient_checkpointing=False, report_to='none',
                    save_steps=2, eval_strategy='steps', eval_steps=2, logging_steps=1,
                    load_best_model_at_end=True, metric_for_best_model='eval_reward', greater_is_better=True,
                    max_completion_length=5, beta=.001, sync_ref_model=True, ref_model_sync_steps=512,
                    model_init_kwargs={'torch_dtype': 'float32'}, seed=42, disable_tqdm=True)
                conf.max_prompt_length = 64
                capture = Capture()
                callbacks = [capture, Snapshots()]
                if budget:
                    callbacks.append(StopAfterUpdates(budget, [1, budget], output / 'artifacts'))
                if interrupt:
                    callbacks.append(Interrupt())
                trainer = ReReTrainer(model=str(base), base_model=str(base), processing_class=tok,
                    args=conf, train_dataset=Dataset.from_list(rows), eval_dataset=Dataset.from_list(rows[:2]),
                    reward_funcs=[reward], beam_search=True, test_during_training=False,
                    info_file=str(info), callbacks=callbacks)
                return trainer, capture

            full, full_rates = make('full')
            full.train()
            short, short_rates = make('short', 3)
            short.train()
            self.assertEqual(full.state.max_steps, 4)
            self.assertEqual(short.state.max_steps, 4)
            self.assertEqual(short.state.global_step, 3)
            self.assertEqual(short_rates.rates, full_rates.rates[:3])
            self.assertEqual([x['step'] for x in short.state.log_history if 'eval_reward' in x], [2])
            from safetensors.torch import load_file
            full_snapshot = load_file(Path(tmp) / 'full/checkpoint-3/model.safetensors')
            short_snapshot = load_file(Path(tmp) / 'short/checkpoint-3/model.safetensors')
            for name, value in full_snapshot.items():
                torch.testing.assert_close(value, short_snapshot[name], rtol=0, atol=0)
            broken, _ = make('resume', 3, interrupt=True)
            with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                broken.train()
            resumed, resumed_rates = make('resume', 3)
            resumed.train(resume_from_checkpoint=str(Path(tmp) / 'resume/checkpoint-1'))
            self.assertEqual(resumed_rates.rates, short_rates.rates[1:])
            self.assertEqual(resumed.state.global_step, 3)
            for name, value in short.model.state_dict().items():
                torch.testing.assert_close(value, resumed.model.state_dict()[name], rtol=0, atol=0)
            for name, value in short.ref_model.state_dict().items():
                torch.testing.assert_close(value, resumed.ref_model.state_dict()[name], rtol=0, atol=0)
            resumed_snapshot = load_file(Path(tmp) / 'resume/checkpoint-3/model.safetensors')
            for name, value in short_snapshot.items():
                torch.testing.assert_close(value, resumed_snapshot[name], rtol=0, atol=0)
            # The guard prevents silently spending an extra optimizer step.
            with self.assertRaisesRegex(ValueError, 'already reached stop budget'):
                resumed.train(resume_from_checkpoint=str(Path(tmp) / 'resume/checkpoint-3'))


if __name__ == '__main__':
    unittest.main()
