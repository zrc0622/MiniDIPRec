"""Two-dataset orchestration, fixed recipe, early checkpoints and recovery contracts."""
import fcntl
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from reproduction.prepare import CATEGORIES, sha256, write_json
from scripts import run_qwen3_two as two
from scripts import qwen3_two_utils as helpers
from scripts.five_utils import checkpoint_manifest, resumable_checkpoint
from scripts.six_utils import file_manifest, read_json
from tests import test_evaluate_checkpoints as checkpoint_tests
from tests import test_rl_five as five_tests
from tests import test_rl_six as six_tests
from tests.test_reproduction_runtime import HAS_RUNTIME


class TwoSuiteTests(unittest.TestCase):
    def fixture(self, tmp):
        # Reuse the realistic raw-data fixture, then remove every old-run artifact.
        repo, _, _, _ = five_tests.FiveSuiteTests().fixture(tmp)
        shutil.rmtree(repo / 'results')
        shutil.rmtree(repo / 'checkpoints')
        for name in two.source_files(two.REPO):
            target = repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(two.REPO / name, target)
        args = two.parser().parse_args(['--gpus', '2,3,6,7'])
        return repo, args

    def gpu(self, calls, fail=None):
        evaluate = six_tests.SixSuiteTests().gpu([])
        def launch(command, env, repo, root, log):
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '2,3,6,7')
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text('Simulated GPU orchestration fixture\n')
            if '-c' in command:
                write_json(command[-1], {'fake_environment': True})
                return
            if 'scripts.train_qwen3_two' not in command:
                calls.append(('eval', command[command.index('--model') + 1]))
                return evaluate(command, env, repo, root, log)
            spec = read_json(command[command.index('--spec') + 1])
            stage = command[command.index('--stage') + 1]
            label = spec['variant']['label']
            calls.append((label, stage))
            helpers.validate_variant(spec['variant'])
            if label == fail and stage == 'rl':
                raise RuntimeError('simulated interruption')
            if stage == 'preflight':
                self.assertEqual(spec['model'], helpers.DEFAULT_MODEL)
                write_json(spec['lengths'], dict(recipe='current',
                    data_audit_sha256=sha256(Path(spec['data']) / 'audit.json'), sft_limit=256,
                    rl_prompt_limit=2048, eval_prompt_max=256, completion_limit=5))
            elif stage == 'sft':
                self.assertIn('torch.distributed.run', command)
                self.assertEqual(spec['model'], helpers.DEFAULT_MODEL)
                self.assertNotIn('--resume', command)
                model = Path(spec['output']) / 'selected_model'
                checkpoint_tests.CheckpointValidationTests().write_model(model)
                write_json(Path(spec['artifacts']) / 'complete.json', dict(model=str(model),
                    selected_step=378, final_step=504, model_sha256=file_manifest(model)))
            else:
                self.assertIn('accelerate.commands.launch', command)
                self.assertEqual(Path(spec['model']).parent.parent.name, label)
                schedule = helpers.schedule_spec(55842)
                for step in helpers.SNAPSHOTS:
                    checkpoint = Path(spec['output']) / f'checkpoint-{step}'
                    checkpoint_tests.CheckpointValidationTests().write_model(checkpoint, step)
                    write_json(checkpoint / 'trainer_state.json', dict(global_step=step,
                        max_steps=schedule['full_scheduler_steps']))
                    (checkpoint / 'reference_model.pt').write_bytes(b'reference')
                    write_json(checkpoint / 'five_complete.json', dict(stage='rl', step=step,
                        files=checkpoint_manifest(checkpoint)))
                write_json(Path(spec['artifacts']) / 'schedule.json', dict(schedule,
                    trainer_max_steps=schedule['full_scheduler_steps']))
        return launch

    def test_serial_fresh_models_all_snapshots_and_idempotent_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, args = self.fixture(tmp)
            old_sources = {name: sha256(repo / name) for name in two.source_files(repo)}
            calls = []
            two.run(repo, args, self.gpu(calls))
            root = repo / 'results2' / args.run_name
            self.assertEqual([c for c in calls if c[1] in ('sft', 'rl')],
                [('01_office', 'sft'), ('01_office', 'rl'), ('02_industrial', 'sft'), ('02_industrial', 'rl')])
            first_industrial = calls.index(('02_industrial', 'sft'))
            self.assertEqual(sum(c[0] == 'eval' for c in calls[:first_industrial]), 6)
            self.assertEqual(len(read_json(root / 'summary.json')), 10)
            self.assertEqual(len(list(root.rglob('valid.predictions.jsonl.gz'))), 12)
            self.assertFalse(list(root.rglob('valid.predictions.jsonl')))
            self.assertFalse(any(p.suffix in ('.safetensors', '.pt', '.bin') for p in root.rglob('*')))
            self.assertEqual(old_sources, {name: sha256(repo / name) for name in old_sources})
            calls.clear()
            two.run(repo, args, self.gpu(calls))
            self.assertEqual(calls, [])
            from scripts.package_qwen3_two import package
            self.assertTrue(package(repo, args.run_name).exists())

    def test_failure_stops_later_dataset_then_resumes_own_sft(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, args = self.fixture(tmp)
            calls = []
            with self.assertRaisesRegex(RuntimeError, 'simulated'):
                two.run(repo, args, self.gpu(calls, fail='01_office'))
            self.assertNotIn(('02_industrial', 'sft'), calls)
            calls.clear()
            two.run(repo, args, self.gpu(calls))
            self.assertNotIn(('01_office', 'sft'), calls)
            args.eval_batch_size = 2
            with self.assertRaisesRegex(ValueError, 'settings changed'):
                two.run(repo, args, self.gpu([]))

    def test_finalize350_without_training_requires25_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, args = self.fixture(tmp)
            args.experiments, args.stage = [1], 'train'
            two.run(repo, args, self.gpu([]))
            root = repo / 'results2' / args.run_name
            done = root / 'experiments/01_office/rl/complete.json'
            done.unlink()
            calls = []
            two.run(repo, args, self.gpu(calls))
            self.assertEqual(calls, [])
            spec = read_json(root / 'experiments/01_office/rl_spec.json')
            missing = Path(spec['output']) / 'checkpoint-25/five_complete.json'
            missing.unlink()
            with self.assertRaises(FileNotFoundError):
                helpers.finish_rl(spec)

    def test_setup_interruption_source_changes_and_no_test_dependency(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, args = self.fixture(tmp)
            args.stage = 'prepare'
            with patch.object(two, 'prepare_official', side_effect=RuntimeError('prepare interrupted')):
                with self.assertRaisesRegex(RuntimeError, 'prepare interrupted'):
                    two.run(repo, args, self.gpu([]))
            two.run(repo, args, self.gpu([]))
            root = repo / 'results2' / args.run_name
            self.assertFalse(list((root / 'shared').rglob('test.csv')))
            with (repo / 'reproduction/contracts.py').open('a') as stream:
                stream.write('\n# changed\n')
            with self.assertRaisesRegex(ValueError, 'changed'):
                two.run(repo, args, self.gpu([]))

    def test_dryrun_input_validation_and_packaging_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, args = self.fixture(tmp)
            args.dry_run = True
            before = file_manifest(repo)
            two.run(repo, args, self.gpu([]))
            self.assertEqual(before, file_manifest(repo))
            for model in ['Qwen/Qwen3-0.6B', 'Qwen/Qwen3-1.7B-Base', 'Qwen/Qwen2.5-1.5B']:
                args.model = model
                with self.assertRaisesRegex(ValueError, 'Qwen3-1.7B'):
                    two.settings(repo, args)
            args.model = helpers.DEFAULT_MODEL
            args.checkpoint_root = 'results2'
            with self.assertRaisesRegex(ValueError, 'separate'):
                two.settings(repo, args)
            args.checkpoint_root, args.dry_run, args.stage = 'checkpoints', False, 'prepare'
            two.run(repo, args, self.gpu([]))
            root = repo / 'results2' / args.run_name
            from scripts.package_qwen3_two import package
            with (root / '.five.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(ValueError, 'running'):
                    package(repo, args.run_name)
            (root / 'model.safetensors').write_bytes(b'weight')
            with self.assertRaisesRegex(ValueError, 'Weights'):
                package(repo, args.run_name)


@unittest.skipUnless(HAS_RUNTIME, 'Pinned ML runtime required')
class TwoRuntimeTests(unittest.TestCase):
    def setUp(self):
        guard = patch('transformers.utils.import_utils._mlx_available', False)
        guard.start()
        self.addCleanup(guard.stop)

    def test_1p7b_contract_rejects_other_scales_and_preserves_current_config(self):
        from transformers import Qwen3Config
        from trl import GRPOConfig
        from scripts.train_qwen3_two import training_config
        from scripts.train_rl_five import training_config as previous
        from scripts.five_utils import variants as previous_variants
        config = Qwen3Config(hidden_size=2048, intermediate_size=6144, num_hidden_layers=28,
            num_attention_heads=16, num_key_value_heads=8, head_dim=128, tie_word_embeddings=True)
        count = helpers.model_contract(config, 152265)['parameters_after_sid_extension']
        self.assertGreater(count, 1_600_000_000)
        self.assertLess(count, 1_800_000_000)
        config.hidden_size = 1024
        with self.assertRaisesRegex(ValueError, 'architecture'):
            helpers.model_contract(config, 152265)
        lengths = dict(completion_limit=5, rl_prompt_limit=1792)
        with tempfile.TemporaryDirectory() as tmp, patch('trl.GRPOConfig',
                side_effect=lambda **kwargs: GRPOConfig(**kwargs, use_cpu=True)):
            new = training_config(tmp, lengths, helpers.variants()[0], helpers.schedule_spec(55842))
            old = previous(tmp, lengths, previous_variants()[0], helpers.schedule_spec(55842))
            self.assertEqual(new.to_dict(), old.to_dict())
            self.assertEqual(new.beta, .04)
            self.assertEqual(new.per_device_train_batch_size * new.gradient_accumulation_steps * 4, 1024)

    def test_25_snapshot_full_schedule_and_stop350(self):
        from transformers import TrainerState, TrainerControl, TrainingArguments
        from scripts.train_rl_six import stop_callback
        schedule = helpers.schedule_spec(55842)
        self.assertEqual(schedule['full_scheduler_steps'], 1746)
        self.assertEqual(schedule['warmup_steps'], 53)
        with tempfile.TemporaryDirectory() as tmp:
            callback = stop_callback(schedule, tmp)
            args = TrainingArguments(tmp, use_cpu=True, report_to='none')
            state = TrainerState(global_step=0, max_steps=1746, is_world_process_zero=True)
            callback.on_train_begin(args, state, TrainerControl())
            for step in [24, 25, 50, 100, 175, 349, 350]:
                state.global_step = step
                control = callback.on_step_end(args, state, TrainerControl())
                self.assertEqual(control.should_save, step in helpers.SNAPSHOTS)
                self.assertEqual(control.should_training_stop, step == 350)

    def test_actual_worker_sft_export_and_rl_exact_resume(self):
        """Run the new worker with real trainers; only scale/hardware/data are tiny."""
        import json
        import torch
        from transformers import AutoModelForCausalLM, TrainingArguments
        from trl import GRPOConfig
        from tests.test_reproduction_runtime import RuntimeTests
        from scripts import train_qwen3_two as worker
        from scripts.five_utils import row_digest
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / 'base'; base.mkdir()
            tok, _, sids, info = RuntimeTests().fixture(base)
            data = root / 'data'; data.mkdir()
            shutil.copyfile(info, data / 'info.txt')
            write_json(data / 'Office_Products.index.json',
                {str(i): [f'<a_{i}>', '<b_0>', '<c_0>'] for i in range(20)})
            write_json(data / 'audit.json', {'generated_sha256': file_manifest(data)})
            rows = [dict(prompt='### Response:\n', target=sids[i] + '\n',
                         sample_id=f'train:0:{i}') for i in range(4)]
            prefix = tok.encode('### Response:\n', add_special_tokens=False)
            sft_rows = []
            for row in rows:
                target = tok.encode(row['target'], add_special_tokens=False) + [tok.eos_token_id]
                ids = prefix + target
                sft_rows.append(dict(input_ids=ids, attention_mask=[1] * len(ids),
                                     labels=[-100] * len(prefix) + target))
            lengths = root / 'lengths.json'
            write_json(lengths, dict(recipe='current', data_audit_sha256=sha256(data / 'audit.json'),
                sft_limit=128, rl_prompt_limit=64, completion_limit=5, rl_row_order_sha256=row_digest(rows)))
            spec = dict(model=str(base), data=str(data), category='Office_Products',
                variant=helpers.variants()[0], lengths=str(lengths), sft_micro_batch=2,
                artifacts=str(root / 'sft_artifacts'), output=str(root / 'sft'))
            def cpu_sft_args(**kwargs):
                kwargs.update(use_cpu=True, bf16=False, gradient_accumulation_steps=1,
                    max_steps=2, eval_steps=1, save_steps=1, warmup_steps=0)
                return TrainingArguments(**kwargs)
            load_model = AutoModelForCausalLM.from_pretrained
            def cpu_model(*args, **kwargs):
                kwargs['torch_dtype'] = torch.float32
                return load_model(*args, **kwargs)
            with patch.object(worker, 'require_four_gpus'), patch.object(worker, 'model_contract'), \
                    patch.object(worker, 'sft_data', return_value=([sft_rows], sft_rows[:2])), \
                    patch('transformers.TrainingArguments', side_effect=cpu_sft_args), \
                    patch.object(AutoModelForCausalLM, 'from_pretrained', side_effect=cpu_model):
                worker.train(spec, 'sft')
            record = read_json(root / 'sft_artifacts/complete.json')
            self.assertEqual(record['final_step'], 2)
            self.assertIn(record['selected_step'], [1, 2])
            self.assertEqual(record['model_sha256'], file_manifest(root / 'sft/selected_model'))
            self.assertEqual(read_json(root / 'sft_artifacts/prepared_runtime.json')['world_size'], 1)
            schedule = dict(stop_after_steps=3, snapshot_steps=[1, 3], full_scheduler_steps=8)
            def cpu_rl_config(output, lengths, variant, schedule):
                conf = GRPOConfig(output_dir=str(output), use_cpu=True, bf16=False,
                    per_device_train_batch_size=16, per_device_eval_batch_size=16, num_generations=16,
                    gradient_accumulation_steps=2, num_train_epochs=2, max_steps=-1,
                    learning_rate=1e-5, warmup_steps=1, lr_scheduler_type='cosine',
                    optim='adamw_torch', report_to='none', save_strategy='no', eval_strategy='no',
                    max_completion_length=5, beta=.04, model_init_kwargs={'torch_dtype': 'float32'},
                    sync_ref_model=True, ref_model_sync_steps=1,
                    gradient_checkpointing=False, disable_tqdm=True, logging_steps=1)
                conf.max_prompt_length = 64
                return conf
            original_prepared = worker.prepared_callback
            def interrupt_callback(artifacts):
                class Interrupt(type(original_prepared(artifacts))):
                    def on_save(self, args, state, control, **kwargs):
                        if state.global_step == 1:
                            raise RuntimeError('interrupted after committed checkpoint')
                return Interrupt()
            def run(name, resume=None, interrupt=False):
                rl_spec = dict(spec, model=record['model'], output=str(root / name),
                               artifacts=str(root / f'{name}_artifacts'))
                with patch.object(worker, 'require_four_gpus'), patch.object(worker, 'model_contract'), \
                        patch.object(worker, 'rl_data', return_value=(rows, rows[:2])), \
                        patch.object(worker, 'schedule_spec', return_value=schedule), \
                        patch.object(worker, 'training_config', side_effect=cpu_rl_config), \
                        patch.object(worker, 'prepared_callback', side_effect=(
                            interrupt_callback if interrupt else original_prepared)):
                    worker.train(rl_spec, 'rl', resume)
            run('full')
            with self.assertRaisesRegex(RuntimeError, 'committed checkpoint'):
                run('resumed', interrupt=True)
            resume = resumable_checkpoint(root / 'resumed', 'rl')
            self.assertEqual(Path(resume).name, 'checkpoint-1')
            run('resumed', resume)
            full = load_model(root / 'full/checkpoint-3')
            resumed = load_model(root / 'resumed/checkpoint-3')
            for name, value in full.state_dict().items():
                torch.testing.assert_close(value, resumed.state_dict()[name], rtol=0, atol=0)
            for filename in ['reference_model.pt', 'scheduler.pt']:
                left = torch.load(root / 'full/checkpoint-3' / filename, weights_only=True)
                right = torch.load(root / 'resumed/checkpoint-3' / filename, weights_only=True)
                for key in left:
                    if isinstance(left[key], torch.Tensor):
                        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
                    else:
                        self.assertEqual(left[key], right[key])
            metrics = [json.loads(line) for line in (root / 'resumed_artifacts/metrics.jsonl').read_text().splitlines()]
            self.assertEqual([row['step'] for row in metrics if 'loss' in row], [1, 2, 3])
            self.assertTrue(any(row.get('grad_norm', 0) > 0 for row in metrics))
            runtime = read_json(root / 'resumed_artifacts/prepared_runtime.json')
            self.assertEqual(runtime['first_step'], 1)
            self.assertEqual(runtime['model_dtype_after_prepare'], 'torch.float32')


if __name__ == '__main__':
    unittest.main()
