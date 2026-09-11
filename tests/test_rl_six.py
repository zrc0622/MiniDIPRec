"""Six-suite contracts, resumable orchestration and actual CPU Ranking lifecycle."""
import ast
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from reproduction.prepare import sha256, write_json
from scripts import run_rl_six as six
from scripts import six_utils as helpers
from tests import test_rl_short as previous
from tests import test_evaluate_checkpoints as validation
from tests.test_reproduction_runtime import HAS_RUNTIME


class SixSuiteTests(unittest.TestCase):
    def fixture(self, temporary):
        repo, source, model, _ = previous.ShortRunTests().fixture(temporary)
        for name in six.SUITE_FILES:
            dest = repo / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(six.REPO / name, dest)
        category = 'Office_Products'
        data = source / category / 'data'
        index = json.loads((data / f'{category}.index.json').read_text())
        index.update({f'extra{i}': [f'<a_{i+100}>', '<b_0>', '<c_0>'] for i in range(60)})
        write_json(data / f'{category}.index.json', index)
        audit = json.loads((data / 'audit.json').read_text())
        audit['generated_sha256'][f'{category}.index.json'] = sha256(data / f'{category}.index.json')
        write_json(data / 'audit.json', audit)
        write_json(source / category / 'lengths.json', {'model': 'Qwen/Qwen3-0.6B',
            'data_audit_sha256': sha256(data / 'audit.json'), 'completion_limit': 5,
            'sft_limit': 512, 'rl_prompt_limit': 2048, 'eval_prompt_max': 1024})
        args = six.parser().parse_args(['--run-name', 'six', '--source-run', 'run', '--gpus', '2,3,6,7'])
        return repo, source, model, args

    def gpu(self, calls, fail=None):
        def launch(command, env, repo, root, log):
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '2,3,6,7')
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text('fake GPU output\n')
            if '-c' in command:
                write_json(command[-1], {'fake_environment': True})
                return
            if 'scripts.train_rl_six' in command:
                spec = json.loads(Path(command[command.index('--spec') + 1]).read_text())
                stage = command[command.index('--stage') + 1]
                label = spec.get('variant', {}).get('label', '06_history10')
                calls.append((label, stage))
                if label == fail and stage == 'rl':
                    raise RuntimeError('simulated worker failure')
                artifacts, output = Path(spec['artifacts']), Path(spec['output'])
                artifacts.mkdir(parents=True, exist_ok=True)
                if stage == 'preflight':
                    write_json(spec['lengths'], {'sft_limit': 512, 'rl_prompt_limit': 1024, 'completion_limit': 5,
                        'eval_prompt_max': 1024, 'data_audit_sha256': sha256(Path(spec['data']) / 'audit.json')})
                elif stage == 'sft':
                    self.assertEqual(spec['model'], 'Qwen/Qwen3-0.6B')
                    model = output / 'selected_model'
                    validation.CheckpointValidationTests().write_model(model)
                    write_json(artifacts / 'complete.json', {'model': str(model), 'selected_step': 42,
                        'final_step': 50, 'model_sha256': helpers.file_manifest(model)})
                else:
                    self.assertNotIn('--resume', command)
                    schedule = helpers.schedule_spec(55842, spec['variant'])
                    for step in helpers.SNAPSHOTS:
                        checkpoint = output / f'checkpoint-{step}'
                        validation.CheckpointValidationTests().write_model(checkpoint, step)
                        write_json(checkpoint / 'trainer_state.json', {'global_step': step, 'max_steps': schedule['full_scheduler_steps']})
                        (checkpoint / 'reference_model.pt').write_bytes(b'reference fixture')
                    write_json(artifacts / 'schedule.json', dict(schedule, trainer_max_steps=schedule['full_scheduler_steps']))
                    write_json(artifacts / 'training_args.json', spec['variant'])
                return
            self.assertIn('reproduction.evaluate', command)
            self.assertEqual(command[command.index('--split') + 1], 'valid')
            target = Path(command[command.index('--output') + 1])
            data = Path(command[command.index('--data') + 1])
            model = command[command.index('--model') + 1]
            label = target.parent.parent.name if target.parent.name == 'validation' else target.name
            calls.append((label, 'eval'))
            category = command[command.index('--category') + 1]
            catalog = {}
            for item, tokens in json.loads((data / f'{category}.index.json').read_text()).items():
                catalog.setdefault(''.join(tokens), []).append(item)
            predictions = []
            with (data / 'valid.csv').open() as stream:
                for row in csv.DictReader(stream):
                    candidates = list(catalog)[:50]
                    predictions.append(dict(sample_id=row['sample_id'], user_id=row['user_id'],
                        target_position=int(row['target_position']), history_item_id=ast.literal_eval(row['history_item_id']),
                        target_item_id=row['item_id'], target_sid=row['item_sid'], predictions=candidates,
                        predicted_item_id_groups=[catalog[c] for c in candidates], scores=[-i * .1 for i in range(50)]))
            target.mkdir(parents=True, exist_ok=True)
            (target / 'valid.predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in predictions))
            from reproduction.evaluate import metrics
            write_json(target / 'valid.metrics.json', {**metrics(predictions), 'model': model, 'split': 'valid',
                'invalid_candidates': 0, 'generation_config': dict(num_beams=50, num_return_sequences=50,
                    do_sample=False, temperature=1., length_penalty=0., use_cache=True)})
        return launch

    def test_all_six_serial_shared_data_gzip_and_no_repeat_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, source, model, args = self.fixture(tmp)
            before = helpers.file_manifest(source)
            calls = []
            six.run(repo, args, self.gpu(calls))
            training = [x for x in calls if x[1] in ('rl', 'sft', 'preflight')]
            self.assertEqual(training, [(v['label'], 'rl') for v in helpers.variants()[:5]] +
                [('06_history10', 'preflight'), ('06_history10', 'sft'), ('06_history10', 'rl')])
            root = repo / 'results/six'
            rows = helpers.read_json(root / 'summary.json')
            self.assertEqual(len(rows), 24)
            self.assertEqual(len(list(root.rglob('valid.predictions.jsonl.gz'))), 26)
            self.assertEqual(len(list(root.rglob('valid.predictions.jsonl'))), 0)
            self.assertEqual(len(list(root.rglob('train.csv'))), 2)
            self.assertFalse(any(p.name.startswith('test.') or p.is_symlink() or p.suffix in ('.pt', '.safetensors', '.bin') for p in root.rglob('*')))
            models = [helpers.read_json(root / 'experiments' / v['label'] / 'rl_spec.json')['model'] for v in helpers.variants()]
            self.assertEqual(len(set(models[:5])), 1)
            self.assertNotEqual(models[0], models[5])
            self.assertEqual(helpers.file_manifest(source), before)
            self.assertEqual({r['prompt_groups_seen'] for r in rows if r['step'] == 350}, {2800, 22400})
            # Shared copy makes resume independent of the original source directory.
            shutil.rmtree(source)
            calls.clear()
            six.run(repo, args, self.gpu(calls))
            self.assertEqual(calls, [])
            script = repo / 'scripts/package_results.py'
            result = subprocess.run([sys.executable, str(script), 'six'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((repo / 'results/six.tar.gz').exists())

    def test_failure_stops_later_experiments_and_resume_pins(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, _, _, args = self.fixture(tmp)
            calls = []
            with self.assertRaisesRegex(RuntimeError, 'simulated worker failure'):
                six.run(repo, args, self.gpu(calls, fail='02_beta01'))
            self.assertNotIn(('03_batch128', 'rl'), calls)
            self.assertEqual(len(helpers.read_json(repo / 'results/six/summary.json')), 4)
            calls.clear()
            six.run(repo, args, self.gpu(calls))
            self.assertNotIn(('01_lr5e6', 'rl'), calls)
            args.gpus = '0,1,2,3'
            with self.assertRaisesRegex(ValueError, 'settings changed'):
                six.run(repo, args, self.gpu([]))
            args.gpus = '2,3,6,7'
            (repo / 'results/six/shared/history50/data/train.csv').write_text('tamper')
            with self.assertRaisesRegex(ValueError, 'Artifact changed'):
                six.run(repo, args, self.gpu([]))

    def test_final_checkpoint_recovery_no_extra_optimizer_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, _, _, args = self.fixture(tmp)
            args.experiments = [1]
            args.stage = 'train'
            six.run(repo, args, self.gpu([]))
            done = repo / 'results/six/experiments/01_lr5e6/rl/complete.json'
            done.unlink()
            calls = []
            six.run(repo, args, self.gpu(calls))
            self.assertEqual(calls, [])
            self.assertEqual(helpers.read_json(done)['final_step'], 350)

    def test_completed_sft_export_recovery_needs_no_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, artifacts = Path(tmp) / 'sft', Path(tmp) / 'artifacts'
            checkpoint = output / 'checkpoint-42'
            validation.CheckpointValidationTests().write_model(checkpoint, 42)
            write_json(artifacts / 'fit_complete.json', dict(final_step=50, selected_step=42,
                best_checkpoint=str(checkpoint), best_metric=.5,
                model_sha256={name: sha256(checkpoint / name) for name in six.evaluation.model_files(checkpoint)}))
            six.recover_sft_export(dict(output=str(output), artifacts=str(artifacts)))
            done = helpers.read_json(artifacts / 'complete.json')
            self.assertEqual(done['selected_step'], 42)
            self.assertEqual(done['final_step'], 50)
            helpers.verify_files(output / 'selected_model', done['model_sha256'])
            self.assertFalse((output / 'selected_model/trainer_state.json').exists())

    def test_dry_run_no_writes_and_invalid_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, _, _, args = self.fixture(tmp)
            args.dry_run = True
            six.run(repo, args, lambda *a: self.fail('GPU from dry run'))
            self.assertFalse((repo / 'results/six').exists())
            for key, value in [('sft_micro_batch', 0), ('eval_batch_size', 0), ('gpus', '0,0,1,2')]:
                old = getattr(args, key)
                setattr(args, key, value)
                with self.subTest(key=key), self.assertRaises(ValueError):
                    six.run(repo, args)
                setattr(args, key, old)

    def test_history10_preserves_targets_split_and_three_histories(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / 'source.csv', Path(tmp) / 'target.csv'
            row = dict(sample_id='valid:0', user_id='u', item_id='target', item_sid='sid_target',
                target_position='31', history_item_id=repr(list(range(30))),
                history_item_sid=repr([f's{i}' for i in range(30)]),
                history_item_title=repr([f't{i}' for i in range(30)]))
            with src.open('w') as f:
                writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
            report = helpers.write_history10(src, dst)
            self.assertEqual(report, {'rows': 1, 'truncated_rows': 1})
            with dst.open() as f:
                actual = next(csv.DictReader(f))
            for key, value in row.items():
                self.assertEqual(actual[key], repr(ast.literal_eval(value)[-10:]) if key.startswith('history_') else value)

    def test_balanced_exposure_deterministic_no_label_or_group_corruption(self):
        rows = [dict(sample_id=f'train:{t}:{i}', prompt=f'p{t}-{i}', target=f't{t}-{i}')
                for t, size in enumerate((17, 3, 10)) for i in range(size)]
        actual = helpers.balance_rows(rows)
        self.assertEqual(actual, helpers.balance_rows(rows))
        self.assertEqual(Counter(r['sample_id'].split(':')[1] for r in actual), {'0': 10, '1': 10, '2': 10})
        self.assertEqual(len({r['sample_id'] for r in actual}), len(actual))
        originals = {(r['prompt'], r['target']) for r in rows}
        self.assertTrue(all((r['prompt'], r['target']) in originals for r in actual))

    def test_budget_scheduler_and_reference_mapping(self):
        plans = [helpers.schedule_spec(55842, v) for v in helpers.variants()]
        self.assertTrue(all(p['stop_after_steps'] == 350 and p['snapshot_steps'] == [50,100,175,350] for p in plans))
        self.assertEqual(plans[2]['warmup_steps'], 424)
        self.assertEqual(plans[2]['ref_model_sync_steps'], 4096)
        self.assertEqual(plans[2]['full_scheduler_steps'], 13968)
        self.assertEqual(plans[4]['candidates_per_update'], 2048)
        self.assertEqual(plans[4]['prompt_groups_per_update'], 64)
        self.assertEqual(plans[0]['full_scheduler_steps'], 1746)

    def test_console_tail_bounded_and_failure_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code = 'import sys; sys.stdout.write("x"*900000); print("\\nERROR sentinel"); sys.exit(3)'
            with self.assertRaisesRegex(RuntimeError, 'exited 3'):
                helpers.launch([sys.executable, '-c', code], {}, root, root, root / 'run.log')
            self.assertLessEqual((root / 'run.log').stat().st_size, 262144)
            self.assertIn('ERROR sentinel', (root / 'run.warnings.log').read_text())
            self.assertEqual(json.loads((root / 'commands.jsonl').read_text().splitlines()[-1])['exit_code'], 3)


@unittest.skipUnless(HAS_RUNTIME, 'Torch/TRL runtime required')
class SixRuntimeTests(unittest.TestCase):
    def test_four_rank_g32_and_g16_sampler_exposure(self):
        from accelerate.data_loader import BatchSamplerShard
        from torch.utils.data import BatchSampler
        from minionerec_trainer import RepeatRandomSampler
        prefixes = []
        for variant in helpers.variants():
            ranks = []
            for rank in range(4):
                sampler = RepeatRandomSampler(list(range(129)), variant['generations'], seed=42)
                batches = BatchSamplerShard(BatchSampler(sampler, variant['micro_batch'], drop_last=False),
                    num_processes=4, process_index=rank, split_batches=False, even_batches=True)
                ranks.append(list(batches))
            groups = []
            for microstep in range(variant['accumulation']):
                for rank in ranks:
                    batch = rank[microstep]
                    self.assertEqual(len(batch), variant['micro_batch'])
                    for start in range(0, len(batch), variant['generations']):
                        group = batch[start:start + variant['generations']]
                        self.assertEqual(len(set(group)), 1)
                        groups.append(group[0])
            expected = helpers.schedule_spec(129, variant)['prompt_groups_per_update']
            self.assertEqual(len(groups), expected)
            prefixes.append(groups)
        self.assertEqual(prefixes[4], prefixes[0])
        self.assertEqual(prefixes[2], prefixes[0][:8])

    def test_g32_actual_generation_and_reward_backward(self):
        import torch
        from datasets import Dataset
        from trl import GRPOConfig
        from tests.test_reproduction_runtime import RuntimeTests
        from scripts.train_rl_six import reward_functions, trainer_class
        from transformers import set_seed
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / 'model'; base.mkdir()
            tok, model, sids, info = RuntimeTests().fixture(base)
            extra = [f'<a_{i}>' for i in range(20, 40)]
            tok.add_tokens(extra); model.resize_token_embeddings(len(tok))
            tok.save_pretrained(base); model.save_pretrained(base)
            sids += [f'<a_{i}><b_0><c_0>' for i in range(20,40)]
            info.write_text(''.join(f'{s}\titem\t{i}\n' for i,s in enumerate(sids)))
            rows = [{'prompt': '### Response:\n', 'target': sids[i]+'\n', 'sample_id': f'train:0:{i}'} for i in range(2)]
            set_seed(42)
            conf = GRPOConfig(output_dir=str(Path(tmp)/'out'), use_cpu=True, bf16=False,
                per_device_train_batch_size=32, per_device_eval_batch_size=32, num_generations=32,
                optim='adamw_torch', report_to='none', max_completion_length=5, beta=.04,
                model_init_kwargs={'torch_dtype':'float32'}, gradient_checkpointing=False, disable_tqdm=True)
            conf.max_prompt_length=64
            trainer=trainer_class()(model=str(base), base_model=str(base), processing_class=tok, args=conf,
                train_dataset=Dataset.from_list(rows), eval_dataset=Dataset.from_list(rows),
                reward_funcs=reward_functions(32), beam_search=True, test_during_training=False, info_file=str(info))
            trainer.generation_config.do_sample=False
            prepared=trainer._prepare_inputs([row for row in rows for _ in range(32)])
            self.assertEqual(prepared['advantages'].shape[0],64)
            decoded=tok.batch_decode(prepared['completion_ids'], skip_special_tokens=True)
            self.assertEqual([len(set(decoded[i:i+32])) for i in (0,32)],[32,32])
            # The random fixture can miss both original targets, legitimately giving
            # zero advantage/gradient at policy == reference. Choose an actual
            # rank21 beam from each group to exercise a nonzero learning signal.
            for row, target in zip(rows, (decoded[20], decoded[52])):
                row['target'] = target
            prepared=trainer._prepare_inputs([row for row in rows for _ in range(32)])
            self.assertTrue((prepared['advantages'].abs() > 0).any())
            # Correct candidate placed beyond16 must be a hit within its G32 group.
            candidates=[sids[i] for i in range(32)]
            rewards=reward_functions(32)
            ex=rewards[0](['p']*32,candidates,[sids[20]]*32)
            self.assertEqual(ex[20],1.)
            loss=trainer.compute_loss(trainer.model,prepared)
            loss.backward()
            grads=[p.grad for p in trainer.model.parameters() if p.grad is not None]
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertTrue(any(g.abs().sum()>0 for g in grads))

    def test_actual_new_scheduler_stop_and_exact_resume(self):
        import torch
        from datasets import Dataset
        from transformers import TrainerCallback,set_seed
        from trl import GRPOConfig
        from tests.test_reproduction_runtime import RuntimeTests
        from scripts.train_rl_six import trainer_class,reward_functions,stop_callback
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)/'model';base.mkdir()
            tok,_,sids,info=RuntimeTests().fixture(base)
            rows=[dict(prompt='### Response:\n',target=sids[i]+'\n',sample_id=f'train:0:{i}') for i in range(4)]
            class Interrupt(TrainerCallback):
                def on_save(self,args,state,control,**kwargs):
                    if state.global_step==1:raise RuntimeError('interrupted after checkpoint')
            def make(name,interrupt=False):
                set_seed(42)
                output=Path(tmp)/name
                conf=GRPOConfig(output_dir=str(output),use_cpu=True,bf16=False,
                    per_device_train_batch_size=16,per_device_eval_batch_size=16,num_generations=16,
                    gradient_accumulation_steps=2,num_train_epochs=2,max_steps=-1,
                    learning_rate=1e-5,warmup_steps=1,lr_scheduler_type='cosine',
                    optim='adamw_torch',report_to='none',save_strategy='no',eval_strategy='no',
                    max_completion_length=5,beta=.04,model_init_kwargs={'torch_dtype':'float32'},
                    gradient_checkpointing=False,disable_tqdm=True,logging_steps=1,load_best_model_at_end=False)
                conf.max_prompt_length=64
                schedule=dict(stop_after_steps=3,snapshot_steps=[1,3],full_scheduler_steps=8)
                callbacks=[stop_callback(schedule,output/'artifacts')]
                if interrupt:callbacks.append(Interrupt())
                trainer=trainer_class()(model=str(base),base_model=str(base),processing_class=tok,args=conf,
                    train_dataset=Dataset.from_list(rows),eval_dataset=Dataset.from_list(rows[:2]),
                    reward_funcs=reward_functions(16),beam_search=True,test_during_training=False,
                    info_file=str(info),callbacks=callbacks)
                trainer.six_scheduler_steps=8
                trainer.generation_config.do_sample=False
                return trainer
            full=make('full');full.train()
            self.assertEqual(full.state.global_step,3)
            self.assertEqual(full.state.max_steps,4)
            broken=make('resume',True)
            with self.assertRaisesRegex(RuntimeError,'interrupted'):broken.train()
            resumed=make('resume');resumed.train(resume_from_checkpoint=str(Path(tmp)/'resume/checkpoint-1'))
            self.assertEqual(resumed.state.global_step,3)
            for name,value in full.model.state_dict().items():
                torch.testing.assert_close(value,resumed.model.state_dict()[name],rtol=0,atol=0)
            self.assertEqual(full.lr_scheduler.state_dict(),resumed.lr_scheduler.state_dict())
            for name,value in full.ref_model.state_dict().items():
                torch.testing.assert_close(value,resumed.ref_model.state_dict()[name],rtol=0,atol=0)
            with self.assertRaisesRegex(ValueError,'Budget already reached'):
                resumed.train(resume_from_checkpoint=str(Path(tmp)/'resume/checkpoint-3'))


if __name__ == '__main__':
    unittest.main()
