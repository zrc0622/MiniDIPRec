"""Five-suite orchestration, original-data fidelity and real small-model lifecycle."""
import ast
import csv
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from reproduction.prepare import CATEGORIES, SUFFIX, sha256, write_json
from scripts import run_rl_five as five
from scripts import five_utils as helpers
from scripts.six_utils import file_manifest, read_json
from tests import test_evaluate_checkpoints as checkpoint_tests
from tests import test_rl_six as six_tests
from tests.test_reproduction_runtime import HAS_RUNTIME


class FiveSuiteTests(unittest.TestCase):
    def fixture(self, tmp):
        repo = Path(tmp)
        for name in five.source_files(five.REPO):
            target = repo / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(five.REPO / name, target)
        for category in CATEGORIES:
            raw = repo / 'data/Amazon'
            index = {str(i): [f'<a_{i}>', '<b_0>', '<c_0>'] for i in range(60)}
            items = {str(i): dict(title=f'title{i}', description=f'description{i}') for i in range(60)}
            write_json(raw / 'index' / f'{category}.index.json', index)
            write_json(raw / 'index' / f'{category}.item.json', items)
            info = raw / 'info' / f'{category}{SUFFIX}.txt'; info.parent.mkdir(parents=True, exist_ok=True)
            info.write_text(''.join(f'{"".join(sid)}\ttitle\t{i}\n' for i, sid in index.items()))
            for split, history, target in [('train', [59], 0), ('valid', [59, 0], 1)]:
                path = raw / split / f'{category}{SUFFIX}.csv'; path.parent.mkdir(exist_ok=True)
                row = dict(user_id='u', history_item_id=repr(history),
                    history_item_sid=repr([''.join(index[str(i)]) for i in history]),
                    history_item_title=repr([items[str(i)]['title'] for i in history]),
                    item_id=str(target), item_sid=''.join(index[str(target)]), item_title=items[str(target)]['title'])
                with path.open('w', newline='') as f:
                    writer=csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
        source = repo / 'results/old_six'
        helpers.prepare_official(repo / 'data/Amazon', source / 'shared/history10/data', CATEGORIES[0])
        model = repo / 'checkpoints/old_six/06_history10/sft/selected_model'
        checkpoint_tests.CheckpointValidationTests().write_model(model)
        write_json(source / 'suite_config.json', dict(dataset=CATEGORIES[0]))
        write_json(source / 'experiments/06_history10/sft/complete.json', dict(
            selected_step=378, final_step=504, model=str(model), model_sha256=file_manifest(model)))
        write_json(source / 'experiments/06_history10/lengths.json', dict(
            data_audit_sha256=sha256(source / 'shared/history10/data/audit.json')))
        write_json(source / 'shared/manifest.json', dict(source_sha256={'data.py':sha256(repo/'data.py')}))
        args = five.parser().parse_args(['--source-suite', str(source), '--gpus', '2,3,6,7'])
        return repo, source, model, args

    def gpu(self, calls, fail=None):
        eval_launcher = six_tests.SixSuiteTests().gpu([])
        def launch(command, env, repo, root, log):
            log.parent.mkdir(parents=True, exist_ok=True); log.write_text('simulated GPU\n')
            if '-c' in command:
                write_json(command[-1], {'fake_environment':True}); return
            if 'scripts.train_rl_five' not in command:
                calls.append(('eval', command[command.index('--model')+1]))
                return eval_launcher(command, env, repo, root, log)
            spec = read_json(command[command.index('--spec')+1]); stage = command[command.index('--stage')+1]
            label = spec['variant']['label']; calls.append((label, stage))
            if label == fail and stage == 'rl':
                raise RuntimeError('simulated failure')
            if stage == 'preflight':
                write_json(spec['lengths'], dict(recipe=spec['variant']['recipe'],
                    data_audit_sha256=sha256(Path(spec['data'])/'audit.json'), sft_limit=512,
                    rl_prompt_limit=2048, eval_prompt_max=2048, completion_limit=5))
            elif stage == 'sft':
                self.assertEqual(spec['variant']['recipe'], 'official')
                self.assertEqual(spec['model'], 'Qwen/Qwen3-0.6B' if spec['variant']['family']=='qwen3' else 'Qwen/Qwen2.5-0.5B')
                model = Path(spec['output'])/'selected_model'
                checkpoint_tests.CheckpointValidationTests().write_model(model)
                write_json(Path(spec['artifacts'])/'complete.json', dict(model=str(model), selected_step=378,
                    final_step=504, model_sha256=file_manifest(model)))
            else:
                schedule = helpers.schedule_spec(55842)
                self.assertNotIn('--resume', command)
                for step in helpers.SNAPSHOTS:
                    checkpoint = Path(spec['output'])/f'checkpoint-{step}'
                    checkpoint_tests.CheckpointValidationTests().write_model(checkpoint, step)
                    write_json(checkpoint/'trainer_state.json', dict(global_step=step,max_steps=schedule['full_scheduler_steps']))
                    (checkpoint/'reference_model.pt').write_bytes(b'reference')
                    write_json(checkpoint/'five_complete.json', dict(stage='rl',step=step,files=helpers.checkpoint_manifest(checkpoint)))
                write_json(Path(spec['artifacts'])/'schedule.json',dict(schedule,trainer_max_steps=schedule['full_scheduler_steps']))
        return launch

    def test_serial_five_results2_sources_immutable_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, source, model, args = self.fixture(tmp)
            old = file_manifest(repo/'results'); old_model=file_manifest(model); calls=[]
            five.run(repo,args,self.gpu(calls))
            trained=[c for c in calls if c[1] in ('rl','sft')]
            expected=[]
            for v in helpers.variants():
                if v['recipe']=='official': expected.append((v['label'],'sft'))
                expected.append((v['label'],'rl'))
            self.assertEqual(trained,expected)
            root=repo/'results2/minionerec_five350'
            self.assertEqual(len(read_json(root/'summary.json')),20)
            self.assertEqual(len(list(root.rglob('valid.predictions.jsonl.gz'))),24)
            self.assertFalse(list(root.rglob('valid.predictions.jsonl')))
            self.assertEqual(file_manifest(repo/'results'),old)
            self.assertEqual(file_manifest(model),old_model)
            self.assertFalse(any(p.is_symlink() or p.suffix in ('.pt','.bin','.safetensors') for p in root.rglob('*')))
            specs=[read_json(root/'experiments'/v['label']/'rl_spec.json') for v in helpers.variants()]
            self.assertEqual(specs[0]['model'],specs[1]['model'])
            self.assertEqual(len({s['model'] for s in specs}),4)
            shutil.rmtree(source); calls.clear()
            five.run(repo,args,self.gpu(calls)); self.assertEqual(calls,[])
            from scripts.package_results2 import package
            self.assertTrue(package(repo,args.run_name).exists())

    def test_failure_stops_then_resume_and_config_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo,_,_,args=self.fixture(tmp);calls=[]
            with self.assertRaisesRegex(RuntimeError,'simulated failure'):
                five.run(repo,args,self.gpu(calls,fail='02_history10_adamw'))
            self.assertNotIn(('03_official_qwen3_office','sft'),calls)
            calls.clear();five.run(repo,args,self.gpu(calls))
            self.assertNotIn(('01_history10_restart','rl'),calls)
            args.qwen25_model='different'
            with self.assertRaisesRegex(ValueError,'settings changed'):five.run(repo,args,self.gpu([]))

    def test_final_checkpoint_and_incomplete_prediction_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo,_,_,args=self.fixture(tmp);args.experiments=[1];args.stage='train'
            five.run(repo,args,self.gpu([]))
            root=repo/'results2/minionerec_five350'
            (root/'experiments/01_history10_restart/rl/complete.json').unlink()
            calls=[];five.run(repo,args,self.gpu(calls));self.assertEqual(calls,[])
            args.stage='all'
            baseline=root/'baselines/imported_sft10';baseline.mkdir(exist_ok=True)
            (baseline/'valid.predictions.jsonl').write_text('{partial')
            write_json(baseline/'valid.metrics.json',{})
            calls=[];five.run(repo,args,self.gpu(calls));self.assertEqual(len(calls),5)

    def test_checkpoint_commit_markers_and_packaging_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'rl'; good=output/'checkpoint-50'
            checkpoint_tests.CheckpointValidationTests().write_model(good,50)
            (good/'reference_model.pt').write_bytes(b'ref')
            write_json(good/'five_complete.json',dict(step=50,stage='rl',files=helpers.checkpoint_manifest(good)))
            broken=output/'checkpoint-100';broken.mkdir();write_json(broken/'trainer_state.json',dict(global_step=100))
            self.assertEqual(helpers.resumable_checkpoint(output,'rl'),str(good))
            self.assertTrue((output/'.incomplete-checkpoint-100').exists())
            (good/'reference_model.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):helpers.resumable_checkpoint(output,'rl')
            root=Path(tmp)/'results2/run';root.mkdir(parents=True)
            from scripts.package_results2 import package
            with (root/'.five.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaisesRegex(ValueError,'running'):package(tmp,'run')
            (root/'bad.pt').write_bytes(b'weight')
            with self.assertRaisesRegex(ValueError,'Weights'):package(tmp,'run')

    def test_original_data_no_test_dependency_and_dryrun_no_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo,_,_,args=self.fixture(tmp)
            prepared=repo/'prepared'
            helpers.prepare_official(repo/'data/Amazon',prepared,CATEGORIES[1])
            self.assertFalse((prepared/'test.csv').exists())
            with (prepared/'valid.csv').open() as f: row=next(csv.DictReader(f))
            self.assertEqual(ast.literal_eval(row['history_item_id']),[59,0])
            before=file_manifest(repo);args.dry_run=True
            five.run(repo,args,self.gpu([]));self.assertEqual(file_manifest(repo),before)
            args.checkpoint_root='results2/bad'
            with self.assertRaisesRegex(ValueError,'separate'):five.run(repo,args,self.gpu([]))


@unittest.skipUnless(HAS_RUNTIME,'Runtime dependencies unavailable')
class FiveRuntimeTests(unittest.TestCase):
    def setUp(self):
        # Optional MLX is installed locally but Metal is unavailable in sandbox.
        # Exclude it from generic tensor detection in these PyTorch CPU tests.
        guard=patch('transformers.utils.import_utils._mlx_available',False)
        guard.start();self.addCleanup(guard.stop)

    def test_config_isolation_and_parameter_caps(self):
        from scripts.train_rl_five import training_config
        from transformers import Qwen2Config,Qwen3Config
        from trl import GRPOConfig
        vs=helpers.variants();lengths=dict(completion_limit=5,rl_prompt_limit=1792)
        with tempfile.TemporaryDirectory() as tmp:
            with patch('trl.GRPOConfig',side_effect=lambda **kwargs:GRPOConfig(**kwargs,use_cpu=True)):
                configs=[training_config(tmp,lengths,v,helpers.schedule_spec(55842)).to_dict() for v in vs]
            differing={k for k in configs[0] if configs[0][k]!=configs[1][k]}
            self.assertEqual(differing,{'optim'})
            self.assertEqual(configs[2]['beta'],.001)
            self.assertIsNone(configs[2]['model_init_kwargs'])
            self.assertEqual(configs[0]['model_init_kwargs'],{'torch_dtype':'bfloat16'})
            self.assertEqual(configs[2]['warmup_steps'],53)
        q3=Qwen3Config(hidden_size=1024,intermediate_size=3072,num_hidden_layers=28,
                      num_attention_heads=16,num_key_value_heads=8,head_dim=128,tie_word_embeddings=True)
        q2=Qwen2Config(hidden_size=896,intermediate_size=4864,num_hidden_layers=24,
                      num_attention_heads=14,num_key_value_heads=2,tie_word_embeddings=True)
        for cfg,family in [(q3,'qwen3'),(q2,'qwen2')]:
            self.assertLess(helpers.model_contract(cfg,family,152300)['parameters_after_sid_extension'],1_000_000_000)
        q2.num_hidden_layers=48
        with self.assertRaisesRegex(ValueError,'architecture'):helpers.model_contract(q2,'qwen2',152300)

    def test_scalar_log_resume_and_qwen2_actual_training(self):
        import torch
        from datasets import Dataset
        from transformers import Qwen2Config,Qwen2ForCausalLM,TrainerState,TrainerControl,TrainerCallback,set_seed
        from trl import GRPOConfig
        from scripts.train_rl_five import artifact_callback
        from scripts.train_rl_six import trainer_class,reward_functions,stop_callback
        from tests.test_reproduction_runtime import RuntimeTests
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)/'base';base.mkdir()
            tok,_,sids,info=RuntimeTests().fixture(base)
            cfg=Qwen2Config(vocab_size=len(tok),hidden_size=32,intermediate_size=64,num_hidden_layers=1,
                num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=1024,
                eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,tie_word_embeddings=True)
            Qwen2ForCausalLM(cfg).save_pretrained(base)
            rows=[dict(prompt='### Response:\n',target=sids[i]+'\n',sample_id=f'train:0:{i}') for i in range(4)]
            class Interrupt(TrainerCallback):
                def on_save(self,args,state,control,**kwargs):
                    if state.global_step==1:raise RuntimeError('intentional save interruption')
            def make(name,interrupt=False):
                set_seed(42)
                output=Path(tmp)/name;artifacts=output/'artifacts'
                conf=GRPOConfig(output_dir=str(output),use_cpu=True,bf16=False,optim='adamw_torch',
                    per_device_train_batch_size=16,per_device_eval_batch_size=16,num_generations=16,
                    num_train_epochs=2,gradient_accumulation_steps=2,max_steps=-1,learning_rate=1e-5,
                    warmup_steps=1,save_strategy='no',eval_strategy='no',report_to='none',logging_steps=1,
                    max_completion_length=5,beta=.001,model_init_kwargs=None,gradient_checkpointing=False,disable_tqdm=True)
                conf.max_prompt_length=64
                schedule=dict(stop_after_steps=2,snapshot_steps=[1,2],full_scheduler_steps=4)
                callbacks=[artifact_callback(artifacts),stop_callback(schedule,artifacts)]
                if interrupt:callbacks.append(Interrupt())
                trainer=trainer_class()(model=str(base),base_model=str(base),processing_class=tok,args=conf,
                    train_dataset=Dataset.from_list(rows),eval_dataset=Dataset.from_list(rows),
                    reward_funcs=reward_functions(16),beam_search=True,test_during_training=False,info_file=str(info),
                    callbacks=callbacks)
                trainer.six_scheduler_steps=4
                trainer.generation_config.do_sample=True
                return trainer
            trainer=make('out');trainer.train()
            self.assertEqual(trainer.state.global_step,2)
            latest=helpers.resumable_checkpoint(Path(tmp)/'out','rl')
            self.assertTrue(latest.endswith('checkpoint-2'))
            logs=Path(tmp)/'out/artifacts/metrics.jsonl'
            with logs.open('ab') as f:f.write(b'\x00'*20)
            callback=artifact_callback(logs.parent)
            callback.on_train_begin(trainer.args,TrainerState(global_step=1,is_world_process_zero=True),TrainerControl())
            records=[json.loads(line) for line in logs.read_text().splitlines()]
            self.assertTrue(all(r['step']<=1 for r in records))
            self.assertTrue(all(torch.isfinite(p).all() for p in trainer.model.parameters()))
            interrupted=make('resume',True)
            with self.assertRaisesRegex(RuntimeError,'intentional'):interrupted.train()
            resume=helpers.resumable_checkpoint(Path(tmp)/'resume','rl')
            continued=make('resume');continued.train(resume_from_checkpoint=resume)
            self.assertEqual(continued.state.global_step,2)
            for name,value in trainer.model.state_dict().items():
                torch.testing.assert_close(value,continued.model.state_dict()[name],rtol=0,atol=0)
            for name,value in trainer.ref_model.state_dict().items():
                torch.testing.assert_close(value,continued.ref_model.state_dict()[name],rtol=0,atol=0)
            self.assertEqual(trainer.lr_scheduler.state_dict(),continued.lr_scheduler.state_dict())

    def test_official_metadata_dedup_differs_from_current(self):
        from scripts.five_utils import official_data_module
        import data as current
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            write_json(root/'index.json',{'1':['<a_0>','<b_0>','<c_0>'],'2':['<a_1>','<b_0>','<c_0>']})
            write_json(root/'items.json',{'1':{'title':'same','description':'same'},'2':{'title':'same','description':'same'}})
            official=official_data_module().RLTitle2SidDataset(str(root/'items.json'),str(root/'index.json'))
            adapted=current.RLTitle2SidDataset(str(root/'items.json'),str(root/'index.json'))
            self.assertEqual(len(official),2)
            self.assertEqual(len(adapted),4)


if __name__=='__main__':
    unittest.main()
