"""Direct upstream fidelity, serial routing, recovery and portable artifacts."""
import ast
import csv
import fcntl
import gzip
from pathlib import Path
import shutil
import tempfile
import unittest

from reproduction.prepare import write_json
from scripts import run_official_two as suite
from scripts import official_utils as helpers
from scripts.package_official_two import package
from scripts.six_utils import file_manifest, read_json
from tests import test_evaluate_checkpoints as checkpoints


class OfficialSuiteTests(unittest.TestCase):
    def fixture(self, tmp):
        repo=Path(tmp)
        for name in suite.source_files(suite.REPO):
            target=repo/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(suite.REPO/name,target)
        for category in helpers.CATEGORIES:
            raw=repo/'data/Amazon'
            index={str(i):[f'<a_{i}>','<b_0>','<c_0>'] for i in range(60)}
            items={str(i):dict(title=f'title{i}',description=f'description{i}') for i in range(60)}
            write_json(raw/f'index/{category}.index.json',index)
            write_json(raw/f'index/{category}.item.json',items)
            info=raw/f'info/{category}{helpers.SUFFIX}.txt';info.parent.mkdir(parents=True,exist_ok=True)
            info.write_text(''.join(''.join(sid)+f'\ttitle{i}\t{i}\n' for i,sid in index.items()))
            for split,target in [('train',0),('valid',1)]:
                path=raw/f'{split}/{category}{helpers.SUFFIX}.csv';path.parent.mkdir(exist_ok=True)
                row=dict(user_id='u',history_item_id='[59]',history_item_sid=repr([''.join(index['59'])]),
                    history_item_title="['title59']",item_id=str(target),item_sid=''.join(index[str(target)]),item_title=f'title{target}')
                with path.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
        return repo,suite.parser().parse_args(['--gpus','2,3,6,7'])

    def checkpoint(self, path, step):
        checkpoints.CheckpointValidationTests().write_model(path,step)
        write_json(path/'trainer_state.json',dict(global_step=step,epoch=step/864))
        (path/'direct_reference.pt').write_bytes(b'reference')
        write_json(path/'direct_complete.json',dict(step=step,stage='rl',files={
            str(p.relative_to(path)):p.stat().st_size for p in path.rglob('*') if p.is_file()}))

    def launcher(self,calls,fail=False):
        def launch(command,env,repo,root,log):
            log.parent.mkdir(parents=True,exist_ok=True);log.write_text('simulated CUDA worker\n')
            if '-c' in command:write_json(command[-1],dict(simulated=True));return
            spec=read_json(command[command.index('--spec')+1]);stage=command[command.index('--stage')+1]
            calls.append((spec['category'],stage))
            if fail and stage=='rl':raise RuntimeError('simulated interruption')
            if stage=='preflight':write_json(spec['preflight'],dict(checked=True))
            elif stage=='sft':
                self.assertEqual(spec['model'],'Qwen/Qwen2.5-0.5B')
                model=Path(spec['output'])/'selected_model'
                checkpoints.CheckpointValidationTests().write_model(model)
                write_json(Path(spec['artifacts'])/'complete.json',dict(model=str(model),model_sha256=helpers.model_hashes(model)))
            elif stage=='rl':
                self.assertEqual(Path(spec['model']).parts[-2:],('sft','selected_model'))
                self.assertIn('accelerate.commands.launch',command)
                for step in helpers.SNAPSHOTS:self.checkpoint(Path(spec['output'])/f'checkpoint-{step}',step)
                write_json(Path(spec['artifacts'])/'schedule.json',dict(full_scheduler_steps=1728,stop_after_steps=700))
            else:
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'2')
                with Path(spec['valid']).open() as f:expected=next(csv.DictReader(f))
                catalog=[s.split('\t')[0] for s in Path(spec['info']).read_text().splitlines()]
                history=', '.join(ast.literal_eval(expected['history_item_sid']))
                write_json(spec['predictions'],[dict(input='Can you predict the next possible item the user may expect, given the following chronological interaction history: '+history,
                    output=expected['item_sid'],predict=catalog[:50])])
        return launch

    def test_serial_fresh_sft_rl700_resume_and_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo,args=self.fixture(tmp);calls=[]
            suite.run(repo,args,self.launcher(calls))
            expected=[(c,'preflight') for c in helpers.CATEGORIES]
            for c in helpers.CATEGORIES:expected.extend([(c,'sft'),(c,'eval'),(c,'rl')]+[(c,'eval')]*7)
            self.assertEqual(calls,expected)
            root=repo/'results2'/args.run_name
            self.assertEqual(len(read_json(root/'summary.json')),14)
            self.assertEqual(len(list(root.rglob('valid.predictions.json.gz'))),16)
            self.assertFalse(list(root.rglob('valid.predictions.json')))
            self.assertFalse(any(p.is_symlink() or p.suffix in ('.pt','.bin','.safetensors') for p in root.rglob('*')))
            calls.clear();suite.run(repo,args,self.launcher(calls));self.assertEqual(calls,[])
            self.assertTrue(package(repo,args.run_name).exists())

    def test_failure_stops_industrial_and_resumes_without_repeating_sft(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo,args=self.fixture(tmp);calls=[]
            with self.assertRaisesRegex(RuntimeError,'interruption'):suite.run(repo,args,self.launcher(calls,True))
            self.assertNotIn((helpers.CATEGORIES[1],'sft'),calls)
            calls.clear();suite.run(repo,args,self.launcher(calls))
            self.assertNotIn((helpers.CATEGORIES[0],'sft'),calls)
            args.sft_micro_batch=8
            with self.assertRaisesRegex(ValueError,'Settings changed'):suite.run(repo,args,self.launcher([]))

    def test_finalize700_without_extra_training_and_raw_prediction_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo,args=self.fixture(tmp);args.stage='train';args.datasets=[helpers.CATEGORIES[0]]
            suite.run(repo,args,self.launcher([]));root=repo/'results2'/args.run_name
            (root/f'experiments/{helpers.CATEGORIES[0]}/rl/complete.json').unlink()
            calls=[];suite.run(repo,args,self.launcher(calls));self.assertEqual(calls,[])
            args.stage='all';dest=root/f'experiments/{helpers.CATEGORIES[0]}/validation/sft'
            dest.mkdir(parents=True);(dest/'valid.predictions.json').write_text('{broken')
            suite.run(repo,args,self.launcher(calls));self.assertEqual(len(calls),8)

    def test_incomplete_checkpoint_isolated_and_committed_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);good=root/'checkpoint-50';self.checkpoint(good,50)
            (root/'checkpoint-100').mkdir()
            self.assertEqual(helpers.committed_checkpoint(root,'rl'),str(good))
            self.assertTrue((root/'.incomplete-checkpoint-100').exists())
            (good/'direct_reference.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):helpers.committed_checkpoint(root,'rl')

    def test_original_algorithms_unchanged_and_source_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime=Path(tmp)/'upstream';helpers.materialize(runtime)
            helpers.materialize(runtime)
            for name in ['LogitProcessor.py','calc.py']:
                self.assertEqual((runtime/name).read_text(),(helpers.SNAPSHOT/(name+'.txt')).read_text())
            before=ast.parse((helpers.SNAPSHOT/'minionerec_trainer.py.txt').read_text())
            after=ast.parse((runtime/'minionerec_trainer.py').read_text())
            def functions(tree):return {n.name:ast.dump(n) for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
            a,b=functions(before),functions(after)
            for name in ['_get_train_sampler','_get_eval_sampler','_prepare_inputs','compute_loss']:
                self.assertEqual(a[name],b[name])
            for tree in [before,after]:
                sampler=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RepeatRandomSampler')
                if tree is before:original=ast.dump(sampler)
                else:self.assertEqual(original,ast.dump(sampler))
            (runtime/'calc.py').write_text('tampered')
            with self.assertRaisesRegex(ValueError,'changed'):helpers.materialize(runtime)

    def test_storage_lock_dryrun_and_catalog_metric_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo,args=self.fixture(tmp);before=file_manifest(repo);args.dry_run=True
            suite.run(repo,args,self.launcher([]));self.assertEqual(file_manifest(repo),before)
            args.dry_run=False;args.stage='prepare';suite.run(repo,args,self.launcher([]))
            root=repo/'results2'/args.run_name
            self.assertFalse(list((root/'shared/data').rglob('test.csv')))
            with (root/'.official.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaisesRegex(ValueError,'running'):package(repo,args.run_name)
            (root/'bad.pt').write_bytes(b'weight')
            with self.assertRaisesRegex(ValueError,'weight'):package(repo,args.run_name)
            args.checkpoint_root='results2/bad'
            with self.assertRaisesRegex(ValueError,'outside'):suite.run(repo,args,self.launcher([]))


if __name__=='__main__':unittest.main()
