#!/usr/bin/env python3
"""Serial direct-upstream Qwen2.5-0.5B Office/Industrial SFT + RL700."""
import argparse
import csv
import fcntl
import gzip
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reproduction.prepare import sha256, write_json
from scripts.official_utils import (CATEGORIES, SNAPSHOT, SNAPSHOTS, SUFFIX,
    committed_checkpoint, materialize, model_hashes, recover_export)
from scripts.six_utils import file_manifest, launch, pin_json, read_json, verify_files

REPO=Path(__file__).resolve().parents[1]


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name',default='official_qwen25_two700')
    p.add_argument('--model',default='Qwen/Qwen2.5-0.5B')
    p.add_argument('--gpus',default='0,1,2,3')
    p.add_argument('--checkpoint-root',default='checkpoints')
    p.add_argument('--sft-micro-batch',type=int,default=4)
    p.add_argument('--eval-batch-size',type=int,default=2)
    p.add_argument('--datasets',nargs='+',choices=CATEGORIES,default=CATEGORIES)
    p.add_argument('--stage',choices=['all','prepare','train','eval'],default='all')
    p.add_argument('--offline',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    return p


def source_files(repo):
    return ['scripts/run_official_two.py','scripts/train_official_direct.py','scripts/official_utils.py',
            'requirements-official-direct.txt','requirements-reproduction.txt',
            'scripts/package_official_two.py','scripts/six_utils.py','scripts/evaluate_checkpoints.py',
            'reproduction/prepare.py','reproduction/evaluate.py','reproduction/contracts.py',
            'scripts/five_utils.py',
            *[str(p.relative_to(repo)) for p in (repo/'reproduction/official_direct').iterdir() if p.is_file()]]


def settings(repo,args):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',args.run_name):raise ValueError('Use a simple run name')
    devices=args.gpus.split(',')
    if len(devices)!=4 or len(set(devices))!=4 or any(not re.fullmatch(r'\d+|GPU-[A-Za-z0-9-]+',d) for d in devices):
        raise ValueError('Exactly four distinct GPU IDs required')
    if args.sft_micro_batch<=0 or 256%args.sft_micro_batch or args.eval_batch_size<=0:
        raise ValueError('SFT micro must divide256; eval batch must be positive')
    model=args.model
    if 'instruct'in model.lower():raise ValueError('Use Qwen2.5-0.5B Base, not Instruct')
    if Path(model).expanduser().exists():model=str(Path(model).expanduser().resolve())
    weights=Path(args.checkpoint_root).expanduser()
    if not weights.is_absolute():weights=repo/weights
    weights=(weights/args.run_name).resolve()
    for results in [repo/'results',repo/'results2']:
        results=results.resolve()
        if weights==results or results in weights.parents or weights in results.parents:
            raise ValueError('Weights must stay outside results/results2')
    root=repo/'results2'/args.run_name
    config=dict(run_name=args.run_name,model=model,gpus=args.gpus,checkpoint_root=str(weights),
        sft_micro=args.sft_micro_batch,eval_batch_size=args.eval_batch_size,offline=args.offline,
        categories=CATEGORIES,stop_after_steps=700,snapshots=SNAPSHOTS,
        implementation='pinned upstream entrypoints; original reward dictionaries/sampler/decoder',
        evaluation='official evaluate.py + calc.py on validation; single designated GPU per evaluation',
        budget='official two-epoch LR horizon; early stop700; effective candidate batch1024/G16')
    if (root/'suite_config.json').exists() and read_json(root/'suite_config.json')!=config:
        raise ValueError('Settings changed; use saved resume.sh or a new run name')
    return root,config


def setup(repo,root,config):
    if not (root/'suite_config.json').exists() and any(p.name!='.official.lock' for p in root.iterdir()):
        raise ValueError('Nonempty results directory belongs to an unknown run')
    pin_json(root/'suite_config.json',config)
    source=materialize(root/'shared/upstream')
    for category in CATEGORIES:
        data=root/'shared/data'/category;data.mkdir(parents=True,exist_ok=True)
        inputs={f'{split}.csv':repo/f'data/Amazon/{split}/{category}{SUFFIX}.csv' for split in ['train','valid']}
        inputs.update({'info.txt':repo/f'data/Amazon/info/{category}{SUFFIX}.txt',
                       f'{category}.index.json':repo/f'data/Amazon/index/{category}.index.json',
                       f'{category}.item.json':repo/f'data/Amazon/index/{category}.item.json'})
        for name,original in inputs.items():
            target=data/name
            if target.exists():
                if sha256(target)!=sha256(original):raise ValueError(f'Data changed: {target}')
            else:
                temporary=target.with_suffix(target.suffix+'.tmp')
                shutil.copyfile(original,temporary);temporary.replace(target)
        pin_json(data/'manifest.json',dict(source_sha256={name:sha256(p) for name,p in inputs.items()}))
    sources={name:sha256(repo/name) for name in source_files(repo)}
    pin_json(root/'source_sha256.json',sources)
    for name in sources:
        target=root/'shared/launcher_source'/name
        if target.exists():
            if sha256(target)!=sources[name]:raise ValueError(f'Launcher snapshot changed: {name}')
        else:
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repo/name,target)
    weights=Path(config['checkpoint_root'])
    if weights.exists() and not (weights/'suite_identity.json').exists() and any(weights.iterdir()):
        raise ValueError('Nonempty weights directory belongs to an unknown run')
    pin_json(weights/'suite_identity.json',config)
    pin_json(root/'setup_complete.json',dict(upstream_commit=source['upstream']['commit']))


def environment(repo,config):
    env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=config['gpus'],TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled',
        WANDB_DISABLED='true',NCCL_IB_DISABLE='1',HF_HUB_DISABLE_PROGRESS_BARS='1')
    if config['offline']:env.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1')
    env['PYTHONPATH']=str(repo)+(os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    return env


def command(spec,stage,resume=None):
    spec=Path(spec);record=read_json(spec)
    if stage=='rl':
        cmd=[sys.executable,'-m','accelerate.commands.launch','--config_file',
             str(Path(record['upstream'])/'config/zero2_opt.yaml'),'--num_processes','4','--main_process_port','29503']
    elif stage=='sft':cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=4']
    else:cmd=[sys.executable]
    cmd+=['-m','scripts.train_official_direct','--spec',str(spec),'--stage',stage]
    if resume:cmd+=['--resume',resume]
    return cmd


def prediction_metrics(rows,valid,info):
    """Recompute official calc.py metrics, and additionally report all invalid items."""
    import ast
    with Path(valid).open() as f:source=list(csv.DictReader(f))
    if len(rows)!=len(source):raise ValueError('Incomplete official predictions')
    catalog={line.split('\t')[0].strip() for line in Path(info).read_text().splitlines()}
    counts={f'{metric}@{k}':0. for k in [1,3,5,10,20,50] for metric in ['HR','NDCG']}
    invalid=duplicates=cc=0
    for row,expected in zip(rows,source):
        history=', '.join(ast.literal_eval(expected['history_item_sid']))
        prompt=f'Can you predict the next possible item the user may expect, given the following chronological interaction history: {history}'
        if row['input']!=prompt or row['output'].strip(' \n"')!=expected['item_sid']:
            raise ValueError('Official evaluation rows/targets changed')
        predictions=[s.strip('"\n').strip() for s in row['predict']]
        if len(predictions)!=50:raise ValueError('Expected50 beam outputs')
        invalid+=sum(x not in catalog for x in predictions);duplicates+=50-len(set(predictions))
        rank=next((i+1 for i,x in enumerate(predictions) if x==expected['item_sid']),1000000)
        # Upstream CC only counts invalid outputs before the first target hit.
        cc+=sum(x not in catalog for x in predictions[:rank])
        for k in [1,3,5,10,20,50]:
            if rank<=k:
                counts[f'HR@{k}']+=1;counts[f'NDCG@{k}']+=1/math.log2(rank+1)
    return dict(samples=len(rows),split='valid',invalid_candidates=invalid,duplicate_candidates=duplicates,
                official_CC=cc,**{k:v/len(rows) for k,v in counts.items()})


def evaluate(repo,root,config,spec,model,label,launcher,env):
    dest=root/'experiments'/spec['category']/'validation'/label
    dest.mkdir(parents=True,exist_ok=True)
    done=dest/'complete.json'
    if done.exists():verify_files(dest,read_json(done)['files']);return
    provenance=dict(model=str(model),model_sha256=model_hashes(model))
    pin_json(dest/'provenance.json',provenance)
    job=dict(spec,model=str(model),predictions=str(dest/'valid.predictions.json'),
             eval_batch_size=config['eval_batch_size'])
    pin_json(dest/'spec.json',job)
    raw=Path(job['predictions']);compressed=raw.with_suffix('.json.gz')
    if compressed.exists():
        with gzip.open(compressed,'rt') as f:rows=json.load(f)
    else:
        # Any interrupted raw JSON is overwritten only within this new evaluation job.
        evalenv=dict(env,CUDA_VISIBLE_DEVICES=config['gpus'].split(',')[0])
        launcher(command(dest/'spec.json','eval'),evalenv,repo,root,dest/'eval.tail.log')
        rows=read_json(raw)
    metrics=prediction_metrics(rows,spec['valid'],spec['info'])
    if not compressed.exists():
        tmp=compressed.with_suffix('.tmp')
        with gzip.open(tmp,'wt') as f:json.dump(rows,f,ensure_ascii=False,separators=(',',':'))
        with gzip.open(tmp,'rt') as f:
            if json.load(f)!=rows:raise ValueError('Prediction compression mismatch')
        tmp.replace(compressed)
    if raw.exists():raw.unlink()
    write_json(dest/'valid.metrics.json',metrics)
    verify_files(model,provenance['model_sha256'])
    write_json(done,dict(files={p.name:sha256(p) for p in [compressed,dest/'valid.metrics.json',dest/'provenance.json',dest/'spec.json']}))


def finish_rl(spec):
    output,artifacts=Path(spec['output']),Path(spec['artifacts'])
    latest=committed_checkpoint(output,'rl')
    if not latest or Path(latest).name!='checkpoint-700':raise ValueError('Missing completed RL700')
    saved={}
    for step in SNAPSHOTS:
        checkpoint=output/f'checkpoint-{step}'
        marker=read_json(checkpoint/'direct_complete.json')
        if marker['step']!=step:raise ValueError('Snapshot step mismatch')
        for name,digest in model_hashes(checkpoint).items():saved[f'checkpoint-{step}/{name}']=digest
        saved[f'checkpoint-{step}/direct_reference.pt']=sha256(checkpoint/'direct_reference.pt')
    write_json(artifacts/'complete.json',dict(final_step=700,parent_model=spec['model'],checkpoint_files=saved,
        schedule=read_json(artifacts/'schedule.json')))


def summarize(root):
    rows,status=[],[]
    for category in CATEGORIES:
        exp=root/'experiments'/category
        points=[s for s in SNAPSHOTS if (exp/f'validation/checkpoint-{s}/complete.json').exists()]
        status.append(dict(category=category,sft_complete=(exp/'sft/complete.json').exists(),
            rl_complete=(exp/'rl/complete.json').exists(),evaluated_steps=points))
        if not (exp/'validation/sft/complete.json').exists():continue
        base=read_json(exp/'validation/sft/valid.metrics.json')
        for step in points:
            score=read_json(exp/f'validation/checkpoint-{step}/valid.metrics.json')
            rows.append(dict(category=category,step=step,sft_HR10=base['HR@10'],sft_NDCG10=base['NDCG@10'],
                HR10=score['HR@10'],NDCG10=score['NDCG@10'],HR50=score['HR@50'],
                delta_HR10=score['HR@10']-base['HR@10'],delta_NDCG10=score['NDCG@10']-base['NDCG@10'],
                invalid_candidates=score['invalid_candidates'],official_CC=score['official_CC']))
    write_json(root/'summary.json',rows);write_json(root/'status.json',status)
    columns=['category','step','HR10','NDCG10','delta_HR10','delta_NDCG10','HR50','official_CC']
    lines=['# Direct official Qwen2.5 Base — two RL700 runs','',
        'Original upstream reward lookup / sampler / decoder; each dataset compared with its own fresh SFT.',
        'Official full two-epoch LR schedule, stopped at700; validation only. Check invalid_candidates and CC.', '',
        '| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
    for row in rows:lines.append('| '+' | '.join(f'{row[k]:.6f}' if isinstance(row[k],float) else str(row[k]) for k in columns)+' |')
    (root/'summary.md').write_text('\n'.join(lines)+'\n')
    if rows:
        with (root/'summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run(repo,args,launcher=launch):
    repo=Path(repo).resolve();root,config=settings(repo,args)
    selected=[c for c in CATEGORIES if c in args.datasets]
    if args.dry_run:
        print(json.dumps(dict(results=str(root),settings=config,order=selected,
            stages='each: fresh original SFT -> official SFT valid -> RL700 -> seven official valid evaluations'),indent=2))
        return
    root.mkdir(parents=True,exist_ok=True)
    with (root/'.official.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise ValueError('Official suite already running') from exc
        setup(repo,root,config)
        resume=[sys.executable,str(repo/'scripts/run_official_two.py'),'--run-name',config['run_name'],
                '--model',config['model'],'--gpus',config['gpus'],'--checkpoint-root',str(Path(config['checkpoint_root']).parent),
                '--sft-micro-batch',str(config['sft_micro']),'--eval-batch-size',str(config['eval_batch_size'])]
        if config['offline']:resume+=['--offline']
        (root/'resume.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(repo))+'\n'+shlex.join(resume)+'\n')
        summarize(root)
        if args.stage=='prepare':return
        env=environment(repo,config)
        if not (root/'environment.json').exists():
            probe=('import torch,transformers,trl,accelerate,deepspeed,bitsandbytes,fire,wandb,sklearn,json,sys; '
                   'from pathlib import Path; '
                   'assert torch.cuda.is_available() and torch.cuda.device_count()==4; '
                   'assert transformers.__version__=="4.57.1" and trl.__version__=="0.24.0"; '
                   'Path(sys.argv[1]).write_text(json.dumps({x.__name__:x.__version__ for x in '
                   '(torch,transformers,trl,accelerate,deepspeed,bitsandbytes)},indent=2))')
            launcher([sys.executable,'-c',probe,str(root/'environment.json')],env,repo,root,root/'environment.tail.log')
        specs=[]
        for category in selected:
            exp=root/'experiments'/category;exp.mkdir(parents=True,exist_ok=True)
            data=root/'shared/data'/category
            spec=dict(upstream=str(root/'shared/upstream'),category=category,model=config['model'],
                train=str(data/'train.csv'),valid=str(data/'valid.csv'),info=str(data/'info.txt'),
                index=str(data/f'{category}.index.json'),items=str(data/f'{category}.item.json'),
                preflight=str(exp/'preflight.json'),sft_micro=config['sft_micro'],
                artifacts=str(exp/'sft'),output=str(Path(config['checkpoint_root'])/category/'sft'))
            pin_json(exp/'sft_spec.json',spec)
            if not Path(spec['preflight']).exists():
                if args.stage=='eval':raise ValueError('Run preparation/training first')
                launcher(command(exp/'sft_spec.json','preflight'),env,repo,root,exp/'preflight.tail.log')
            pin_json(exp/'preflight_sha256.json',{'sha256':sha256(spec['preflight'])})
            specs.append((exp,spec))
        for exp,spec in specs:
            print('[official] '+spec['category']+' '+args.stage,flush=True)
            try:
                sftdone=exp/'sft/complete.json'
                if not sftdone.exists() and (exp/'sft/fit_complete.json').exists():recover_export(spec)
                if not sftdone.exists():
                    if args.stage=='eval':raise ValueError('SFT not complete')
                    output=Path(spec['output']);checkpoint=committed_checkpoint(output,'sft')
                    if output.exists() and not checkpoint and any(not p.name.startswith('.incomplete-') for p in output.iterdir()):
                        raise ValueError('Nonempty SFT directory lacks a complete checkpoint')
                    launcher(command(exp/'sft_spec.json','sft',checkpoint),env,repo,root,exp/'sft/train.tail.log')
                sft=read_json(sftdone);model=Path(spec['output'])/'selected_model'
                if sft['model']!=str(model):raise ValueError('SFT model path changed')
                verify_files(model,sft['model_sha256'])
                if args.stage in ('all','eval'):evaluate(repo,root,config,spec,model,'sft',launcher,env)
                rl=dict(spec,model=str(model),artifacts=str(exp/'rl'),
                        output=str(Path(config['checkpoint_root'])/spec['category']/'rl'))
                pin_json(exp/'rl_spec.json',rl)
                done=exp/'rl/complete.json';output=Path(rl['output'])
                if done.exists():
                    record=read_json(done)
                    if record['final_step']!=700 or record['parent_model']!=str(model):raise ValueError('RL provenance changed')
                    verify_files(output,record['checkpoint_files'])
                elif args.stage in ('all','train'):
                    checkpoint=committed_checkpoint(output,'rl')
                    step=int(Path(checkpoint).name.split('-')[-1]) if checkpoint else 0
                    if step>700:raise ValueError('RL budget exceeded')
                    if step<700:
                        if output.exists() and not checkpoint and any(not p.name.startswith('.incomplete-') for p in output.iterdir()):
                            raise ValueError('Nonempty RL directory lacks a complete checkpoint')
                        launcher(command(exp/'rl_spec.json','rl',checkpoint),env,repo,root,exp/'rl/train.tail.log')
                    finish_rl(rl)
                else:raise ValueError('RL not complete')
                if args.stage in ('all','eval'):
                    for step in SNAPSHOTS:evaluate(repo,root,config,spec,output/f'checkpoint-{step}',f'checkpoint-{step}',launcher,env)
            finally:summarize(root)
        print('[official] Results: '+str(root/'summary.md'),flush=True)


if __name__=='__main__':run(REPO,parser().parse_args())
