"""Run pinned upstream entrypoints with storage/stop instrumentation only."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys

from reproduction.prepare import sha256, write_json
from scripts.official_utils import SNAPSHOTS, model_hashes
from scripts.six_utils import read_json, verify_files


def upstream(spec):
    root = Path(spec['upstream']).resolve()
    verify_files(root, read_json(root / 'direct_manifest.json')['files'])
    sys.path.insert(0, str(root))
    for name in ('data', 'minionerec_trainer', 'LogitProcessor', 'sft', 'rl', 'evaluate', 'calc'):
        if name in sys.modules and Path(sys.modules[name].__file__).parent != root:
            raise ValueError(f'Local module would shadow official source: {name}')
    return root


def preflight(spec):
    from transformers import AutoConfig, AutoTokenizer
    from scripts.five_utils import model_contract
    data = importlib.import_module('data')
    tokenizer = AutoTokenizer.from_pretrained(spec['model'], padding_side='left')
    cfg = AutoConfig.from_pretrained(spec['model'])
    indices = read_json(spec['index'])
    sid_tokens = sorted({t for v in indices.values() for t in v})
    if set(sid_tokens) & tokenizer.get_added_vocab().keys():
        raise ValueError('Use original Qwen2.5-0.5B Base, not an existing recommendation SFT')
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_tokens(sid_tokens)
    contract = model_contract(cfg, 'qwen2', len(tokenizer))
    prefix = tokenizer.encode('### Response:\n', add_special_tokens=False)
    if len(prefix) != 3 or tokenizer.decode(prefix) != '### Response:\n':
        raise ValueError('Tokenizer does not match the official three-token prefix contract')
    if tokenizer.eos_token_id != 151643:
        raise ValueError('Expected Qwen2.5 Base EOS')
    for sid in indices.values():
        ids = tokenizer.encode(''.join(sid) + '\n', add_special_tokens=False)
        if len(ids) != 4 or tokenizer.decode(ids) != ''.join(sid) + '\n':
            raise ValueError('Official SID encoding mismatch')
    category = {'Office_Products':'office products',
                'Industrial_and_Scientific':'industrial and scientific items'}[spec['category']]
    common = dict(tokenizer=tokenizer, max_len=cfg.max_position_embeddings, seed=42, category=category)
    sft = [data.SidSFTDataset(spec['train'], **common),
           data.SidItemFeatDataset(item_file=spec['items'], index_file=spec['index'], **common),
           data.FusionSeqRecDataset(spec['train'], item_file=spec['items'], index_file=spec['index'], **common),
           data.SidSFTDataset(spec['valid'], **common)]
    sft_lengths = [max(len(row['input_ids']) for row in d) for d in sft]
    sft_counts = [len(d) for d in sft]; del sft
    if max(sft_lengths) > 512:
        raise ValueError('Official SFT512 would truncate this dataset')
    rl = [data.SidDataset(spec['train'], category=category),
          data.RLTitle2SidDataset(spec['items'], spec['index'], category=category),
          data.RLSeqTitle2SidDataset(spec['train'], sample=10000, category=category)]
    valid = data.SidDataset(spec['valid'], category=category)
    maxima = [max(len(tokenizer.encode(row['prompt'])) for row in d) for d in rl + [valid]]
    if max(maxima) + 128 > cfg.max_position_embeddings:
        raise ValueError('Original RL prompts exceed model context')
    # Report original dictionary collisions, but deliberately preserve their targets.
    prompt2history, history2target = {}, {}
    for d in rl + [valid]:
        prompt2history.update(d.prompt2history)
        history2target.update(d.history2target)
    mismatches = [sum(history2target[prompt2history[row['prompt']]] != row['completion'] for row in d)
                  for d in rl + [valid]]
    evaluation = data.EvalSidDataset(spec['valid'], tokenizer, test=True, max_len=2560, category=category)
    evaluation_max = max(len(row['input_ids']) for row in evaluation)
    if evaluation_max + 256 > cfg.max_position_embeddings:
        raise ValueError('Official evaluation exceeds model context')
    write_json(spec['preflight'], dict(model=spec['model'], model_contract=contract,
        sft_task_counts=sft_counts, sft_max_lengths=sft_lengths, rl_task_counts=[len(d) for d in rl],
        rl_prompt_max_lengths=maxima, valid_rows=len(valid), eval_max_length=evaluation_max,
        official_reward_target_mismatches_by_task_and_valid=mismatches,
        tokenizer=dict(vocab=len(tokenizer), eos=tokenizer.eos_token_id, sid_tokens=len(sid_tokens)),
        policy='Original data classes, prompt, dictionary targets and constrained decoder; no task edits'))
    print('[official] preflight PASS', contract, flush=True)


def callbacks(spec, stage, tokenizer):
    import torch
    from transformers import TrainerCallback

    class Artifacts(TrainerCallback):
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            if stage == 'rl' and (state.global_step >= 700 or state.max_steps <= 700):
                raise ValueError('Expected a full official schedule >700; completed700 must only be finalized')
            if state.is_world_process_zero:
                artifacts = Path(spec['artifacts']); artifacts.mkdir(parents=True, exist_ok=True)
                logs = artifacts / 'metrics.jsonl'
                kept = []
                if logs.exists():
                    lines = logs.read_bytes().splitlines()
                    for i,line in enumerate(lines):
                        try: row = json.loads(line)
                        except (ValueError, UnicodeDecodeError):
                            if i != len(lines)-1: raise
                            break
                        if row['step'] <= state.global_step:
                            kept.append(json.dumps(row))
                    logs.write_text(''.join(line+'\n' for line in kept))
                write_json(artifacts / 'schedule.json', dict(full_scheduler_steps=state.max_steps,
                    warmup_steps=args.get_warmup_steps(state.max_steps), ref_model_sync_steps=getattr(args,'ref_model_sync_steps',None),
                    stop_after_steps=700 if stage=='rl' else None,
                    snapshot_steps=SNAPSHOTS if stage=='rl' else None,
                    first_step=state.global_step, model_dtype_after_prepare=str(next(model.parameters()).dtype)))

        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.is_world_process_zero:
                row = dict(logs or {})
                row.update(step=state.global_step, epoch=state.epoch)
                with (Path(spec['artifacts']) / 'metrics.jsonl').open('a') as f:
                    f.write(json.dumps(row) + '\n')
                if state.global_step % 10 == 0 or any(k.startswith('eval_') for k in (logs or {})):
                    print('[six] official ' + json.dumps(row), flush=True)

        def on_step_end(self, args, state, control, **kwargs):
            if stage == 'rl':
                if state.global_step in SNAPSHOTS: control.should_save = True
                if state.global_step >= 700:
                    control.should_training_stop = True; control.should_save = True
            return control

        def on_save(self, args, state, control, **kwargs):
            import torch.distributed as dist
            if dist.is_initialized(): dist.barrier()
            if state.is_world_process_zero:
                checkpoint = Path(args.output_dir) / f'checkpoint-{state.global_step}'
                tokenizer.save_pretrained(checkpoint)
                sizes = {str(p.relative_to(checkpoint)):p.stat().st_size for p in checkpoint.rglob('*')
                         if p.is_file() and p.name != 'direct_complete.json'}
                write_json(checkpoint / 'direct_complete.json', dict(stage=stage,step=state.global_step,files=sizes))
            if dist.is_initialized(): dist.barrier()

        def on_train_end(self, args, state, control, **kwargs):
            if state.is_world_process_zero and stage == 'sft':
                best = Path(state.best_model_checkpoint)
                write_json(Path(spec['artifacts']) / 'fit_complete.json', dict(final_step=state.global_step,
                    selected_step=int(best.name.split('-')[-1]), best_checkpoint=str(best),
                    best_metric=state.best_metric, model_sha256=model_hashes(best)))
    return Artifacts()


def instrument(base, spec, stage, tokenizer):
    """No override of upstream sampler, generation, reward or compute_loss."""
    import torch

    class DirectTrainer(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.add_callback(callbacks(spec, stage, tokenizer))
            if self.is_world_process_zero():
                write_json(Path(spec['artifacts'])/'training_args.json', self.args.to_dict())
                record=dict(stage=stage, class_name=f'{base.__module__}.{base.__name__}',
                    training_rows=len(self.train_dataset), valid_rows=len(self.eval_dataset))
                if stage=='rl':
                    record.update(generation_config=self.generation_config.to_dict(),
                        ref_dtype_before_train=str(next(self.ref_model.parameters()).dtype),
                        data_order_sha256=hashlib.sha256(json.dumps(self.train_dataset.to_dict(),ensure_ascii=False).encode()).hexdigest(),
                        policy='upstream sampler, lookup targets, compute_loss and constrained logits processor unchanged')
                write_json(Path(spec['artifacts'])/'runtime.json',record)

        def _save_checkpoint(self, model, trial):
            super()._save_checkpoint(model, trial)
            if stage=='rl' and self.is_world_process_zero():
                checkpoint=Path(self.args.output_dir)/f'checkpoint-{self.state.global_step}'
                ref=self.accelerator.unwrap_model(self.ref_model)
                tmp=checkpoint/'direct_reference.pt.tmp'
                torch.save({k:v.detach().cpu() for k,v in ref.state_dict().items()},tmp)
                tmp.replace(checkpoint/'direct_reference.pt')
            self.accelerator.wait_for_everyone()

        def train(self, *args, **kwargs):
            resume=spec.get('resume')
            if resume:
                # Both datasets reach700 within epoch1. Reconstruct original epoch1
                # sampler, then let Trainer skip consumed microbatches and restore RNG.
                state=read_json(Path(resume)/'trainer_state.json')
                if stage=='rl':
                    if state['epoch'] >= 1:
                        raise ValueError('Original mutable sampler resume is only supported within epoch1')
                    self.accelerator.unwrap_model(self.ref_model).load_state_dict(
                        torch.load(Path(resume)/'direct_reference.pt',map_location='cpu',weights_only=True))
                kwargs['resume_from_checkpoint']=resume
            return super().train(*args,**kwargs)
    return DirectTrainer


def train(spec, stage):
    import torch
    import transformers
    from transformers import AutoTokenizer
    if int(os.environ.get('WORLD_SIZE','1')) != 4 or torch.cuda.device_count() != 4:
        raise ValueError('Exactly four visible CUDA GPUs and ranks required')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    Path(spec['artifacts']).mkdir(parents=True,exist_ok=True)
    module=importlib.import_module(stage)
    tok=AutoTokenizer.from_pretrained(spec['model'],padding_side='left');tok.pad_token=tok.eos_token
    if stage=='sft':
        tok.add_tokens(sorted({t for v in read_json(spec['index']).values() for t in v}))
        module.transformers.Trainer=instrument(transformers.Trainer,spec,stage,tok)
    else:
        module.ReReTrainer=instrument(module.ReReTrainer,spec,stage,tok)

    def finish(trainer, tokenizer, output):
        if stage=='sft':
            selected=Path(output)/'selected_model'
            trainer.save_model(str(selected))
            if trainer.is_world_process_zero():
                tokenizer.save_pretrained(selected)
                fit=read_json(Path(spec['artifacts'])/'fit_complete.json')
                write_json(Path(spec['artifacts'])/'complete.json',dict(fit,model=str(selected),model_sha256=model_hashes(selected)))
        elif trainer.state.global_step != 700:
            raise ValueError('Official RL did not complete700 optimizer updates')
        trainer.accelerator.wait_for_everyone()
    module._direct_finish=finish
    common=dict(train_file=spec['train'],eval_file=spec['valid'],category=spec['category'],
        sid_index_path=spec['index'],item_meta_path=spec['items'],output_dir=spec['output'],
        seed=42,wandb_project='official_direct',wandb_run_name=spec['category']+'_'+stage)
    if stage=='sft':
        module.train(base_model=spec['model'],batch_size=1024,micro_batch_size=spec['sft_micro'],**common)
    else:
        module.train(model_path=spec['model'],info_file=spec['info'],train_batch_size=16,eval_batch_size=16,
            gradient_accumulation_steps=16,num_train_epochs=2,eval_step=.0999,reward_type='ranking',
            num_generations=16,mask_all_zero=False,dynamic_sampling=False,sync_ref_model=True,
            beam_search=True,test_during_training=False,temperature=1.,learning_rate=1e-5,
            add_gt=False,beta=.001,dapo=False,**common)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',required=True)
    parser.add_argument('--stage',choices=['preflight','sft','rl','eval'],required=True)
    parser.add_argument('--resume')
    args=parser.parse_args();spec=read_json(args.spec)
    if args.resume:spec['resume']=args.resume
    upstream(spec)
    if args.stage=='preflight':preflight(spec)
    elif args.stage=='eval':
        module=importlib.import_module('evaluate')
        module.main(base_model=spec['model'],info_file=spec['info'],category=spec['category'],
            test_data_path=spec['valid'],result_json_data=spec['predictions'],
            batch_size=spec['eval_batch_size'],num_beams=50,max_new_tokens=256,length_penalty=0.)
        importlib.import_module('calc').gao(spec['predictions'],spec['info'])
    else:train(spec,args.stage)


if __name__=='__main__':main()
