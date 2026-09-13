"""CPU smoke of actual pinned entrypoints; requires a real Qwen2.5 Base tokenizer.

Run as a separate process, keeping upstream's top-level module names isolated.
Only the CPU test substitutes optimizer/precision/budget and optional CLI imports.
"""
import argparse
import ast
import contextlib
import csv
import importlib
import importlib.machinery
import io
import json
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch


def main(tokenizer_path):
    import transformers.utils.import_utils as availability
    availability._mlx_available=False  # Local optional MLX has no Metal device.
    import torch
    from transformers import AutoTokenizer,Qwen2Config,Qwen2ForCausalLM,TrainerCallback,TrainerState,TrainerControl,set_seed
    from trl import GRPOConfig
    from reproduction.prepare import write_json
    from scripts.official_utils import materialize,committed_checkpoint
    from scripts.train_official_direct import upstream,instrument,callbacks
    from scripts.run_official_two import prediction_metrics
    torch.set_num_threads(1)
    # fire and bnb are unused imports in the official CPU execution paths below.
    # Real CUDA launcher requires their real packages, never these test shims.
    for name in ['fire','bitsandbytes','wandb']:
        if importlib.util.find_spec(name) is None:
            shim=types.ModuleType(name);shim.__spec__=importlib.machinery.ModuleSpec(name,loader=None)
            if name=='fire':shim.Fire=lambda *a,**k:None
            sys.modules[name]=shim
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp);runtime=root/'upstream';materialize(runtime);upstream(dict(upstream=str(runtime)))
        from tests.test_official_two import OfficialSuiteTests
        repo,_=OfficialSuiteTests().fixture(root/'fixture')
        upstream(dict(upstream=str(runtime)))
        data=repo/'data/Amazon';category='Office_Products'
        # Use the fixture's exact suffix, not external train/test data.
        train_path=next((data/'train').glob('Office*'));valid_path=next((data/'valid').glob('Office*'))
        with train_path.open() as f:rows=list(csv.DictReader(f))
        original=rows[0]
        with train_path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(original));w.writeheader()
            for i in range(10000):w.writerow(dict(original,user_id=str(i)))
        tok=AutoTokenizer.from_pretrained(tokenizer_path,padding_side='left');tok.pad_token=tok.eos_token
        index=json.loads((data/f'index/{category}.index.json').read_text())
        tok.add_tokens(sorted({t for sid in index.values() for t in sid}))
        assert len(tok.encode('### Response:\n'))==3
        base=root/'base';tok.save_pretrained(base)
        cfg=Qwen2Config(vocab_size=len(tok),hidden_size=16,intermediate_size=32,num_hidden_layers=1,
            num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=1024,
            eos_token_id=tok.eos_token_id,pad_token_id=tok.pad_token_id,tie_word_embeddings=True)
        set_seed(42);Qwen2ForCausalLM(cfg).save_pretrained(base)
        info=str(next((data/'info').glob('Office*')))
        rl_info=root/'rl_info.txt'
        rl_info.write_text(''.join(Path(info).read_text().splitlines(True)[:16]))
        official=importlib.import_module('rl');original_trainer=official.ReReTrainer
        created=[];generations=[]
        class Budget(TrainerCallback):
            def on_step_end(self,args,state,control,**kwargs):
                control.should_save=True
                if state.global_step==2:control.should_training_stop=True
                return control
        class Interrupt(TrainerCallback):
            def on_save(self,args,state,control,**kwargs):
                if state.global_step==1:raise RuntimeError('test saved interruption')
        def cpu_config(**kwargs):
            kwargs.update(use_cpu=True,bf16=False,optim='adamw_torch',gradient_checkpointing=False,
                per_device_train_batch_size=16,per_device_eval_batch_size=16,gradient_accumulation_steps=1,
                eval_strategy='no',save_strategy='no',max_completion_length=8,ref_model_sync_steps=1,
                warmup_steps=0,warmup_ratio=0.)
            return GRPOConfig(**kwargs)
        def run(name,resume=None,interrupt=False):
            output=root/name;artifacts=output/'artifacts';artifacts.mkdir(parents=True,exist_ok=True)
            spec=dict(output=str(output),artifacts=str(artifacts),resume=resume)
            def factory(*args,**kwargs):
                trainer=instrument(original_trainer,spec,'rl',tok)(*args,**kwargs)
                trainer.add_callback(Budget())
                if interrupt:trainer.add_callback(Interrupt())
                created.append(trainer)
                original_generate=trainer.model.generate
                def generation(*args,**kwargs):
                    config=kwargs['generation_config']
                    assert config.do_sample and config.num_beams==16 and config.temperature==1.
                    outputs=original_generate(*args,**kwargs)
                    completions=tok.batch_decode(outputs[:,args[0].shape[1]:],skip_special_tokens=True)
                    assert len(completions)==16 and len(set(completions))==16
                    catalog={''.join(sid) for sid in index.values()}
                    assert all(c.strip() in catalog for c in completions)
                    generations.append(completions)
                    return outputs
                trainer.model.generate=generation
                return trainer
            def finish(trainer,tokenizer,output):assert trainer.state.global_step==2
            with patch.object(official,'GRPOConfig',side_effect=cpu_config),patch.object(official,'ReReTrainer',side_effect=factory):
                official._direct_finish=finish
                official.train(model_path=str(base),train_file=str(train_path),eval_file=str(valid_path),
                    info_file=str(rl_info),category=category,output_dir=str(output),
                    sid_index_path=str(data/f'index/{category}.index.json'),item_meta_path=str(data/f'index/{category}.item.json'),
                    train_batch_size=16,eval_batch_size=16,gradient_accumulation_steps=16,num_train_epochs=2,
                    num_generations=16,reward_type='ranking',beam_search=True,test_during_training=False,
                    sync_ref_model=True,learning_rate=1e-5,beta=.001,temperature=1.,eval_step=.0999)
            return created[-1]
        full=run('full');full_generations=generations.copy();generations.clear()
        try:run('resumed',interrupt=True)
        except RuntimeError as e:assert 'test saved interruption' in str(e)
        else:raise AssertionError('Missing interruption')
        checkpoint=committed_checkpoint(root/'resumed','rl');assert checkpoint.endswith('checkpoint-1')
        resumed=run('resumed',resume=checkpoint)
        assert full.state.max_steps>700 and resumed.state.max_steps==full.state.max_steps
        assert generations==full_generations,'Resumed beam sampling/RNG differs'
        for key,value in full.model.state_dict().items():
            assert torch.equal(value,resumed.model.state_dict()[key]),key
        for key,value in full.ref_model.state_dict().items():
            assert torch.equal(value,resumed.ref_model.state_dict()[key]),'reference '+key
        assert any(not torch.equal(value,full.model.state_dict()[key]) for key,value in full.ref_model.state_dict().items())
        assert any(row.get('grad_norm',0)>0.01 for row in full.state.log_history),'Need meaningful nonzero gradient'
        initial=Qwen2ForCausalLM.from_pretrained(base)
        assert any((value-initial.state_dict()[key]).abs().max()>1e-7 for key,value in full.ref_model.state_dict().items())
        assert full.reward_funcs[0].__module__=='rl' and full.reward_funcs[1].__module__=='rl'
        # Production callback stop700 never alters the scheduler's max_steps.
        callback=callbacks(dict(artifacts=str(root/'budget')),'rl',tok)
        state=TrainerState(global_step=700,max_steps=1728)
        control=callback.on_step_end(full.args,state,TrainerControl())
        assert control.should_save and control.should_training_stop and state.max_steps==1728
        # Actual official evaluator and calc, original hash decoder and real tokenizer.
        evaluator=importlib.import_module('evaluate');pred=root/'predictions.json'
        evaluator.main(base_model=str(base),info_file=info,category=category,test_data_path=str(valid_path),
            result_json_data=str(pred),batch_size=1,num_beams=50,max_new_tokens=8,length_penalty=0.)
        metrics=prediction_metrics(json.loads(pred.read_text()),valid_path,info)
        assert metrics['invalid_candidates']==metrics['duplicate_candidates']==0
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream):importlib.import_module('calc').gao(str(pred),info)
        import numpy as np
        printed=stream.getvalue()
        for label in ['HR','NDCG']:
            line=next(line for line in printed.splitlines() if line.startswith(label))
            values=np.fromstring(line[line.index('[')+1:line.index(']')],sep=' ')
            assert np.allclose(values,[metrics[f'{label}@{k}'] for k in [1,3,5,10,20,50]])
        # Actual SFT entrypoint: SID extension, original labels, best checkpoint
        # and recovery after fit finishes but before selected-model export.
        import transformers
        from scripts.official_utils import recover_export,model_hashes
        from scripts.six_utils import read_json
        sft_module=importlib.import_module('sft')
        unextended=AutoTokenizer.from_pretrained(tokenizer_path,padding_side='left')
        unextended.pad_token=unextended.eos_token
        sft_base=root/'sft_base';unextended.save_pretrained(sft_base)
        sft_cfg=Qwen2Config(**dict(cfg.to_dict(),vocab_size=len(unextended)))
        Qwen2ForCausalLM(sft_cfg).save_pretrained(sft_base)
        sft_csv=root/'sft.csv'
        with sft_csv.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(original));w.writeheader();w.writerow(original)
        output=root/'sft_output';artifacts=root/'sft_artifacts';artifacts.mkdir()
        spec=dict(output=str(output),artifacts=str(artifacts))
        actual_arguments=transformers.TrainingArguments
        def sft_args(**kwargs):
            kwargs.update(use_cpu=True,bf16=False,max_steps=2,gradient_checkpointing=False,
                per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=1,
                save_steps=1,eval_steps=1,warmup_steps=0)
            return actual_arguments(**kwargs)
        def sft_finish(trainer,tokenizer,output):
            assert trainer.state.global_step==2
            assert len(tokenizer)==len(tok)
        sft_module._direct_finish=sft_finish
        with patch.object(transformers,'TrainingArguments',side_effect=sft_args),patch.object(transformers,'Trainer',instrument(transformers.Trainer,spec,'sft',tok)):
            sft_module.train(base_model=str(sft_base),train_file=str(sft_csv),eval_file=str(valid_path),
                category=category,output_dir=str(output),batch_size=2,micro_batch_size=1,
                sid_index_path=str(data/f'index/{category}.index.json'),item_meta_path=str(data/f'index/{category}.item.json'))
        assert committed_checkpoint(output,'sft').endswith('checkpoint-2')
        fit=read_json(artifacts/'fit_complete.json');assert fit['selected_step'] in [1,2]
        recover_export(spec)
        assert model_hashes(output/'selected_model')==fit['model_sha256']
        print('PASS actual official SFT, RL ranking closures, G16 beam sampling, nonzero backward, reference sync/resume, stop700, beam50 evaluation/calc and selected SFT recovery')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--tokenizer',required=True)
    main(p.parse_args().tokenizer)
