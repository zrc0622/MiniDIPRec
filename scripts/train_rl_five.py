"""Four-GPU worker for the five-run results2 suite; official recipe or current H10."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path

from reproduction.prepare import sha256, write_json
from scripts.five_utils import model_contract, rl_data, row_digest, schedule_spec, sft_data
from scripts.six_utils import file_manifest, pin_json, read_json, verify_files
from scripts.train_rl_six import compact_callback, reward_functions, rl_config, stop_callback, trainer_class


def preflight(spec):
    from transformers import AutoConfig, AutoTokenizer
    from reproduction.contracts import SIDTrie, check_tokenizer, next_prompt
    import ast
    import csv
    data = Path(spec['data'])
    variant = spec['variant']
    tok = AutoTokenizer.from_pretrained(spec['model'], padding_side='left')
    cfg = AutoConfig.from_pretrained(spec['model'])
    # Only the imported H10 arm may receive an already extended SFT tokenizer.
    if variant['recipe'] == 'official' and tok.get_added_vocab().keys() & \
            {t for sid in read_json(data / f'{spec["category"]}.index.json').values() for t in sid}:
        raise ValueError('Official SFT requires original pretrained model, not an existing SID SFT')
    tok.pad_token = tok.eos_token
    indices = read_json(data / f'{spec["category"]}.index.json')
    tok.add_tokens(sorted({t for sid in indices.values() for t in sid}))
    contract = model_contract(cfg, variant['family'], len(tok))
    token_contract = check_tokenizer(tok, indices)
    trie = SIDTrie(tok, [''.join(s) for s in indices.values()])
    datasets, valid = sft_data(data, spec['category'], tok, cfg.max_position_embeddings, variant['recipe'])
    counts = {type(d).__name__: len(d) for d in datasets}
    maxima = {type(d).__name__: max(len(r['input_ids']) for r in d) for d in datasets}
    maxima['validation'] = max(len(r['input_ids']) for r in valid)
    del datasets, valid
    train, valid = rl_data(data, spec['category'], variant['recipe'])
    rl_max = {split: max(len(tok.encode(r['prompt'], add_special_tokens=False)) for r in rows)
              for split, rows in [('train', train), ('valid', valid)]}
    with (data / 'valid.csv').open() as f:
        eval_max = max(len(tok.encode(next_prompt(ast.literal_eval(row['history_item_sid'])),
                                     add_special_tokens=False)) for row in csv.DictReader(f))
    sft_limit = 512 if variant['recipe'] == 'official' else math.ceil(max(maxima.values()) / 128) * 128
    rl_limit = math.ceil(max(rl_max.values()) / 128) * 128
    if max(maxima.values()) > sft_limit:
        raise ValueError('Official SFT cutoff512 would truncate data; refusing silent recipe change')
    if trie.max_new_tokens != 5 or max(sft_limit, rl_limit + 5, eval_max + 5) > cfg.max_position_embeddings:
        raise ValueError('Unsupported SID completion/context length')
    write_json(spec['lengths'], dict(model=spec['model'], model_contract=contract, tokenizer=token_contract,
        data_audit_sha256=sha256(data / 'audit.json'), sft_maxima=maxima, sft_limit=sft_limit,
        rl_prompt_maxima=rl_max, rl_prompt_limit=rl_limit, eval_prompt_max=eval_max, completion_limit=5,
        task_counts=dict(sft=counts, rl_train=len(train), rl_valid=len(valid),
                         rl_by_task=dict(Counter(r['sample_id'].split(':')[1] for r in train))),
        rl_row_order_sha256=row_digest(train), recipe=variant['recipe'],
        policy='all train/valid tasks scanned; original official SFT cutoff512; no test read or truncation'))
    print('[five] preflight ' + json.dumps(contract), flush=True)


def training_config(output, lengths, variant, schedule):
    conf = rl_config(output, lengths, variant, schedule)
    conf.optim = variant['optimizer']
    if variant['recipe'] == 'official':
        # Upstream passes a model path without explicit model_init_kwargs.
        # Preserve that loading path and record actual policy/reference dtype.
        conf.model_init_kwargs = None
    return conf


def artifact_callback(artifacts, sft=False):
    """Trim only replayed/incomplete scalar rows and mark fully saved checkpoints."""
    from scripts.five_utils import checkpoint_manifest
    parent = type(compact_callback(artifacts, sft=sft))

    class FiveArtifacts(parent):
        def on_train_begin(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                path = Path(artifacts) / 'metrics.jsonl'
                if path.exists():
                    kept = []
                    lines = path.read_bytes().splitlines()
                    for i, line in enumerate(lines):
                        try:
                            row = json.loads(line)
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            if i != len(lines)-1:
                                raise ValueError('Corrupt nonterminal scalar log record')
                            break
                        if row['step'] <= state.global_step:
                            kept.append(json.dumps(row))
                    temporary = path.with_suffix('.tmp')
                    temporary.write_text(''.join(line+'\n' for line in kept))
                    temporary.replace(path)

        def on_save(self, args, state, control, **kwargs):
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
            if state.is_world_process_zero:
                checkpoint = Path(args.output_dir) / f'checkpoint-{state.global_step}'
                write_json(checkpoint / 'five_complete.json', dict(step=state.global_step,
                    files=checkpoint_manifest(checkpoint), stage='sft' if sft else 'rl'))
            if dist.is_available() and dist.is_initialized():
                dist.barrier()
    return FiveArtifacts(artifacts)


def train(spec, stage, resume=None):
    import torch
    from datasets import Dataset, concatenate_datasets
    from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
                              DataCollatorForSeq2Seq, EarlyStoppingCallback, set_seed)
    from reproduction.contracts import check_tokenizer
    if int(os.environ.get('WORLD_SIZE', '1')) != 4 or torch.cuda.device_count() != 4:
        raise ValueError('Exactly four visible CUDA GPUs/ranks required')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    set_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    data, artifacts, output = map(Path, (spec['data'], spec['artifacts'], spec['output']))
    artifacts.mkdir(parents=True, exist_ok=True)
    lengths = read_json(spec['lengths'])
    variant = spec['variant']
    if lengths['data_audit_sha256'] != sha256(data / 'audit.json') or lengths['recipe'] != variant['recipe']:
        raise ValueError('Preflight/data/recipe mismatch')
    verify_files(data, read_json(data / 'audit.json')['generated_sha256'])
    tok = AutoTokenizer.from_pretrained(spec['model'], padding_side='left')
    tok.pad_token = tok.eos_token
    indices = read_json(data / f'{spec["category"]}.index.json')
    if stage == 'sft':
        if variant['recipe'] != 'official':
            raise ValueError('Runs1/2 must import completed SFT10, never retrain it')
        model = AutoModelForCausalLM.from_pretrained(spec['model'], torch_dtype=torch.bfloat16)
        tok.add_tokens(sorted({t for sid in indices.values() for t in sid}))
        model_contract(model.config, variant['family'], len(tok))
        model.resize_token_embeddings(len(tok))
        model.config.pad_token_id, model.config.eos_token_id = tok.pad_token_id, tok.eos_token_id
        check_tokenizer(tok, indices, model)
        datasets, valid = sft_data(data, spec['category'], tok, lengths['sft_limit'], 'official')
        training = concatenate_datasets([Dataset.from_list(list(d)) for d in datasets]).shuffle(seed=42)
        # Upstream applies both its seed shuffle and a second fixed seed42 shuffle.
        valid = Dataset.from_list(list(valid)).shuffle(seed=42).shuffle(seed=42)
        micro = spec['sft_micro_batch']
        conf = TrainingArguments(output_dir=str(output), per_device_train_batch_size=micro,
            per_device_eval_batch_size=micro, gradient_accumulation_steps=256 // micro,
            num_train_epochs=10, learning_rate=3e-4, warmup_steps=20, bf16=True, optim='adamw_torch',
            logging_steps=1, eval_strategy='steps', eval_steps=.05, save_strategy='steps', save_steps=.05,
            save_total_limit=2, load_best_model_at_end=True, metric_for_best_model='eval_loss',
            greater_is_better=False, ddp_find_unused_parameters=False, report_to='none', seed=42,
            max_steps=-1, restore_callback_states_from_checkpoint=True, disable_tqdm=True)
        trainer = Trainer(model=model, args=conf, train_dataset=training, eval_dataset=valid,
            processing_class=tok, data_collator=DataCollatorForSeq2Seq(tok,
                pad_to_multiple_of=8, return_tensors='pt', padding=True),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3), artifact_callback(artifacts, sft=True)])
        model.config.use_cache = False
    else:
        train_rows, valid_rows = rl_data(data, spec['category'], variant['recipe'])
        if row_digest(train_rows) != lengths['rl_row_order_sha256']:
            raise ValueError('RL preflight row order changed')
        schedule = schedule_spec(len(train_rows))
        conf = training_config(output, lengths, variant, schedule)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        training, validation = Dataset.from_list(train_rows), Dataset.from_list(valid_rows)
        if variant['recipe'] == 'official':
            training, validation = training.shuffle(seed=42), validation.shuffle(seed=42)
        trainer = trainer_class()(model=spec['model'], base_model=spec['model'], processing_class=tok,
            args=conf, train_dataset=training, eval_dataset=validation,
            reward_funcs=reward_functions(16), beam_search=True, add_gt=False, dynamic_sampling=False,
            test_during_training=False, dapo=False, gspo=False, info_file=str(data / 'info.txt'),
            callbacks=[artifact_callback(artifacts), stop_callback(schedule, artifacts)])
        trainer.six_scheduler_steps = schedule['full_scheduler_steps']
        trainer.generation_config.do_sample = variant['do_sample']
        trainer.generation_config.validate()
        check_tokenizer(tok, indices, trainer.model)
        model_contract(trainer.model.config, variant['family'], len(tok))
        if trainer.is_world_process_zero():
            pin_json(artifacts / 'data_recipe.json', dict(recipe=variant['recipe'],
                task_counts=dict(Counter(r['sample_id'].split(':')[1] for r in train_rows)),
                row_order_sha256=row_digest(train_rows), generations=16, do_sample=variant['do_sample'],
                dataset_shuffle_seed=42 if variant['recipe']=='official' else None,
                optimizer=variant['optimizer'], use_model_defaults=False,
                policy_dtype=str(next(trainer.model.parameters()).dtype),
                reference_dtype=str(next(trainer.ref_model.parameters()).dtype),
                fp32_probability_override=False, prompt_style='upstream raw' if variant['recipe']=='official' else 'SFT aligned',
                reward_target='row-local target; no train/validation lookup merge'))
    if trainer.is_world_process_zero():
        write_json(artifacts / 'training_args.json', trainer.args.to_dict())
    trainer.train(resume_from_checkpoint=resume)
    if trainer.is_world_process_zero():
        optimizer = trainer.optimizer
        while hasattr(optimizer, 'optimizer'):
            optimizer = optimizer.optimizer
        state_dtypes = Counter()
        for value in getattr(optimizer, 'state', {}).values():
            for key, tensor in value.items():
                if hasattr(tensor, 'dtype'):
                    state_dtypes[f'{key}:{tensor.dtype}'] += 1
        write_json(artifacts / 'optimizer_record.json', dict(
            configured=str(trainer.args.optim), runtime_class=f'{type(optimizer).__module__}.{type(optimizer).__name__}',
            local_state_tensor_counts=dict(state_dtypes),
            beta1=trainer.args.adam_beta1, beta2=trainer.args.adam_beta2,
            epsilon=trainer.args.adam_epsilon, weight_decay=trainer.args.weight_decay))
    if stage == 'sft':
        selected = output / 'selected_model'
        trainer.save_model(str(selected))
        if trainer.is_world_process_zero():
            tok.save_pretrained(selected)
            best = Path(trainer.state.best_model_checkpoint)
            write_json(artifacts / 'complete.json', dict(final_step=trainer.state.global_step,
                selected_step=int(best.name.split('-')[-1]), best_checkpoint=str(best),
                best_metric=trainer.state.best_metric, model=str(selected), model_sha256=file_manifest(selected),
                selection='minimum validation loss, patience3'))
    elif trainer.state.global_step != 350:
        raise ValueError('RL did not finish exactly350 updates')
    trainer.accelerator.wait_for_everyone()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', required=True)
    p.add_argument('--stage', choices=['preflight', 'sft', 'rl'], required=True)
    p.add_argument('--resume')
    args = p.parse_args()
    spec = read_json(args.spec)
    if args.stage == 'preflight':
        preflight(spec)
    else:
        train(spec, args.stage, args.resume)


if __name__ == '__main__':
    main()
