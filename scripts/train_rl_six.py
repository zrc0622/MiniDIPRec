"""Worker for six experiments: history10 preflight/SFT and official Ranking RL."""
import argparse
import json
import os
from pathlib import Path

from scripts.six_utils import balance_rows, file_manifest, pin_json, read_json, schedule_spec
from reproduction.prepare import sha256, write_json


def task_data(data, category, mix):
    from reproduction.datasets import rl_data
    train, valid = rl_data(data, category)
    if mix == 'equal':
        train = balance_rows(train)
    elif mix != 'original':
        raise ValueError('Unknown task mixture')
    return train, valid


def reward_functions(generations):
    from reproduction.contracts import ranking_rewards
    def rule_reward(prompts, completions, target, **kwargs):
        return ranking_rewards(completions, target, generations)[0]
    def ndcg_rule_reward(prompts, completions, target, **kwargs):
        return ranking_rewards(completions, target, generations)[1]
    return [rule_reward, ndcg_rule_reward]


def rl_config(output, lengths, variant, schedule):
    from trl import GRPOConfig
    conf = GRPOConfig(output_dir=str(output),
        per_device_train_batch_size=variant['micro_batch'], per_device_eval_batch_size=variant['micro_batch'],
        gradient_accumulation_steps=variant['accumulation'], num_train_epochs=2,
        learning_rate=variant['learning_rate'], warmup_steps=schedule['warmup_steps'],
        warmup_ratio=.03, max_grad_norm=.3, optim='paged_adamw_32bit', lr_scheduler_type='cosine',
        bf16=True, beta=variant['beta'], num_generations=variant['generations'], temperature=1.,
        sync_ref_model=True, ref_model_mixup_alpha=.6, ref_model_sync_steps=schedule['ref_model_sync_steps'],
        max_completion_length=lengths['completion_limit'], model_init_kwargs={'torch_dtype': 'bfloat16'},
        eval_strategy='steps', eval_steps=schedule['eval_steps'], save_strategy='no', save_total_limit=6,
        load_best_model_at_end=False, logging_steps=1, report_to='none', seed=42, disable_tqdm=True,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        ddp_find_unused_parameters=False, max_steps=-1)
    conf.max_prompt_length = lengths['rl_prompt_limit']
    return conf


def trainer_class():
    from minionerec_trainer import ReReTrainer
    class ExposureSchedulerTrainer(ReReTrainer):
        """Preserve the full C scheduler in sample units despite dataloader rounding."""
        def create_scheduler(self, num_training_steps, optimizer=None):
            return super().create_scheduler(self.six_scheduler_steps, optimizer)
    return ExposureSchedulerTrainer


def stop_callback(schedule, artifacts):
    from transformers import TrainerCallback
    class Stop350(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            if state.global_step >= schedule['stop_after_steps']:
                raise ValueError('Budget already reached; finalize without another optimizer step')
            if state.max_steps < schedule['stop_after_steps']:
                raise ValueError('Dataset full schedule is shorter than requested stop budget')
            if state.is_world_process_zero:
                pin_json(Path(artifacts) / 'schedule.json', dict(schedule, trainer_max_steps=state.max_steps))
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step in schedule['snapshot_steps']:
                control.should_save = True
            if state.global_step >= schedule['stop_after_steps']:
                control.should_training_stop = True
                control.should_save = True
            return control
    return Stop350()


def compact_callback(artifacts, sft=False):
    from reproduction.runtime import ArtifactCallback
    class CompactCallback(ArtifactCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            super().on_log(args, state, control, logs, **kwargs)
            if state.is_world_process_zero and (state.global_step % 10 == 0 or 'eval_loss' in (logs or {})):
                print('[six] ' + json.dumps({'step': state.global_step, **(logs or {})}), flush=True)
        def on_train_end(self, args, state, control, **kwargs):
            if sft and state.is_world_process_zero:
                from scripts.evaluate_checkpoints import model_files
                best = Path(state.best_model_checkpoint)
                write_json(Path(artifacts) / 'fit_complete.json', dict(
                    final_step=state.global_step, selected_step=int(best.name.split('-')[-1]),
                    best_checkpoint=str(best), best_metric=state.best_metric,
                    model_sha256={name: sha256(best / name) for name in model_files(best)}))
    return CompactCallback(artifacts)


def preflight(spec):
    """Scan all enabled history10 train/valid tasks; no test data required or evaluated."""
    import ast
    import csv
    import math
    from transformers import AutoConfig, AutoTokenizer
    from reproduction.contracts import check_tokenizer, SIDTrie, next_prompt
    from reproduction.datasets import sft_data, rl_data
    data, artifacts = Path(spec['data']), Path(spec['artifacts'])
    tokenizer = AutoTokenizer.from_pretrained(spec['model'], padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    cfg = AutoConfig.from_pretrained(spec['model'])
    if cfg.model_type != 'qwen3' or cfg.hidden_size != 1024 or cfg.num_hidden_layers != 28:
        raise ValueError('Expected original Qwen3-0.6B')
    index = read_json(data / f'{spec["category"]}.index.json')
    tokenizer.add_tokens(sorted({t for tokens in index.values() for t in tokens}))
    contract = check_tokenizer(tokenizer, index)
    trie = SIDTrie(tokenizer, [''.join(s) for s in index.values()])
    datasets, valid = sft_data(data, spec['category'], tokenizer, cfg.max_position_embeddings)
    sft_counts = {type(d).__name__: len(d) for d in datasets}
    sft_lengths = {type(d).__name__: max(len(r['input_ids']) for r in d) for d in datasets}
    sft_lengths['validation'] = max(len(r['input_ids']) for r in valid)
    del datasets, valid
    train, valid = rl_data(data, spec['category'])
    rl_max = {label: max(len(tokenizer.encode(r['prompt'], add_special_tokens=False)) for r in rows)
              for label, rows in [('train', train), ('valid', valid)]}
    with (data / 'valid.csv').open() as f:
        eval_max = max(len(tokenizer.encode(next_prompt(ast.literal_eval(r['history_item_sid'])),
                                            add_special_tokens=False)) for r in csv.DictReader(f))
    lengths = dict(model=spec['model'], tokenizer=contract, data_audit_sha256=sha256(data / 'audit.json'),
                   sft_maxima=sft_lengths, rl_prompt_maxima=rl_max, eval_prompt_max=eval_max,
                   sft_limit=math.ceil(max(sft_lengths.values()) / 128) * 128,
                   rl_prompt_limit=math.ceil(max(rl_max.values()) / 128) * 128,
                   completion_limit=trie.max_new_tokens,
                   task_counts={'sft': sft_counts, 'rl_train': len(train), 'rl_valid': len(valid)},
                   policy='All enabled train/valid tasks scanned; no test read; reject overflow')
    if trie.max_new_tokens != 5 or max(lengths['sft_limit'], lengths['rl_prompt_limit'] + 5, eval_max + 5) > cfg.max_position_embeddings:
        raise ValueError('Unsupported completion/context length')
    write_json(spec['lengths'], lengths)
    print('[six] preflight ' + json.dumps(lengths['task_counts']), flush=True)


def train(spec, stage, resume):
    import torch
    from datasets import Dataset, concatenate_datasets
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed
    from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback
    from reproduction.contracts import check_tokenizer
    from reproduction.datasets import sft_data
    if int(os.environ.get('WORLD_SIZE', '1')) != 4 or torch.cuda.device_count() != 4:
        raise ValueError('Exactly four visible CUDA GPUs/ranks required')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    set_seed(42)
    data, artifacts, output = Path(spec['data']), Path(spec['artifacts']), Path(spec['output'])
    artifacts.mkdir(parents=True, exist_ok=True)
    lengths = read_json(spec['lengths'])
    if lengths['data_audit_sha256'] != sha256(data / 'audit.json'):
        raise ValueError('Data and measured lengths disagree')
    tokenizer = AutoTokenizer.from_pretrained(spec['model'], padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    indices = read_json(data / f'{spec["category"]}.index.json')
    if stage == 'sft':
        model = AutoModelForCausalLM.from_pretrained(spec['model'], torch_dtype=torch.bfloat16)
        if model.config.model_type != 'qwen3' or model.config.hidden_size != 1024 or model.config.num_hidden_layers != 28:
            raise ValueError('Expected original Qwen3-0.6B base')
        tokenizer.add_tokens(sorted({t for sid in indices.values() for t in sid}))
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id, model.config.eos_token_id = tokenizer.pad_token_id, tokenizer.eos_token_id
        check_tokenizer(tokenizer, indices, model)
        datasets, valid = sft_data(data, spec['category'], tokenizer, lengths['sft_limit'])
        training = concatenate_datasets([Dataset.from_list(list(d)) for d in datasets]).shuffle(seed=42)
        valid = Dataset.from_list(list(valid)).shuffle(seed=42)
        micro = spec['sft_micro_batch']
        conf = TrainingArguments(output_dir=str(output), per_device_train_batch_size=micro,
            per_device_eval_batch_size=micro, gradient_accumulation_steps=256 // micro,
            num_train_epochs=10, learning_rate=3e-4, warmup_steps=20, bf16=True, optim='adamw_torch',
            logging_steps=1, eval_strategy='steps', eval_steps=.05, save_strategy='steps', save_steps=.05,
            save_total_limit=1, load_best_model_at_end=True, metric_for_best_model='eval_loss',
            greater_is_better=False, ddp_find_unused_parameters=False, report_to='none', seed=42,
            max_steps=-1, restore_callback_states_from_checkpoint=True, disable_tqdm=True)
        trainer = Trainer(model=model, args=conf, train_dataset=training, eval_dataset=valid,
            processing_class=tokenizer, data_collator=DataCollatorForSeq2Seq(tokenizer,
                pad_to_multiple_of=8, return_tensors='pt', padding=True),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3), compact_callback(artifacts, sft=True)])
        model.config.use_cache = False
    else:
        variant = spec['variant']
        train_rows, valid_rows = task_data(data, spec['category'], variant['task_mix'])
        schedule = schedule_spec(len(train_rows), variant)
        conf = rl_config(output, lengths, variant, schedule)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        trainer = trainer_class()(model=spec['model'], base_model=spec['model'], processing_class=tokenizer,
            args=conf, train_dataset=Dataset.from_list(train_rows), eval_dataset=Dataset.from_list(valid_rows),
            reward_funcs=reward_functions(variant['generations']), beam_search=True, add_gt=False,
            dynamic_sampling=False, test_during_training=False, dapo=False, gspo=False,
            info_file=str(data / 'info.txt'), callbacks=[compact_callback(artifacts), stop_callback(schedule, artifacts)])
        trainer.six_scheduler_steps = schedule['full_scheduler_steps']
        trainer.generation_config.do_sample = False
        trainer.generation_config.validate()
        check_tokenizer(tokenizer, indices, trainer.model)
        if trainer.is_world_process_zero():
            from collections import Counter
            pin_json(artifacts / 'data_recipe.json', {'task_counts': dict(Counter(r['sample_id'].split(':')[1] for r in train_rows)),
                'row_order_sha256': __import__('hashlib').sha256(json.dumps(train_rows, ensure_ascii=False).encode()).hexdigest(),
                'generations': variant['generations'], 'do_sample': False, 'use_model_defaults': False,
                'policy_dtype': str(next(trainer.model.parameters()).dtype),
                'reference_dtype': str(next(trainer.ref_model.parameters()).dtype),
                'task_mix': variant['task_mix']})
    if trainer.is_world_process_zero():
        write_json(artifacts / 'training_args.json', trainer.args.to_dict())
    trainer.train(resume_from_checkpoint=resume)
    if stage == 'sft':
        selected = output / 'selected_model'
        trainer.save_model(str(selected))
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(selected)
            best = trainer.state.best_model_checkpoint
            write_json(artifacts / 'complete.json', dict(final_step=trainer.state.global_step,
                selected_step=int(Path(best).name.split('-')[-1]) if best else trainer.state.global_step,
                best_checkpoint=best, best_metric=trainer.state.best_metric, model=str(selected),
                model_sha256=file_manifest(selected), selection='minimum validation loss, patience3'))
    else:
        if trainer.state.global_step != 350:
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
