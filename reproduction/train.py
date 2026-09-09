"""Four-GPU adapters retaining official Trainer/ReReTrainer and enabled recipes."""
import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['sft', 'rl'], required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--category', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--artifacts', required=True)
    p.add_argument('--lengths', required=True)
    p.add_argument('--micro-batch', type=int, required=True)
    p.add_argument('--resume')
    p.add_argument('--max-steps', type=int, default=-1)
    args = p.parse_args()
    import torch
    from datasets import Dataset, concatenate_datasets
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq, EarlyStoppingCallback, set_seed
    from trl import GRPOConfig
    from reproduction.contracts import check_tokenizer, ranking_rewards
    from reproduction.datasets import sft_data, rl_data
    from reproduction.runtime import ArtifactCallback, training_record
    from reproduction.prepare import write_json
    if int(os.environ.get('WORLD_SIZE', '1')) != 4 or torch.cuda.device_count() != 4:
        raise ValueError('Training requires torchrun/accelerate with exactly four visible GPUs')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    if 1024 % (4 * args.micro_batch):
        raise ValueError('micro batch must divide 256 to preserve official effective batch 1024')
    if args.stage == 'rl' and args.micro_batch % 16:
        raise ValueError('RL micro batch must be a multiple of G=16')
    set_seed(42)
    root = Path(args.data)
    lengths = json.loads(Path(args.lengths).read_text())
    artifact_dir = Path(args.artifacts)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    indices = json.loads((root / f'{args.category}.index.json').read_text())
    accumulation = 1024 // (4 * args.micro_batch)
    callback = ArtifactCallback(artifact_dir)
    if args.stage == 'sft':
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
        if model.config.model_type != 'qwen3' or model.config.hidden_size != 1024 or model.config.num_hidden_layers != 28:
            raise ValueError('Expected original Qwen3-0.6B model (or its local copy)')
        tokenizer.add_tokens(sorted({t for sid in indices.values() for t in sid}))
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        check_tokenizer(tokenizer, indices, model)
        datasets, valid = sft_data(root, args.category, tokenizer, lengths['sft_limit'])
        def as_hf(ds):
            return Dataset.from_list(list(ds))
        train = concatenate_datasets([as_hf(d) for d in datasets]).shuffle(seed=42)
        valid = as_hf(valid).shuffle(seed=42)
        # Official sft.py defaults and sft.sh global batch; only micro/accum/length adapt.
        conf = TrainingArguments(output_dir=args.output, per_device_train_batch_size=args.micro_batch,
            per_device_eval_batch_size=args.micro_batch, gradient_accumulation_steps=accumulation,
            num_train_epochs=10, learning_rate=3e-4, warmup_steps=20, bf16=True,
            optim='adamw_torch', logging_steps=1, eval_strategy='steps', eval_steps=0.05,
            save_strategy='steps', save_steps=0.05, save_total_limit=1, load_best_model_at_end=True,
            metric_for_best_model='eval_loss', greater_is_better=False,
            ddp_find_unused_parameters=False, report_to='none', seed=42, max_steps=args.max_steps,
            restore_callback_states_from_checkpoint=True)
        trainer = Trainer(model=model, args=conf, train_dataset=train, eval_dataset=valid,
            processing_class=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors='pt', padding=True),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3), callback])
        model.config.use_cache = False
    else:
        from minionerec_trainer import ReReTrainer
        check_tokenizer(tokenizer, indices)
        train, valid = rl_data(root, args.category)
        def rule_reward(prompts, completions, target, **kwargs):
            return ranking_rewards(completions, target)[0]
        def ndcg_rule_reward(prompts, completions, target, **kwargs):
            return ranking_rewards(completions, target)[1]
        # Retain the official paged optimizer and reference synchronization. Align save
        # with validation to select on validation reward, never on test labels.
        conf = GRPOConfig(output_dir=args.output, per_device_train_batch_size=args.micro_batch,
            per_device_eval_batch_size=args.micro_batch, gradient_accumulation_steps=accumulation,
            num_train_epochs=2, learning_rate=1e-5, warmup_ratio=0.03, max_grad_norm=0.3,
            optim='paged_adamw_32bit', lr_scheduler_type='cosine', bf16=True, beta=1e-3,
            num_generations=16, temperature=1.0, sync_ref_model=True,
            ref_model_mixup_alpha=0.6, ref_model_sync_steps=512,
            max_completion_length=lengths['completion_limit'], model_init_kwargs={'torch_dtype': 'bfloat16'},
            eval_strategy='steps', eval_steps=0.0999, save_strategy='steps', save_steps=0.0999,
            save_total_limit=20, load_best_model_at_end=True, metric_for_best_model='eval_reward',
            greater_is_better=True, logging_steps=1, report_to='none', seed=42,
            gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
            ddp_find_unused_parameters=False, max_steps=args.max_steps)
        conf.max_prompt_length = lengths['rl_prompt_limit']
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        trainer = ReReTrainer(model=args.model, base_model=args.model, processing_class=tokenizer,
            args=conf, train_dataset=Dataset.from_list(train), eval_dataset=Dataset.from_list(valid),
            reward_funcs=[rule_reward, ndcg_rule_reward], beam_search=True, add_gt=False,
            dynamic_sampling=False, test_during_training=False, dapo=False, gspo=False,
            info_file=str(root / 'info.txt'), callbacks=[callback])
        check_tokenizer(tokenizer, indices, trainer.model)
    if trainer.is_world_process_zero():
        recorded_args = trainer.args.to_dict()
        if args.stage == 'rl':
            recorded_args['max_prompt_length'] = lengths['rl_prompt_limit']
        write_json(artifact_dir / 'training_args.json', recorded_args)
    trainer.train(resume_from_checkpoint=args.resume)
    final = Path(args.output) / 'selected_model'
    # Trainer restores the best validation checkpoint before this export.
    trainer.save_model(str(final))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(final)
        training_record(trainer, artifact_dir, final, args.model)
    trainer.accelerator.wait_for_everyone()


if __name__ == '__main__':
    main()
