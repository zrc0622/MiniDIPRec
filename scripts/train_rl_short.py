"""Short RL worker: full two-epoch schedule, explicit LR/beta/beam sampling controls."""
import argparse
import json
import math
import os
from pathlib import Path

from transformers import TrainerCallback

from reproduction.prepare import write_json


class StopAfterUpdates(TrainerCallback):
    """Save extra snapshots without extra sampled evals or a shortened scheduler."""
    def __init__(self, stop_step, snapshot_steps, artifacts):
        if stop_step <= 0 or any(s <= 0 or s > stop_step for s in snapshot_steps):
            raise ValueError('Snapshot steps must be within the positive stop budget')
        self.stop_step = stop_step
        self.snapshot_steps = set(snapshot_steps) | {stop_step}
        self.artifacts = Path(artifacts)

    def on_train_begin(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_step:
            # Trainer may still execute an update after on_train_begin requests
            # stopping. The launcher finalizes already-completed budgets instead.
            raise ValueError('Checkpoint already reached stop budget; use run_rl_short.py --resume')
        if self.stop_step > state.max_steps:
            raise ValueError(f'Stop budget {self.stop_step} exceeds full schedule {state.max_steps}')
        schedule = {'planned_steps': state.max_steps, 'stop_after_steps': self.stop_step,
                    'warmup_steps': args.get_warmup_steps(state.max_steps),
                    'learning_rate': args.learning_rate, 'num_train_epochs': args.num_train_epochs,
                    'eval_steps': state.eval_steps, 'save_steps': state.save_steps,
                    'snapshot_steps': sorted(self.snapshot_steps)}
        if state.is_world_process_zero:
            path = self.artifacts / 'schedule.json'
            if path.exists() and json.loads(path.read_text()) != schedule:
                raise ValueError('Scheduler horizon changed on resume')
            write_json(path, schedule)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step in self.snapshot_steps:
            control.should_save = True
        if state.global_step >= self.stop_step:
            control.should_training_stop = True
            control.should_save = True
        return control


def rl_config(output, lengths, micro_batch, learning_rate, beta=0.001):
    """Keep in parity with reproduction.train's active RL GRPOConfig (tested)."""
    from trl import GRPOConfig
    conf = GRPOConfig(output_dir=output, per_device_train_batch_size=micro_batch,
        per_device_eval_batch_size=micro_batch, gradient_accumulation_steps=1024 // (4 * micro_batch),
        num_train_epochs=2, learning_rate=learning_rate, warmup_ratio=0.03, max_grad_norm=0.3,
        optim='paged_adamw_32bit', lr_scheduler_type='cosine', bf16=True, beta=beta,
        num_generations=16, temperature=1.0, sync_ref_model=True,
        ref_model_mixup_alpha=0.6, ref_model_sync_steps=512,
        max_completion_length=lengths['completion_limit'], model_init_kwargs={'torch_dtype': 'bfloat16'},
        eval_strategy='steps', eval_steps=0.0999, save_strategy='steps', save_steps=0.0999,
        save_total_limit=20, load_best_model_at_end=True, metric_for_best_model='eval_reward',
        greater_is_better=True, logging_steps=1, report_to='none', seed=42,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        ddp_find_unused_parameters=False, max_steps=-1)
    conf.max_prompt_length = lengths['rl_prompt_limit']
    return conf


def configure_candidates(trainer, do_sample):
    """Vary only this instance's G16 beams; the shared official trainer is unchanged."""
    generation = trainer.generation_config
    if not trainer.beam_search or generation.num_beams != 16 or generation.num_return_sequences != 16:
        raise ValueError('Short RL requires 16 beams and 16 candidates per prompt')
    generation.do_sample = do_sample
    generation.validate()
    return {'beta': trainer.beta, 'do_sample': generation.do_sample,
            'num_beams': generation.num_beams, 'num_return_sequences': generation.num_return_sequences,
            'temperature': generation.temperature, 'use_model_defaults': False,
            'applies_to': 'RL training and original G16 validation reward; final ranking eval stays deterministic beam50'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'data', 'category', 'output', 'artifacts', 'lengths'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--micro-batch', type=int, default=16)
    p.add_argument('--learning-rate', type=float, default=1e-5)
    p.add_argument('--beta', type=float, default=0.001)
    p.add_argument('--do-sample', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--stop-after-steps', type=int, required=True)
    p.add_argument('--snapshot-steps', type=int, nargs='+', required=True)
    p.add_argument('--resume')
    args = p.parse_args()
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, set_seed
    from minionerec_trainer import ReReTrainer
    from reproduction.contracts import check_tokenizer, ranking_rewards
    from reproduction.datasets import rl_data
    from reproduction.runtime import ArtifactCallback
    if int(os.environ.get('WORLD_SIZE', '1')) != 4 or torch.cuda.device_count() != 4:
        raise ValueError('Exactly four visible CUDA GPUs required')
    if args.micro_batch < 16 or args.micro_batch % 16 or 256 % args.micro_batch:
        raise ValueError('RL micro batch must be a multiple of G16 dividing 256')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError('Learning rate must be positive and finite')
    if not math.isfinite(args.beta) or args.beta <= 0:
        raise ValueError('Beta must be positive and finite')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    set_seed(42)
    root, artifacts = Path(args.data), Path(args.artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)
    lengths = json.loads(Path(args.lengths).read_text())
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    indices = json.loads((root / f'{args.category}.index.json').read_text())
    check_tokenizer(tokenizer, indices)
    train, valid = rl_data(root, args.category)

    def rule_reward(prompts, completions, target, **kwargs):
        return ranking_rewards(completions, target)[0]

    def ndcg_rule_reward(prompts, completions, target, **kwargs):
        return ranking_rewards(completions, target)[1]

    conf = rl_config(args.output, lengths, args.micro_batch, args.learning_rate, args.beta)
    # Extra snapshots may never evict a requested diagnostic checkpoint.
    conf.save_total_limit = max(conf.save_total_limit, len(set(args.snapshot_steps)) + 20)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    trainer = ReReTrainer(model=args.model, base_model=args.model, processing_class=tokenizer,
        args=conf, train_dataset=Dataset.from_list(train), eval_dataset=Dataset.from_list(valid),
        reward_funcs=[rule_reward, ndcg_rule_reward], beam_search=True, add_gt=False,
        dynamic_sampling=False, test_during_training=False, dapo=False, gspo=False,
        info_file=str(root / 'info.txt'), callbacks=[ArtifactCallback(artifacts),
            StopAfterUpdates(args.stop_after_steps, args.snapshot_steps, artifacts)])
    candidate_config = configure_candidates(trainer, args.do_sample)
    check_tokenizer(tokenizer, indices, trainer.model)
    if trainer.is_world_process_zero():
        recorded = trainer.args.to_dict()
        recorded['max_prompt_length'] = lengths['rl_prompt_limit']
        write_json(artifacts / 'training_args.json', recorded)
        write_json(artifacts / 'candidate_config.json', candidate_config)
        print('RL candidate settings: ' + json.dumps(candidate_config), flush=True)
    trainer.train(resume_from_checkpoint=args.resume)
    # Evaluate saved checkpoints, not the reward-selected in-memory export.
    # The launcher records completion from the final checkpoint's TrainerState.
    trainer.accelerator.wait_for_everyone()


if __name__ == '__main__':
    from torch.distributed.elastic.multiprocessing.errors import record
    record(main)()
