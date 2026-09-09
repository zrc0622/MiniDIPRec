"""Run via torchrun --nproc_per_node=4 on CPU. Verifies actual collective reward slices."""
import json
import os
from pathlib import Path
import tempfile
import torch
import torch.distributed as dist
from datasets import Dataset
from trl import GRPOConfig
from minionerec_trainer import ReReTrainer
from reproduction.contracts import ranking_rewards, validate_groups
from tests.test_reproduction_runtime import RuntimeTests


def main():
    torch.set_num_threads(1)
    dist.init_process_group('gloo')
    rank = dist.get_rank()
    tmp = [tempfile.mkdtemp(prefix='minirepro-ddp-') if rank == 0 else None]
    dist.broadcast_object_list(tmp, src=0)
    root = Path(tmp[0])
    if rank == 0:
        RuntimeTests().fixture(str(root))
    dist.barrier()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(root)
    sids = [f'<a_{i}><b_0><c_0>' for i in range(20)]
    rows = [{'prompt': '### Response:\n', 'target': sids[i] + '\n', 'sample_id': str(i)} for i in range(7)]
    seen = []
    def rule_reward(prompts, completions, target, sample_id):
        validate_groups([{'prompt': p, 'target': t, 'sample_id': s} for p, t, s in zip(prompts, target, sample_id)], 16)
        seen.append(sample_id)
        return ranking_rewards(completions, target, 16)[0]
    def ndcg_rule_reward(prompts, completions, target, **kwargs):
        return ranking_rewards(completions, target, 16)[1]
    conf = GRPOConfig(output_dir=str(root / 'run'), use_cpu=True, bf16=False, fp16=False,
        per_device_train_batch_size=16, per_device_eval_batch_size=16, num_generations=16,
        gradient_accumulation_steps=2, max_steps=2, optim='adamw_torch', learning_rate=1e-3,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
        ddp_find_unused_parameters=False, report_to='none', save_steps=1, eval_strategy='steps', eval_steps=1,
        logging_steps=1, max_completion_length=5, beta=.001, sync_ref_model=True,
        ref_model_sync_steps=2, seed=42, disable_tqdm=True,
        load_best_model_at_end=True, metric_for_best_model='eval_reward', greater_is_better=True)
    conf.max_prompt_length = 64
    trainer = ReReTrainer(model=str(root), base_model=str(root), processing_class=tok, args=conf,
        reward_funcs=[rule_reward, ndcg_rule_reward], train_dataset=Dataset.from_list(rows),
        eval_dataset=Dataset.from_list(rows[:5]), beam_search=True, test_during_training=False,
        info_file=str(root / 'info.txt'))
    trainer.train()
    assert trainer.state.global_step == 2 and seen
    for ids in seen:
        assert len(ids) == 16 and len(set(ids)) == 1
    refs = [None] * 4
    dist.all_gather_object(refs, sum(p.detach().float().sum().item() for p in trainer.ref_model.parameters()))
    assert max(refs) - min(refs) < 1e-5, refs
    assert (root / 'run/checkpoint-2/reference_model.pt').exists()
    print(json.dumps({'rank': rank, 'world': dist.get_world_size(), 'steps': trainer.state.global_step,
        'local_groups_checked': len(seen), 'reference_equal_across_ranks': True, 'checkpoint': str(root / 'run/checkpoint-2')}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
