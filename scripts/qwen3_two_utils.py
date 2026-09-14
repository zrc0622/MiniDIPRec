"""Fixed Qwen3-1.7B/current-H10 contracts, isolated from older sub-1B suites."""
from pathlib import Path

from reproduction.prepare import CATEGORIES, sha256, write_json
from scripts.five_utils import schedule_spec as prior_schedule
from scripts.six_utils import read_json

SNAPSHOTS = [25, 50, 100, 175, 350]
DEFAULT_MODEL = 'Qwen/Qwen3-1.7B'


def variants():
    return [dict(label=label, category=category, family='qwen3', recipe='current',
                 learning_rate=1e-5, beta=.04, generations=16, micro_batch=16,
                 accumulation=16, history_limit=10, task_mix='original',
                 optimizer='paged_adamw_32bit', do_sample=False)
            for label, category in zip(['01_office', '02_industrial'], CATEGORIES)]


def validate_variant(variant):
    if variant not in variants():
        raise ValueError('Expected the fixed Qwen3-1.7B current H10 recipe')


def schedule_spec(rows):
    return dict(prior_schedule(rows), snapshot_steps=SNAPSHOTS)


def model_contract(config, expanded_vocab):
    """Validate 1.7B structure and count expanded parameters without allocating weights."""
    import copy
    import torch
    from transformers import AutoModelForCausalLM
    expected = dict(model_type='qwen3', hidden_size=2048, intermediate_size=6144,
                    num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8,
                    head_dim=128, tie_word_embeddings=True)
    if any(getattr(config, key, None) != value for key, value in expected.items()):
        raise ValueError('Expected Qwen3-1.7B architecture (or its local copy)')
    cfg = copy.deepcopy(config)
    cfg.vocab_size = expanded_vocab
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(cfg)
    count = sum(p.numel() for p in model.parameters())
    if not 1_600_000_000 <= count < 1_800_000_000:
        raise ValueError(f'Unexpected expanded Qwen3-1.7B parameter count: {count}')
    return dict(family='qwen3', size='1.7B', parameters_after_sid_extension=count,
                architecture=expected)


def finish_rl(spec):
    from scripts.evaluate_checkpoints import model_files
    from scripts.five_utils import checkpoint_manifest
    artifacts, output = Path(spec['artifacts']), Path(spec['output'])
    for step in SNAPSHOTS:
        checkpoint = output / f'checkpoint-{step}'
        model_files(checkpoint, step)
        record = read_json(checkpoint / 'five_complete.json')
        if (record['stage'] != 'rl' or record['step'] != step
                or record['files'] != checkpoint_manifest(checkpoint)):
            raise ValueError(f'Checkpoint is not fully committed: {checkpoint}')
        if not (checkpoint / 'reference_model.pt').is_file():
            raise ValueError(f'Missing RL reference: {checkpoint}')
    state = read_json(output / 'checkpoint-350/trainer_state.json')
    schedule = read_json(artifacts / 'schedule.json')
    if (state['global_step'] != 350 or state['max_steps'] != schedule['trainer_max_steps']
            or schedule['stop_after_steps'] != 350 or schedule['snapshot_steps'] != SNAPSHOTS):
        raise ValueError('Saved final step/full schedule/snapshots changed')
    write_json(artifacts / 'complete.json', dict(final_step=350, parent_model=spec['model'],
        final_checkpoint=str(output / 'checkpoint-350'), schedule=schedule,
        checkpoint_files={f'checkpoint-{s}/{name}': sha256(output / f'checkpoint-{s}' / name)
                          for s in SNAPSHOTS
                          for name in model_files(output / f'checkpoint-{s}', s) + ['reference_model.pt']},
        selection='No reward-based model selection; compare every snapshot to own SFT on validation'))
