"""Scan every enabled task before training; choose measured, non-truncating limits."""
import argparse
import json
import math
from pathlib import Path
from reproduction.prepare import write_json, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--category', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    from transformers import AutoConfig, AutoTokenizer
    from reproduction.contracts import SIDTrie, check_tokenizer
    from reproduction.datasets import sft_data, rl_data
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    cfg = AutoConfig.from_pretrained(args.model)
    if cfg.model_type != 'qwen3' or cfg.hidden_size != 1024 or cfg.num_hidden_layers != 28:
        raise ValueError('Expected Qwen/Qwen3-0.6B architecture; use original model or local copy')
    root = Path(args.data)
    index = json.loads((root / f'{args.category}.index.json').read_text())
    tokenizer.add_tokens(sorted({t for ids in index.values() for t in ids}))
    contract = check_tokenizer(tokenizer, index)
    trie = SIDTrie(tokenizer, [''.join(s) for s in index.values()])
    context = cfg.max_position_embeddings
    datasets, valid = sft_data(root, args.category, tokenizer, context)
    sft_counts = {type(d).__name__: len(d) for d in datasets}
    sft_lengths = {type(d).__name__: max(len(row['input_ids']) for row in d) for d in datasets}
    sft_lengths['validation'] = max(len(row['input_ids']) for row in valid)
    del datasets, valid
    train, valid = rl_data(root, args.category)
    rl_lengths = {}
    for name, rows in [('train', train), ('valid', valid)]:
        rl_lengths[name] = max(len(tokenizer.encode(r['prompt'], add_special_tokens=False)) for r in rows)
    # Test targets are not read for selection. Only test input lengths are checked.
    import ast
    import csv
    from reproduction.contracts import next_prompt
    eval_max = 0
    for split in ('valid', 'test'):
        with (root / f'{split}.csv').open() as f:
            for row in csv.DictReader(f):
                prompt = next_prompt(ast.literal_eval(row['history_item_sid']))
                eval_max = max(eval_max, len(tokenizer.encode(prompt, add_special_tokens=False)))
    def rounded(n):
        return math.ceil(n / 128) * 128
    sft_limit = rounded(max(sft_lengths.values()))
    rl_limit = rounded(max(rl_lengths.values()))
    completion_limit = trie.max_new_tokens
    if completion_limit != 5:
        raise ValueError('Expected three atomic SID tokens + newline + EOS')
    if max(sft_limit, rl_limit + completion_limit, eval_max + completion_limit) > context:
        raise ValueError('Data exceeds Qwen3 context; refusing to truncate')
    write_json(args.output, {'model': args.model, 'model_config': cfg.to_dict(), 'tokenizer': contract,
        'data_audit_sha256': sha256(root / 'audit.json'), 'sft_maxima': sft_lengths,
        'rl_prompt_maxima': rl_lengths, 'eval_prompt_max': eval_max,
        'sft_limit': sft_limit, 'rl_prompt_limit': rl_limit, 'completion_limit': completion_limit,
        'task_counts': {'sft': sft_counts, 'rl_train': len(train), 'rl_valid': len(valid)},
        'policy': 'full-task token scan, round up to 128; overflow always raises'})
    tokenizer.save_pretrained(Path(args.output).parent / 'tokenizer')
    print(json.dumps({'sft_limit': sft_limit, 'rl_prompt_limit': rl_limit, 'completion_limit': completion_limit}))


if __name__ == '__main__':
    main()
