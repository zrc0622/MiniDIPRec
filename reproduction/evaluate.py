"""Identical deterministic full-catalog beam ranking for SFT and SFT+RL."""
import argparse
import ast
import csv
import json
import math
import os
from pathlib import Path
from reproduction.prepare import write_json


def metrics(rows):
    if not rows:
        raise ValueError('Empty evaluation')
    scores = {}
    for k in (5, 10):
        hits, gain = 0, 0.0
        for row in rows:
            predictions = row['predictions'][:k]
            if row['target_sid'] in predictions:
                rank = predictions.index(row['target_sid']) + 1
                hits += 1
                gain += 1 / math.log2(rank + 1)
        scores.update({f'HR@{k}': hits / len(rows), f'Recall@{k}': hits / len(rows), f'NDCG@{k}': gain / len(rows)})
    return {'samples': len(rows), **scores}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--category', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--split', choices=['valid', 'test'], required=True)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--lengths', required=True)
    args = p.parse_args()
    import torch
    import torch.distributed as dist
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessorList
    from reproduction.contracts import next_prompt, SIDTrie, check_tokenizer
    if int(os.environ.get('WORLD_SIZE', '1')) != 4 or torch.cuda.device_count() != 4:
        raise ValueError('Evaluation requires exactly four visible GPUs and four ranks')
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    root, output = Path(args.data), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(local_rank).eval()
    indices = json.loads((root / f'{args.category}.index.json').read_text())
    check_tokenizer(tokenizer, indices, model)
    catalog = {}
    for item_id, sid in indices.items():
        catalog.setdefault(''.join(sid), []).append(item_id)
    trie = SIDTrie(tokenizer, catalog)
    limits = json.loads(Path(args.lengths).read_text())
    conf = GenerationConfig(num_beams=50, num_return_sequences=50, do_sample=False,
        length_penalty=0.0, max_new_tokens=trie.max_new_tokens, pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id, top_k=None, top_p=None,
        return_dict_in_generate=True, output_scores=True, use_cache=True)
    with (root / f'{args.split}.csv').open() as f:
        all_rows = list(csv.DictReader(f))
    rows = all_rows[rank::4]
    path = output / f'{args.split}.rank{rank}.jsonl'
    with path.open('w') as f, torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            prompts = []
            for row in batch:
                prompts.append(next_prompt(ast.literal_eval(row['history_item_sid'])))
            inputs = tokenizer(prompts, return_tensors='pt', padding=True, add_special_tokens=False).to(local_rank)
            width = inputs.input_ids.shape[1]
            if width > limits['eval_prompt_max']:
                raise ValueError('Evaluation prompt exceeds scanned length')
            generated = model.generate(**inputs, generation_config=conf,
                logits_processor=LogitsProcessorList([trie.processor(width)]))
            text = tokenizer.batch_decode(generated.sequences[:, width:], skip_special_tokens=True)
            scores = generated.sequences_scores.tolist()
            for j, row in enumerate(batch):
                candidates = [s.strip('\n\" ') for s in text[j * 50:(j + 1) * 50]]
                if len(candidates) != 50 or len(set(candidates)) != 50 or any(s not in catalog for s in candidates):
                    raise ValueError('Constrained evaluation returned invalid/duplicate/missing candidates')
                record = {'sample_id': row['sample_id'], 'user_id': row['user_id'],
                    'target_position': int(row['target_position']), 'history_item_id': ast.literal_eval(row['history_item_id']),
                    'target_item_id': row['item_id'], 'target_sid': row['item_sid'],
                    'predictions': candidates, 'predicted_item_id_groups': [catalog[s] for s in candidates],
                    'scores': scores[j * 50:(j + 1) * 50]}
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
    dist.barrier()
    if rank == 0:
        combined = []
        for r in range(4):
            with (output / f'{args.split}.rank{r}.jsonl').open() as f:
                combined.extend(json.loads(line) for line in f)
        combined.sort(key=lambda r: int(r['sample_id'].split(':')[1]))
        if [r['sample_id'] for r in combined] != [r['sample_id'] for r in all_rows]:
            raise ValueError('Evaluation shard coverage mismatch')
        with (output / f'{args.split}.predictions.jsonl').open('w') as f:
            for row in combined:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        write_json(output / f'{args.split}.metrics.json', {**metrics(combined),
            'model': args.model, 'split': args.split, 'invalid_candidates': 0,
            'generation_config': conf.to_dict(), 'selection_use': 'final reporting only', 'metric_unit': 'official SID match; collisions preserved'})
        # Merged predictions are authoritative; avoid doubling the portable artifact.
        for r in range(4):
            (output / f'{args.split}.rank{r}.jsonl').unlink()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
