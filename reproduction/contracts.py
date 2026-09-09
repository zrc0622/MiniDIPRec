"""Shared official prompts, SID terminal paths and row-local ranking rewards."""
import math

NEXT_INSTRUCTION = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Can you predict the next possible item that the user may expect?

'''
ITEM_INSTRUCTION = '''Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

### Instruction:
Answer the question about item identification.

'''


def next_prompt(history_sids):
    history = ', '.join(history_sids)
    return NEXT_INSTRUCTION + f'### User Input: \nThe user has interacted with items {history} in chronological order. Can you predict the next possible item that the user may expect?\n\n### Response:\n'


def validate_groups(rows, generations):
    if not rows or len(rows) % generations:
        raise ValueError('Every local RL batch must contain complete candidate groups')
    for start in range(0, len(rows), generations):
        group = rows[start:start + generations]
        keys = [(r['prompt'], r['target'], r['sample_id']) for r in group]
        if any(k != keys[0] for k in keys):
            raise ValueError(f'RL candidates are misgrouped at local offset {start}')


def ranking_rewards(completions, targets, generations=16):
    if len(completions) != len(targets) or len(targets) % generations:
        raise ValueError('Incomplete ranking reward group')
    weights = [1 / math.log2(i + 2) for i in range(generations)]
    total = sum(weights)
    exact, ranking = [], []
    for start in range(0, len(targets), generations):
        if len(set(targets[start:start + generations])) != 1:
            raise ValueError('Mixed targets in ranking reward group')
        hits = [c.strip('\n\" ') == t.strip('\n\" ') for c, t in
                zip(completions[start:start + generations], targets[start:start + generations])]
        exact.extend(float(h) for h in hits)
        # Official formula: zero for hit, negative normalized rank penalty for miss,
        # and all zeros when none of the G candidates hit.
        ranking.extend(0.0 if h or not any(hits) else -weights[i] / total for i, h in enumerate(hits))
    return exact, ranking


class SIDTrie:
    def __init__(self, tokenizer, semantic_ids):
        self.children = {}
        self.terminals = set()
        self.eos = tokenizer.eos_token_id
        if self.eos is None:
            raise ValueError('Tokenizer must define EOS')
        lengths = set()
        for sid in semantic_ids:
            ids = tokenizer.encode(sid.strip() + '\n', add_special_tokens=False) + [self.eos]
            lengths.add(len(ids))
            self.terminals.add(tuple(ids))
            decoded = tokenizer.decode(ids[:-1], skip_special_tokens=False)
            if decoded != sid.strip() + '\n':
                raise ValueError(f'SID/newline token round-trip failed: {sid!r} -> {decoded!r}')
            for i, token in enumerate(ids):
                self.children.setdefault(tuple(ids[:i]), set()).add(token)
        if not lengths:
            raise ValueError('Empty SID catalog')
        self.max_new_tokens = max(lengths)

    def allowed(self, prefix):
        prefix = tuple(prefix)
        # Finished beams can be padded with EOS by generation.
        if self.eos in prefix:
            stop = prefix.index(self.eos) + 1
            if prefix[:stop] not in self.terminals or any(t != self.eos for t in prefix[stop:]):
                raise ValueError(f'Invalid generated SID terminal: {prefix}')
            return [self.eos]
        result = sorted(self.children.get(prefix, ()))
        if not result:
            raise ValueError(f'Invalid generated SID prefix: {prefix}')
        return result

    def processor(self, prompt_length):
        import torch
        from transformers import LogitsProcessor
        trie = self

        class Processor(LogitsProcessor):
            def __call__(self, input_ids, scores):
                mask = torch.full_like(scores, -torch.inf)
                for i, row in enumerate(input_ids):
                    mask[i, trie.allowed(row[prompt_length:].tolist())] = 0
                # Match official constrained processor's log-softmax-before-mask.
                return torch.log_softmax(scores, dim=-1) + mask
        return Processor()


def check_tokenizer(tokenizer, indices, model=None):
    tokens = sorted({token for sid in indices.values() for token in sid})
    ids = [tokenizer.encode(t, add_special_tokens=False) for t in tokens]
    if any(len(v) != 1 for v in ids) or len({v[0] for v in ids}) != len(tokens):
        raise ValueError('SID tokens must be unique single vocabulary entries')
    for sid in indices.values():
        if tokenizer.encode(''.join(sid), add_special_tokens=False) != [tokenizer.convert_tokens_to_ids(t) for t in sid]:
            raise ValueError('Adjacent SID tokenization is inconsistent')
    if model is not None:
        if model.get_input_embeddings().weight.shape[0] != len(tokenizer) or model.get_output_embeddings().weight.shape[0] != len(tokenizer):
            raise ValueError('Tokenizer, input embedding and output head sizes disagree')
    return {'vocab_size': len(tokenizer), 'sid_tokens': len(tokens),
            'eos_token_id': tokenizer.eos_token_id, 'pad_token_id': tokenizer.pad_token_id}
