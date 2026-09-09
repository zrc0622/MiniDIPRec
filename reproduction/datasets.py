"""Assemble only the data tasks actually enabled by official sft.py / rl.py."""
from pathlib import Path
from reproduction.contracts import NEXT_INSTRUCTION, ITEM_INSTRUCTION


def sft_data(root, category, tokenizer, max_len):
    from data import SidSFTDataset, SidItemFeatDataset, FusionSeqRecDataset
    root = Path(root)
    common = dict(tokenizer=tokenizer, max_len=max_len, seed=42)
    meta = dict(item_file=str(root / f'{category}.item.json'), index_file=str(root / f'{category}.index.json'))
    train = [SidSFTDataset(str(root / 'train.csv'), **common),
             SidItemFeatDataset(**meta, **common),
             FusionSeqRecDataset(str(root / 'train.csv'), **meta, **common)]
    valid = SidSFTDataset(str(root / 'valid.csv'), **common)
    return train, valid


def rl_data(root, category, title_sample=10000):
    from data import SidDataset, RLTitle2SidDataset, RLSeqTitle2SidDataset
    root = Path(root)
    # The official title-history sample uses pandas random_state=0 (its constructor default).
    train = [SidDataset(str(root / 'train.csv')),
             RLTitle2SidDataset(str(root / f'{category}.item.json'), str(root / f'{category}.index.json')),
             RLSeqTitle2SidDataset(str(root / 'train.csv'), sample=title_sample)]
    valid = SidDataset(str(root / 'valid.csv'))

    def rows(datasets, split):
        result = []
        for task_id, dataset in enumerate(datasets):
            prefix = ITEM_INSTRUCTION if task_id == 1 else NEXT_INSTRUCTION
            for i, row in enumerate(dataset):
                result.append({'prompt': prefix + row['prompt'], 'target': row['completion'],
                               'sample_id': f'{split}:{task_id}:{i}'})
        return result
    return rows(train, 'train'), rows([valid], 'valid')
