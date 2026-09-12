# Pinned upstream sources

Source: https://github.com/AkaliKong/MiniOneRec
Commit: `0c64b955ecb8e3d7a9ae9f1fa88cf938f129b0ed`

The `.txt` files are byte-for-byte upstream sources, pinned by `manifest.json`.
Apache-2.0 license is included. These are provenance snapshots, not independent
training entrypoints. `scripts/five_utils.py` loads the upstream data classes from
`data.py.txt`; at load time it replaces `eval` with literal parsing and guards empty
token lists. Dictionary deduplication, task sampling, Fusion SID lookup and prompts
are retained. No snapshot source is modified at runtime.

Runs 03–05 use the official enabled tasks and launcher recipe through
`scripts/train_rl_five.py`, with the repository's compatible ReReTrainer. They are
explicitly **350-update adapted reproductions**, not untouched upstream execution:

- Four GPU micro batches/accumulation preserve the original candidate batch1024.
- Original train/valid CSV fields and history10 are retained; audit IDs are added.
- SFT cutoff512, task classes, LR3e-4, linear/warmup20, max10epochs and patience3
  are retained, including both upstream validation shuffles. Two checkpoints
  retain best/latest for recovery.
- RL uses raw upstream task prompts, the upstream seed42 dataset shuffle,
  beta.001, beam sampling, G16, LR1e-5, paged AdamW, full two-epoch cosine/warmup3%.
- Official RL leaves model initialization dtype unspecified; the adapter does too
  and records actual policy/reference dtype. This is not the discarded FP32-loss
  experiment. Runs01/02 retain their original explicitBF16 loading.
- Row-local reward targets replace upstream dictionaries merged across train and
  validation. This prevents target overwrites; the exact+ranking scalar formula
  is unchanged. No label injection, CE loss, task rebalance or FP32 math override.
- Model/tokenizer SID registration and EOS/trie are checked for both Qwen families.
  Model generation defaults cannot overwrite the explicit recipe. Actual legal
  completions are5tokens; prompt limits come from train/valid scans rather than
  silent truncation. Cache/checkpointing, reference and distributed sampler resume
  use the existing tested compatibility fixes.
- The shared final evaluator always uses the SFT-aligned recommendation prompt
  and deterministic beam50 on complete validation data, equally for SFT and RL.
  This differs from the raw upstream RL prompt and is reported, not concealed.
- Model selection/metrics use validation only. No test data is read by the new suite.

Consequently, runs03 versus01 compare a recipe/implementation package; they are
not a single-parameter causal test. Runs04 versus03 control model family, and
runs05 versus04 assess dataset dependence via each dataset's own SFT→RL change.
