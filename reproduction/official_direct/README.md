# Direct upstream snapshot

Source: https://github.com/AkaliKong/MiniOneRec at
`0c64b955ecb8e3d7a9ae9f1fa88cf938f129b0ed`.
`manifest.json` records byte-identical upstream files and SHA256. Python/shell
sources use `.txt` here to avoid accidental imports. LICENSE is upstream Apache-2.0.

`scripts/official_utils.py` materializes a verified executable copy inside each
new suite's `shared/upstream/`. Its `runtime.patch` records every source change:

- Guard empty token lists in `BaseDataset.encode`.
- Read the removed TRL `max_prompt_length` field with `getattr(..., None)`;
  preflight verifies untruncated official prompts fit the original model context.
- Skip the auxiliary model unused by the ranking reward closures.
- Disable WandB reporting and Trainer progress bars. SFT retains two checkpoints
  for best-model selection plus restart, and restores its early-stop state.
- Replace redundant final/root model exports with one selected SFT export and
  RL's existing checkpoints; honor the selected evaluation GPU.

The wrapper calls the actual upstream `sft.train`, `rl.train`, `evaluate.main`
and `calc.gao`. Trainer instrumentation adds scalar logs, stop700, snapshots,
tokenizer/checkpoint completion markers and synchronized reference save/load.
It does not override sampling, generation, reward closures or loss computation.
AST regression checks cover these methods; constrained logits and calc files
remain byte-identical. Original CSV/SID files are copied without preprocessing.

The original three-task deduplication, Fusion dictionary lookups, and train/valid
reward dictionary overwrites are deliberately retained. Preflight reports
dictionary target disagreements. These are part of the upstream behavior being
measured; neither balanced task sampling nor prior data corrections apply.

This is a four-GPU, sub-1B partial reproduction: Qwen2.5-0.5B Base, original
effective batch1024, original two-epoch RL LR horizon, stopped at update700.
It is not the paper's larger model or a completed two-epoch RL reproduction.
Training uses the existing pinned reproduction environment plus the imports in
`requirements-official-direct.txt`, not the upstream's entire environment dump.

CPU verification uses tiny random Qwen2 weights and the real Qwen2.5 tokenizer.
The CPU harness substitutes optimizer/precision/budget and optional CLI imports;
production retains official BF16/paged AdamW and must be checked on CUDA.
