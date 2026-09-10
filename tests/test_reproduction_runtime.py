"""CPU lifecycle tests using the real official ReReTrainer, no downloaded weights."""
import json
import os
from pathlib import Path
import tempfile
import unittest

try:
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
    from datasets import Dataset
    from trl import GRPOConfig
    from minionerec_trainer import ReReTrainer, RepeatRandomSampler
    HAS_RUNTIME = True
except ImportError:
    HAS_RUNTIME = False

from reproduction.contracts import SIDTrie, ranking_rewards, validate_groups


@unittest.skipUnless(HAS_RUNTIME, 'Install requirements-reproduction.txt for lifecycle tests')
class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def fixture(self, tmp):
        raw = Tokenizer(models.BPE(unk_token='[UNK]'))
        raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        raw.decoder = decoders.ByteLevel()
        raw.train_from_iterator(['### User Input:\nhello\n\n### Response:\n', 'x y z'], trainers.BpeTrainer(vocab_size=300,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=['[UNK]', '[EOS]']))
        tok = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token='[UNK]', eos_token='[EOS]', pad_token='[EOS]', padding_side='left')
        sids = [f'<a_{i}><b_0><c_0>' for i in range(20)]
        tok.add_tokens([f'<a_{i}>' for i in range(20)] + ['<b_0>', '<c_0>'])
        tok.save_pretrained(tmp)
        cfg = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=16,
            max_position_embeddings=1024, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
            tie_word_embeddings=True)
        model = Qwen3ForCausalLM(cfg)
        model.save_pretrained(tmp)
        info = Path(tmp) / 'info.txt'
        info.write_text(''.join(f'{s}\ttitle\t{i}\n' for i, s in enumerate(sids)))
        return tok, model, sids, info

    def test_trie_generation_and_official_trainer_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            tok, model, sids, info = self.fixture(tmp)
            trie = SIDTrie(tok, sids)
            # Reject malformed continuation instead of silently forcing EOS.
            with self.assertRaisesRegex(ValueError, 'Invalid generated'):
                trie.allowed([tok.unk_token_id])
            rows = [{'prompt': '### Response:\n', 'target': sids[i] + '\n', 'sample_id': str(i)} for i in range(4)]
            def rule_reward(prompts, completions, target, **kwargs):
                return ranking_rewards(completions, target, 16)[0]
            def ndcg_rule_reward(prompts, completions, target, **kwargs):
                return ranking_rewards(completions, target, 16)[1]
            def make(output, steps):
                conf = GRPOConfig(output_dir=str(output), use_cpu=True, bf16=False, fp16=False,
                    per_device_train_batch_size=16, per_device_eval_batch_size=16, num_generations=16,
                    gradient_accumulation_steps=2, max_steps=steps, optim='adamw_torch',
                    gradient_checkpointing=False, report_to='none', save_steps=1, eval_strategy='steps',
                    eval_steps=1, logging_steps=1, max_completion_length=trie.max_new_tokens, beta=.001,
                    sync_ref_model=True, ref_model_sync_steps=1, seed=42, disable_tqdm=True,
                    model_init_kwargs={'torch_dtype': 'float32'},
                    load_best_model_at_end=True, metric_for_best_model='eval_reward', greater_is_better=True)
                conf.max_prompt_length = 64
                return ReReTrainer(model=str(tmp), base_model=str(tmp), processing_class=tok, args=conf,
                    reward_funcs=[rule_reward, ndcg_rule_reward], train_dataset=Dataset.from_list(rows),
                    eval_dataset=Dataset.from_list(rows[:2]), beam_search=True, test_during_training=False,
                    info_file=str(info))
            first = make(Path(tmp) / 'run', 1)
            # Same metadata write as the reproduction entrypoint, before training.
            from reproduction.prepare import write_json
            from reproduction.runtime import training_record
            write_json(Path(tmp) / 'training_args.json', first.args.to_dict())
            first.train()
            training_record(first, tmp, Path(tmp) / 'run', str(tmp))
            saved = Path(tmp) / 'run/checkpoint-1'
            self.assertTrue((saved / 'reference_model.pt').is_file())
            self.assertEqual(first.state.global_step, 1)
            self.assertTrue(any('eval_reward' in log for log in first.state.log_history))
            expected = {k: v.clone() for k, v in first.ref_model.state_dict().items()}
            second = make(Path(tmp) / 'run', 2)
            second._load_reference_checkpoint(saved)
            for key, value in second.ref_model.state_dict().items():
                self.assertTrue(torch.equal(value, expected[key]))
            second.train(resume_from_checkpoint=str(saved))
            self.assertEqual(second.state.global_step, 2)
            # Actual local candidate generation / grouping and overflow rejection.
            inputs = [rows[0]] * 16
            second._prepare_inputs(inputs)
            second.max_prompt_length = 1
            with self.assertRaisesRegex(ValueError, 'no truncation'):
                second._prepare_inputs(inputs)

    def test_rl_bfloat16_init_preserves_serializable_config(self):
        from reproduction.prepare import write_json
        with tempfile.TemporaryDirectory() as tmp:
            tok, model, sids, info = self.fixture(tmp)
            original_kwargs = {'torch_dtype': 'bfloat16'}
            conf = GRPOConfig(output_dir=str(Path(tmp) / 'run'), use_cpu=True,
                bf16=False, fp16=False, per_device_train_batch_size=16,
                per_device_eval_batch_size=16, num_generations=16,
                model_init_kwargs=original_kwargs, gradient_checkpointing=True,
                gradient_checkpointing_kwargs={'use_reentrant': False},
                report_to='none', max_completion_length=5)
            rows = [{'prompt': '### Response:\n', 'target': sids[0] + '\n', 'sample_id': '0'}]
            def reward(prompts, completions, target, **kwargs):
                return ranking_rewards(completions, target)[0]
            trainer = ReReTrainer(model=str(tmp), base_model=str(tmp), processing_class=tok,
                args=conf, train_dataset=Dataset.from_list(rows), reward_funcs=[reward],
                beam_search=True, test_during_training=False, info_file=str(info))
            self.assertEqual(trainer.model.dtype, torch.bfloat16)
            self.assertEqual(trainer.ref_model.dtype, torch.bfloat16)
            self.assertFalse(trainer.model.config.use_cache)
            # This used to fail: model loading mutated the config to torch.dtype.
            write_json(Path(tmp) / 'training_args.json', trainer.args.to_dict())
            self.assertEqual(original_kwargs, {'torch_dtype': 'bfloat16'})
            recorded = json.loads((Path(tmp) / 'training_args.json').read_text())
            self.assertEqual(recorded['model_init_kwargs'], original_kwargs)

    def test_sampler_four_rank_layout_and_resume_epoch(self):
        from accelerate.data_loader import BatchSamplerShard
        from torch.utils.data import BatchSampler
        ds = list(range(7))
        for micro in (16, 32, 64):
            ranks = []
            for rank in range(4):
                sampler = RepeatRandomSampler(ds, 16, seed=42)
                sampler.set_epoch(2)
                batches = BatchSamplerShard(BatchSampler(sampler, micro, drop_last=False),
                    num_processes=4, process_index=rank, split_batches=False, even_batches=True)
                ranks.append(list(batches))
            self.assertEqual(len({len(r) for r in ranks}), 1)
            for rank in ranks:
                for batch in rank:
                    self.assertEqual(len(batch), micro)
                    for start in range(0, len(batch), 16):
                        self.assertEqual(len(set(batch[start:start + 16])), 1)
            sampler = RepeatRandomSampler(ds, 16, seed=42)
            sampler.set_epoch(3)
            self.assertEqual(list(sampler), list(sampler))

    def test_larger_micro_batch_resume_uses_new_batch_and_next_prompts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tok, model, sids, info = self.fixture(tmp)
            rows = [{'prompt': '### Response:\n', 'target': sids[i] + '\n', 'sample_id': str(i)} for i in range(12)]
            seen = []
            def reward(prompts, completions, target, sample_id, **kwargs):
                seen.extend(sample_id[::16])
                return ranking_rewards(completions, target)[0]
            def make(micro, accum, steps):
                conf = GRPOConfig(output_dir=str(Path(tmp) / 'run'), use_cpu=True,
                    bf16=False, fp16=False, per_device_train_batch_size=micro,
                    per_device_eval_batch_size=micro, num_generations=16,
                    gradient_accumulation_steps=accum, max_steps=steps,
                    optim='adamw_torch', gradient_checkpointing=False,
                    report_to='none', save_steps=1, logging_steps=1,
                    max_completion_length=5, seed=42, disable_tqdm=True,
                    model_init_kwargs={'torch_dtype': 'float32'})
                conf.max_prompt_length = 64
                return ReReTrainer(model=str(tmp), base_model=str(tmp), processing_class=tok,
                    args=conf, train_dataset=Dataset.from_list(rows), reward_funcs=[reward],
                    beam_search=True, test_during_training=False, info_file=str(info))
            first = make(16, 4, 1)
            first.train()
            self.assertEqual(len(seen), 4)
            checkpoint = Path(tmp) / 'run/checkpoint-1'
            state_path = checkpoint / 'trainer_state.json'
            state = json.loads(state_path.read_text())
            self.assertEqual(state['train_batch_size'], 16)
            state['train_batch_size'] = 32  # same archived JSON change as migration tool
            state_path.write_text(json.dumps(state))
            seen.clear()
            second = make(32, 2, 2)
            second.train(resume_from_checkpoint=str(checkpoint))
            self.assertEqual(second._train_batch_size, 32)
            self.assertEqual(second.state.global_step, 2)
            expected = list(RepeatRandomSampler(rows, 16, seed=42))[64:128:16]
            self.assertEqual(seen, [str(i) for i in expected])

    def test_four_rank_update_groups_unchanged_with_larger_micro(self):
        import math
        from accelerate.data_loader import BatchSamplerShard
        from torch.utils.data import BatchSampler
        # Includes incomplete final batches, and a full update in the next epoch.
        for count in (65, 128, 130):
            for epoch in (0, 1):
                layouts = {}
                for micro in (16, 32, 64):
                    accum = 256 // micro
                    ranks = []
                    for rank in range(4):
                        sampler = RepeatRandomSampler(list(range(count)), 16, seed=42)
                        sampler.set_epoch(epoch)
                        ranks.append(list(BatchSamplerShard(BatchSampler(sampler, micro, False),
                            num_processes=4, process_index=rank, split_batches=False, even_batches=True)))
                    updates = math.ceil(len(ranks[0]) / accum)
                    self.assertEqual(updates, math.ceil(count / 64))
                    layouts[micro] = []
                    for step in range(count // 64):
                        groups = []
                        for rank in ranks:
                            for batch in rank[step * accum:(step + 1) * accum]:
                                for start in range(0, len(batch), 16):
                                    self.assertEqual(len(set(batch[start:start + 16])), 1)
                                groups.extend(batch[::16])
                        layouts[micro].append(sorted(groups))
                self.assertEqual(layouts[16], layouts[32])
                self.assertEqual(layouts[16], layouts[64])

    def test_sft_step_and_strict_length(self):
        from data import SidSFTDataset
        import csv
        with tempfile.TemporaryDirectory() as tmp:
            tok, model, sids, info = self.fixture(tmp)
            path = Path(tmp) / 'data.csv'
            with path.open('w') as f:
                w = csv.DictWriter(f, fieldnames=['history_item_sid', 'item_sid'])
                w.writeheader()
                w.writerow({'history_item_sid': repr(sids[:2]), 'item_sid': sids[2]})
            with self.assertRaisesRegex(ValueError, 'exceed max_len'):
                SidSFTDataset(str(path), tok, max_len=4)
            ds = SidSFTDataset(str(path), tok, max_len=1024)
            row = ds[0]
            self.assertEqual(row['labels'][-1], tok.eos_token_id)
            trainer = Trainer(model=model, args=TrainingArguments(output_dir=str(Path(tmp) / 'sft'),
                use_cpu=True, max_steps=1, report_to='none', save_strategy='no', per_device_train_batch_size=1),
                train_dataset=Dataset.from_list([row]), data_collator=DataCollatorForSeq2Seq(tok))
            trainer.train()
            self.assertEqual(trainer.state.global_step, 1)

@unittest.skipUnless(HAS_RUNTIME and os.environ.get('QWEN3_TOKENIZER'), 'Set QWEN3_TOKENIZER to a local original Qwen3 tokenizer')
class RealQwenTokenizerTests(unittest.TestCase):
    def test_prompt_terminal_and_beam50(self):
        from transformers import AutoTokenizer, GenerationConfig, LogitsProcessorList
        from data import SidSFTDataset
        from reproduction.contracts import check_tokenizer, next_prompt
        import ast
        import csv
        torch.set_num_threads(1)
        root = Path(__file__).resolve().parents[1]
        category = 'Office_Products'
        tokenizer = AutoTokenizer.from_pretrained(os.environ['QWEN3_TOKENIZER'], padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        indices = json.loads((root / f'data/Amazon/index/{category}.index.json').read_text())
        tokenizer.add_tokens(sorted({t for sid in indices.values() for t in sid}))
        check_tokenizer(tokenizer, indices)
        source = root / f'data/Amazon/train/{category}_5_2016-10-2018-11.csv'
        with source.open() as f:
            row = next(csv.DictReader(f))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'one.csv'
            with path.open('w') as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                w.writeheader()
                w.writerow(row)
            sft = SidSFTDataset(str(path), tokenizer, max_len=512)[0]
            prefix = next_prompt(ast.literal_eval(row['history_item_sid']))
            ids = tokenizer.encode(prefix, add_special_tokens=False)
            self.assertEqual(sft['input_ids'][:len(ids)], ids)
            self.assertEqual(sft['labels'][len(ids):], tokenizer.encode(row['item_sid'] + '\n', add_special_tokens=False) + [tokenizer.eos_token_id])
        trie = SIDTrie(tokenizer, {''.join(s) for s in indices.values()})
        self.assertEqual(trie.max_new_tokens, 5)
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tokenizer), hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1, head_dim=16,
            max_position_embeddings=512, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)).eval()
        check_tokenizer(tokenizer, indices, model)
        x = tokenizer(['### Response:\n', prefix], padding=True, return_tensors='pt', add_special_tokens=False)
        with torch.inference_mode():
            generated = model.generate(**x, generation_config=GenerationConfig(num_beams=50, num_return_sequences=50,
                do_sample=False, max_new_tokens=5, length_penalty=0, eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id), logits_processor=LogitsProcessorList([trie.processor(x.input_ids.shape[1])]))
        decoded = tokenizer.batch_decode(generated[:, x.input_ids.shape[1]:], skip_special_tokens=True)
        catalog = {''.join(v) for v in indices.values()}
        self.assertEqual(len(decoded), 100)
        for start in (0, 50):
            self.assertEqual(len(set(decoded[start:start + 50])), 50)
        self.assertTrue(all(s.endswith('\n') and s.strip() in catalog for s in decoded))


if __name__ == '__main__':
    unittest.main()
