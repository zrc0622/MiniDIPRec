import ast
import json
import math
from pathlib import Path
import tempfile
import unittest
from reproduction.prepare import recover, prepare, CATEGORIES
from reproduction.contracts import ranking_rewards, validate_groups
from reproduction.evaluate import metrics


class ReproductionTests(unittest.TestCase):
    def fixture(self, n=62):
        idx = {str(i): [f'<a_{i}>', '<b_0>', '<c_0>'] for i in range(n)}
        items = {str(i): {'title': f'item {i}'} for i in range(n)}
        rows = []
        for i in range(1, n):
            h = list(range(max(0, i - 10), i))
            rows.append(dict(user_id='u', history_item_id=repr(h), history_item_sid=repr([''.join(idx[str(j)]) for j in h]),
                history_item_title=repr([items[str(j)]['title'] for j in h]), item_id=str(i),
                item_sid=''.join(idx[str(i)]), item_title=items[str(i)]['title']))
        return {'train': rows[:40], 'valid': rows[40:50], 'test': rows[50:]}, idx, items

    def test_restore_history_and_split(self):
        splits, idx, items = self.fixture()
        out, stats = recover(splits, idx, items)
        self.assertEqual(ast.literal_eval(out['test'][-1]['history_item_id']), list(range(11, 61)))
        self.assertEqual(ast.literal_eval(out['valid'][0]['history_item_id']), list(range(41)))
        for sp, rows in splits.items():
            self.assertEqual(len(rows), len(out[sp]))
            for old, new in zip(rows, out[sp]):
                for key in ('item_id', 'item_sid', 'item_title', 'user_id'):
                    self.assertEqual(old[key], new[key])
        self.assertEqual(stats['users'], 1)

    def test_missing_window_fails(self):
        splits, idx, items = self.fixture()
        del splits['train'][30]
        with self.assertRaisesRegex(ValueError, 'sliding window'):
            recover(splits, idx, items)

    def test_missing_start_and_misalignment_fail(self):
        splits, idx, items = self.fixture()
        splits['train'].pop(0)
        with self.assertRaisesRegex(ValueError, 'beginning'):
            recover(splits, idx, items)
        splits, idx, items = self.fixture()
        splits['train'][0]['history_item_title'] = "['wrong']"
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            recover(splits, idx, items)

    def test_duplicate_keys_do_not_control_reward(self):
        rows = [{'prompt': 'same', 'target': target, 'sample_id': key} for key, target in [('a', 'A'), ('b', 'B')] for _ in range(16)]
        validate_groups(rows, 16)
        outputs = ['A'] + ['miss'] * 15 + ['B'] + ['miss'] * 15
        exact, rank = ranking_rewards(outputs, [r['target'] for r in rows])
        self.assertEqual(sum(exact), 2)
        self.assertEqual(rank[0], 0)
        self.assertLess(rank[1], 0)
        rows[1]['sample_id'] = 'b'
        with self.assertRaisesRegex(ValueError, 'misgrouped'):
            validate_groups(rows, 16)

    def test_official_reward_formula(self):
        exact, ranks = ranking_rewards(['x'] * 16, ['y'] * 16)
        self.assertEqual(sum(exact + ranks), 0)
        exact, ranks = ranking_rewards(['y'] + ['x'] * 15, ['y'] * 16)
        self.assertAlmostEqual(ranks[1], -(1 / math.log2(3)) / sum(1 / math.log2(i + 2) for i in range(16)))

    def test_metrics(self):
        result = metrics([{'target_sid': 'a', 'predictions': ['a']}, {'target_sid': 'b', 'predictions': ['a', 'b']}])
        self.assertEqual(result['HR@5'], result['Recall@5'])
        self.assertEqual(result['HR@10'], 1)
        self.assertAlmostEqual(result['NDCG@5'], (1 + 1 / math.log2(3)) / 2)

    def test_full_official_data(self):
        source = Path(__file__).resolve().parents[1] / 'data/Amazon'
        with tempfile.TemporaryDirectory() as tmp:
            for category in CATEGORIES:
                dst = Path(tmp) / category
                report = prepare(source, dst, category)
                self.assertGreater(report['stats']['test']['extended_samples'], 0)
                self.assertTrue(any(50 in report['stats'][sp]['history_lengths'] for sp in ('train', 'valid', 'test')))
                self.assertEqual(prepare(source, dst, category)['source_sha256'], report['source_sha256'])
                with (dst / 'train.csv').open('a') as f:
                    f.write('\n')
                with self.assertRaisesRegex(ValueError, 'Generated data changed'):
                    prepare(source, dst, category)

class RunnerContractTests(unittest.TestCase):
    def test_dry_run_stage_order_and_parent_models(self):
        import subprocess
        import sys
        from unittest.mock import patch
        from reproduction import run
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / 'reproduction').mkdir()
            (repo / 'config').mkdir()
            (repo / 'scripts').mkdir()
            for file in ['data.py', 'minionerec_trainer.py', 'requirements-reproduction.txt', 'config/zero2_opt.yaml', 'scripts/reproduce.sh']:
                (repo / file).write_text('fixture')
            def fake_prepare(source, destination, category):
                destination.mkdir(parents=True)
                return {'stats': {'users': 1}}
            old_cwd = Path.cwd()
            try:
                with patch.object(run, 'REPO', repo), patch.object(run, 'prepare', fake_prepare), patch.object(sys, 'argv',
                        ['run', '--run-name', 'fixture', '--gpus', '2,3,6,7', '--dry-run']), patch.object(subprocess, 'Popen', side_effect=AssertionError('dry-run executed a subprocess')):
                    run.main()
                commands = [json.loads(line) for line in (repo / 'results/fixture/commands.jsonl').read_text().splitlines()]
                trained = [c['argv'] for c in commands if 'reproduction.train' in c['argv']]
                self.assertEqual([c[c.index('--category') + 1] for c in trained], [CATEGORIES[0]] * 2 + [CATEGORIES[1]] * 2)
                self.assertEqual([c[c.index('--stage') + 1] for c in trained], ['sft', 'rl', 'sft', 'rl'])
                for i in (0, 2):
                    self.assertEqual(trained[i][trained[i].index('--model') + 1], 'Qwen/Qwen3-0.6B')
                    self.assertIn(CATEGORIES[i // 2] + '/sft/selected_model', trained[i + 1][trained[i + 1].index('--model') + 1])
                self.assertTrue(all(c['env']['CUDA_VISIBLE_DEVICES'] == '2,3,6,7' for c in commands))
                evals = [c for c in commands if 'reproduction.evaluate' in c['argv']]
                self.assertEqual(len(evals), 8)
                self.assertEqual(json.loads((repo / 'results/fixture/summary.json').read_text()), [])
            finally:
                import os
                os.chdir(old_cwd)

    def test_reference_required_for_rl_resume(self):
        from reproduction.run import latest_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for step in (3, 6):
                (root / f'checkpoint-{step}').mkdir()
                (root / f'checkpoint-{step}/trainer_state.json').write_text('{}')
            (root / 'checkpoint-3/reference_model.pt').write_text('fixture')
            self.assertTrue(latest_checkpoint(root).endswith('checkpoint-6'))
            self.assertTrue(latest_checkpoint(root, require_reference=True).endswith('checkpoint-3'))


if __name__ == '__main__':
    unittest.main()
