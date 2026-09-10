import csv
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from reproduction import run
from reproduction.prepare import CATEGORIES, SUFFIX, prepare, sha256, write_json
from scripts.start_rl_from_sft import create_run
from tests import test_reproduction, test_rl_config_migration


class FreshRLTests(unittest.TestCase):
    def fixture(self, tmp):
        repo, source, _ = test_rl_config_migration.MigrationTests().fixture(tmp)
        category = CATEGORIES[0]
        cat = source / category
        sft = cat / 'sft'
        sft.mkdir(parents=True)
        checkpoint_root = repo / 'checkpoints/run'
        model = checkpoint_root / category / 'sft/selected_model'
        model.mkdir(parents=True)
        for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
            (model / name).write_text('{}')
        (model / 'model.safetensors').write_bytes(b'test weight fixture')
        for name in ('sft.complete.json', 'eval-sft.complete.json'):
            write_json(sft / name, {'completed': True})
        for split in ('valid', 'test'):
            write_json(sft / f'{split}.metrics.json', {
                'samples': 1, **{name: 0.5 for name in ('HR@5', 'Recall@5', 'HR@10', 'Recall@10', 'NDCG@5', 'NDCG@10')}})
            (sft / f'{split}.predictions.jsonl').write_text('{}\n')
        write_json(sft / 'training.json', {'final_checkpoint': str(model), 'selected_step': 42,
            'best_checkpoint': str(model.parent / 'checkpoint-42')})
        original_args = run.parser().parse_args(['--run-name', 'run'])
        config = {k: v for k, v in vars(original_args).items() if k not in ('stage', 'dataset', 'resume', 'dry_run')}
        config.update(checkpoint_root=str(checkpoint_root), world_size=4,
            effective_batch=1024, rl_generations=16, sft_accumulation=64,
            rl_accumulation=16, recipe='official-5f4f733-qwen3-history50-v1')
        write_json(source / 'run_config.json', config)
        splits, indices, items = test_reproduction.ReproductionTests().fixture()
        data = repo / 'data/Amazon'
        for split, rows in splits.items():
            path = data / split / f'{category}{SUFFIX}.csv'
            path.parent.mkdir(parents=True)
            with path.open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        write_json(data / f'index/{category}.index.json', indices)
        write_json(data / f'index/{category}.item.json', items)
        (data / 'info').mkdir()
        (data / f'info/{category}{SUFFIX}.txt').write_text(''.join(''.join(sid) + '\titem\n' for sid in indices.values()))
        prepare(data, cat / 'data', category)
        write_json(cat / 'lengths.json', {'model': config['model'], 'data_audit_sha256': sha256(cat / 'data/audit.json')})
        old_rl = checkpoint_root / category / 'rl/checkpoint-100'
        old_rl.mkdir(parents=True)
        write_json(old_rl / 'trainer_state.json', {'global_step': 100})
        (old_rl / 'reference_model.pt').write_bytes(b'old RL reference')
        (cat / 'rl').mkdir()
        (cat / 'rl/train.log').write_text('previous RL')
        return repo, source, model

    def test_import_then_runner_starts_fresh_rl_without_sft(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, source, model = self.fixture(tmp)
            before = {str(p): sha256(p) for p in source.rglob('*') if p.is_file()}
            command = create_run(repo, 'run', 'rl32', CATEGORIES[0], 32)
            root = repo / 'results/rl32'
            copied_model = repo / f'checkpoints/rl32/{CATEGORIES[0]}/sft/selected_model'
            self.assertEqual((copied_model / 'model.safetensors').read_bytes(), (model / 'model.safetensors').read_bytes())
            self.assertNotEqual((copied_model / 'model.safetensors').stat().st_ino, (model / 'model.safetensors').stat().st_ino)
            self.assertFalse(any(p.is_symlink() for p in root.rglob('*')))
            self.assertFalse((root / CATEGORIES[0] / 'rl').exists())
            self.assertFalse((copied_model.parent.parent / 'rl').exists())
            self.assertEqual(before, {str(p): sha256(p) for p in source.rglob('*') if p.is_file()})
            saved = Path.cwd()
            try:
                with patch.object(run, 'REPO', repo), patch.object(sys, 'argv', ['run', *command[3:], '--dry-run']):
                    run.main()
            finally:
                os.chdir(saved)
            calls = [json.loads(line)['argv'] for line in (root / 'commands.jsonl').read_text().splitlines()]
            train_calls = [cmd for cmd in calls if 'reproduction.train' in cmd]
            self.assertEqual(len(train_calls), 1)
            train = train_calls[0]
            self.assertEqual(train[train.index('--stage') + 1], 'rl')
            self.assertEqual(train[train.index('--model') + 1], str(copied_model.resolve()))
            self.assertEqual(train[train.index('--micro-batch') + 1], '32')
            self.assertNotIn('--resume', train)
            self.assertEqual(len([cmd for cmd in calls if 'reproduction.evaluate' in cmd]), 2)
            self.assertEqual(len(json.loads((root / 'summary.json').read_text())), 1)

    def test_existing_destination_and_missing_model_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, source, model = self.fixture(tmp)
            with self.assertRaisesRegex(ValueError, 'distinct run name'):
                create_run(repo, 'run', 'run', CATEGORIES[0])
            (repo / 'results/taken').mkdir()
            with self.assertRaisesRegex(ValueError, 'Destination already exists'):
                create_run(repo, 'run', 'taken', CATEGORIES[0])
            (model / 'model.safetensors').unlink()
            with self.assertRaisesRegex(ValueError, 'Missing SFT model weights'):
                create_run(repo, 'run', 'fresh', CATEGORIES[0])
            self.assertFalse((repo / 'results/fresh').exists())

    def test_failed_copy_does_not_leave_new_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, source, model = self.fixture(tmp)
            with patch('scripts.start_rl_from_sft.copy_files', side_effect=OSError('disk full')):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    create_run(repo, 'run', 'fresh', CATEGORIES[0])
            self.assertFalse((repo / 'results/fresh').exists())
            self.assertFalse((repo / 'checkpoints/fresh').exists())
            self.assertFalse(list((repo / 'results').glob('.sft-import-*')))
            self.assertFalse(list((repo / 'checkpoints').glob('.sft-import-*')))


if __name__ == '__main__':
    unittest.main()
