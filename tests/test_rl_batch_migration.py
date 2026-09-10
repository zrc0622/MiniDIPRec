import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.migrate_rl_batch import migrate_batch
from scripts.migrate_rl_config_fix import atomic_write
from tests import test_rl_config_migration as source_migration_tests


class BatchMigrationTests(unittest.TestCase):
    def fixture(self, tmp):
        repo, root, _ = source_migration_tests.MigrationTests().fixture(tmp)
        ckpts = repo / 'checkpoints/run'
        checkpoint = ckpts / 'Office_Products/rl/checkpoint-100'
        checkpoint.mkdir(parents=True)
        (checkpoint / 'trainer_state.json').write_text(json.dumps({
            'train_batch_size': 16, 'global_step': 100, 'epoch': 0.125}))
        (checkpoint / 'reference_model.pt').write_bytes(b'unchanged reference')
        (checkpoint / 'optimizer.pt').write_bytes(b'unchanged optimizer')
        config = {'checkpoint_root': str(ckpts), 'world_size': 4, 'rl_generations': 16,
            'effective_batch': 1024, 'rl_micro_batch': 16, 'rl_accumulation': 16,
            'sft_micro_batch': 4, 'sft_accumulation': 64}
        (root / 'run_config.json').write_text(json.dumps(config))
        artifact = root / 'Office_Products/rl'
        artifact.mkdir(parents=True)
        (artifact / 'training_args.json').write_text(json.dumps({
            'per_device_train_batch_size': 16, 'gradient_accumulation_steps': 16}))
        return repo, root, checkpoint

    def test_dry_run_then_migration_and_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, root, checkpoint = self.fixture(tmp)
            before = (root / 'run_config.json').read_bytes()
            old_state = (checkpoint / 'trainer_state.json').read_bytes()
            migrate_batch(repo, root, 32, dry_run=True)
            self.assertEqual((root / 'run_config.json').read_bytes(), before)
            self.assertEqual((checkpoint / 'trainer_state.json').read_bytes(), old_state)
            migrate_batch(repo, root, 32)
            updated = json.loads((root / 'run_config.json').read_text())
            self.assertEqual((updated['rl_micro_batch'], updated['rl_accumulation']), (32, 8))
            state = json.loads((checkpoint / 'trainer_state.json').read_text())
            self.assertEqual(state, {'train_batch_size': 32, 'global_step': 100, 'epoch': 0.125})
            self.assertEqual((checkpoint / 'reference_model.pt').read_bytes(), b'unchanged reference')
            self.assertEqual((checkpoint / 'optimizer.pt').read_bytes(), b'unchanged optimizer')
            self.assertEqual((root / 'completed_sft.txt').read_bytes(), b'existing results')
            archives = list((root / 'config_migrations').glob('rl_batch_*'))
            self.assertEqual(len(archives), 1)
            self.assertEqual((archives[0] / '0.before.json').read_bytes(), old_state)
            migrate_batch(repo, root, 32)
            self.assertEqual(len(list((root / 'config_migrations').glob('rl_batch_*'))), 1)

    def test_invalid_size_or_checkpoint_batch_refused(self):
        for micro in (8, 48, 512):
            with tempfile.TemporaryDirectory() as tmp:
                repo, root, checkpoint = self.fixture(tmp)
                with self.assertRaisesRegex(ValueError, 'multiple of 16'):
                    migrate_batch(repo, root, micro)
        with tempfile.TemporaryDirectory() as tmp:
            repo, root, checkpoint = self.fixture(tmp)
            (checkpoint / 'trainer_state.json').write_text(json.dumps({'train_batch_size': 64}))
            with self.assertRaisesRegex(ValueError, 'Unexpected saved'):
                migrate_batch(repo, root, 32)
            self.assertFalse((root / 'config_migrations').exists())

    def test_interrupted_commit_can_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, root, checkpoint = self.fixture(tmp)
            def interrupt(path, content):
                if path.name == 'run_config.json':
                    raise OSError('simulated interruption')
                atomic_write(path, content)
            with patch('scripts.migrate_rl_batch.atomic_write', side_effect=interrupt):
                with self.assertRaises(OSError):
                    migrate_batch(repo, root, 32)
            migrate_batch(repo, root, 32)
            self.assertEqual(json.loads((root / 'run_config.json').read_text())['rl_accumulation'], 8)
            self.assertEqual(json.loads((checkpoint / 'trainer_state.json').read_text())['train_batch_size'], 32)

    def test_missing_checkpoint_root_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, root, checkpoint = self.fixture(tmp)
            path = root / 'run_config.json'
            config = json.loads(path.read_text())
            config['checkpoint_root'] += '-missing'
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'training server'):
                migrate_batch(repo, root, 32)


if __name__ == '__main__':
    unittest.main()
