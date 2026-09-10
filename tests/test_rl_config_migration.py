"""Ensure the one-off metadata fix cannot silently accept other source changes."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.migrate_rl_config_fix import AFTER, BEFORE, FIX, TRAINER, digest, migrate


class MigrationTests(unittest.TestCase):
    def fixture(self, tmp):
        repo = Path(tmp) / 'repo'
        root = repo / 'results/run'
        sources = {
            'reproduction/run.py': b'# runner\n',
            'data.py': b'# data\n',
            TRAINER: b'# trainer\n' + BEFORE,
            'requirements-reproduction.txt': b'# dependencies\n',
            'config/zero2_opt.yaml': b'zero_stage: 2\n',
            'scripts/reproduce.sh': b'# launcher\n',
        }
        for name, content in sources.items():
            for path in (repo / name, root / 'source' / name):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
        original = json.dumps({name: digest(content) for name, content in sources.items()}).encode()
        (root / 'source_sha256.json').write_bytes(original)
        (root / 'completed_sft.txt').write_bytes(b'existing results')
        (repo / TRAINER).write_bytes(sources[TRAINER].replace(BEFORE, AFTER))
        return repo, root, original

    def test_migration_preserves_original_and_is_repeatable(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, root, original = self.fixture(tmp)
            migrate(repo, root, dry_run=True)
            self.assertEqual((root / 'source_sha256.json').read_bytes(), original)
            self.assertFalse((root / 'source_migrations').exists())
            migrate(repo, root)
            archive = root / 'source_migrations' / FIX
            self.assertEqual((archive / 'source_sha256.before.json').read_bytes(), original)
            self.assertEqual((root / 'completed_sft.txt').read_bytes(), b'existing results')
            for name, expected in json.loads((root / 'source_sha256.json').read_text()).items():
                self.assertEqual(digest((root / 'source' / name).read_bytes()), expected)
                self.assertEqual(digest((repo / name).read_bytes()), expected)
            record = (archive / 'record.json').read_bytes()
            migrate(repo, root)
            self.assertEqual((archive / 'record.json').read_bytes(), record)

    def test_unrelated_edits_and_tampering_are_rejected(self):
        for kind in ('other_source', 'extra_trainer_edit', 'snapshot', 'added_source'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                repo, root, original = self.fixture(tmp)
                path = {
                    'other_source': repo / 'data.py',
                    'extra_trainer_edit': repo / TRAINER,
                    'snapshot': root / 'source' / TRAINER,
                    'added_source': repo / 'reproduction/new.py',
                }[kind]
                with path.open('ab') as stream:
                    stream.write(b'# unexpected edit\n')
                with self.assertRaises(ValueError):
                    migrate(repo, root)
                self.assertEqual((root / 'source_sha256.json').read_bytes(), original)
                self.assertFalse((root / 'source_migrations').exists())

    def test_retry_after_interrupted_manifest_commit(self):
        from scripts.migrate_rl_config_fix import atomic_write
        with tempfile.TemporaryDirectory() as tmp:
            repo, root, original = self.fixture(tmp)
            def fail_manifest(path, content):
                if path.name == 'source_sha256.json':
                    raise OSError('simulated interruption')
                atomic_write(path, content)
            with patch('scripts.migrate_rl_config_fix.atomic_write', side_effect=fail_manifest):
                with self.assertRaises(OSError):
                    migrate(repo, root)
            self.assertEqual((root / 'source_sha256.json').read_bytes(), original)
            migrate(repo, root)
            self.assertEqual((root / 'source' / TRAINER).read_bytes(), (repo / TRAINER).read_bytes())
            self.assertNotEqual((root / 'source_sha256.json').read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
