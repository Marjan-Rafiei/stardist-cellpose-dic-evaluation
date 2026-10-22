import tempfile
import unittest
from pathlib import Path

from code.generate_audit_manifests import model_comparison_rows


class ModelComparisonManifestTests(unittest.TestCase):
    def test_model_pair_rows_use_supplied_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            active = root / "models" / "mitosis_model"
            backup = root / "models" / "mitosis_model_baseline_backup"
            active.mkdir(parents=True)
            backup.mkdir(parents=True)
            (active / "weights.h5").write_bytes(b"same")
            (backup / "weights.h5").write_bytes(b"same")
            (active / "thresholds.json").write_bytes(b"active only")

            rows = model_comparison_rows(root)

            self.assertEqual([row["relative_path"] for row in rows], ["thresholds.json", "weights.h5"])
            self.assertEqual(rows[0]["backup_sha256"], "MISSING")
            self.assertEqual(rows[0]["byte_identical"], "false")
            self.assertEqual(rows[1]["byte_identical"], "true")

    def test_missing_model_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                model_comparison_rows(Path(temporary))


if __name__ == "__main__":
    unittest.main()
