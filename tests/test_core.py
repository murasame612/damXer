from __future__ import annotations

import csv
import statistics
import unittest
from pathlib import Path

import torch

from examples.generate_synthetic_data import generate
from scripts.build_engineered_inputs import build_engineered_env, numeric_fill
from scripts.completion_protocol import (
    apply_pseudo_mask,
    build_fixed_block_mask,
    legacy_index_features,
    make_windows,
)
from scripts.paper_protocol import load_config
from scripts.train_damxer import build_token_specs, masked_point_loss


class DamXerCoreTests(unittest.TestCase):
    def test_completion_windows_cover_final_row(self) -> None:
        windows = make_windows(10448, 360, 36)
        self.assertEqual(windows[0], (0, 360))
        self.assertEqual(windows[-1], (10088, 10448))
        self.assertEqual(len(windows), 282)

    def test_legacy_saits_index_features_are_explicit(self) -> None:
        features = legacy_index_features(
            100,
            {"legacy_index_feature_periods_samples": {"short": 24, "long": 8766}},
        )
        self.assertEqual(features.shape, (100, 7))

    def test_completion_pseudo_mask_is_fixed_and_contiguous(self) -> None:
        import pandas as pd

        missing = pd.DataFrame({"dx_001": [False] * 100, "dx_002": [False] * 100})
        first = build_fixed_block_mask(missing, list(missing.columns), 12, 1, 42)
        second = build_fixed_block_mask(missing, list(missing.columns), 12, 1, 42)
        self.assertEqual(first, second)
        self.assertEqual(sum(len(indices) for indices in first.values()), 24)
        for indices in first.values():
            self.assertEqual(max(indices) - min(indices) + 1, 12)
        applied = apply_pseudo_mask(missing, first)
        self.assertEqual(int(applied.to_numpy().sum()), 24)

    def test_observed_only_loss_ignores_masked_cells(self) -> None:
        pred = torch.tensor([[[1.0], [100.0]]])
        true = torch.zeros_like(pred)
        observed = torch.tensor([[[1.0], [0.0]]])
        loss = masked_point_loss(pred, true, observed, "mse", 0.2)
        self.assertTrue(torch.isclose(loss, torch.tensor(1.0)))

    def test_base_environment_bank_contains_60_tokens(self) -> None:
        columns = []
        for family in ("H", "seep", "temp"):
            for lag in (1, 3, 6, 12, 24, 48, 84):
                columns.extend([f"{family}_mean_lag{lag}", f"{family}_mean_delta_lag{lag}"])
            for window in (12, 84):
                columns.extend(
                    [
                        f"{family}_mean_rollmean{window}",
                        f"{family}_mean_rollstd{window}",
                        f"{family}_mean_slope{window}",
                    ]
                )
        specs = build_token_specs(columns, "lag")
        self.assertEqual(len(columns), 60)
        self.assertEqual(len(specs), 60)

    def test_synthetic_monitoring_tables_are_generic_and_incomplete(self) -> None:
        raw, clean = generate(rows=128, seed=3)
        self.assertEqual(list(raw.columns), list(clean.columns))
        self.assertTrue({"dx_001", "H_001", "seeP_001", "temp_001"}.issubset(raw.columns))
        self.assertTrue(raw.drop(columns=["date"]).isna().any().any())
        self.assertFalse(clean.drop(columns=["date"]).isna().any().any())

    def test_raw_environment_order_is_hydraulic_seepage_temperature(self) -> None:
        _, clean = generate(rows=128, seed=3)
        value_columns = [column for column in clean.columns if column != "date"]
        clean_values = numeric_fill(clean, value_columns)
        _, families = build_engineered_env(clean_values, value_columns, "dx", "base")
        raw_environment = [*families["H"], *families["seep"], *families["temp"]]
        self.assertEqual(
            raw_environment,
            ["H_001", "H_002", "seeP_001", "seeP_002", "temp_001", "temp_002"],
        )

    def test_frozen_baseline_profiles_match_paper_protocol(self) -> None:
        _, config = load_config()
        common = config["tslib"]["common"]
        self.assertEqual(common["batch_size"], 16)
        self.assertEqual(common["learning_rate"], 0.0005)
        self.assertEqual(
            config["tslib"]["profiles"]["dx_only"]["models"],
            ["PatchTST", "iTransformer", "DLinear", "TimesNet", "FEDformer"],
        )
        self.assertEqual(config["tslib"]["profiles"]["raw_env"]["input_channels"], 200)
        self.assertEqual(
            config["source_compatibility"]["evaluation_schedule"],
            "final_only",
        )

    def test_seed_ledger_matches_frozen_strict_metrics(self) -> None:
        _, config = load_config()
        ledger = Path(__file__).resolve().parents[1] / "results" / "paper_seed_metrics.csv"
        with ledger.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        for variant, expected in config["expected_metrics"].items():
            selected = [row for row in rows if row["variant"] == variant]
            self.assertEqual(len(selected), 5)
            for split in ("val", "test"):
                values = [float(row[f"{split}_mse"]) for row in selected]
                self.assertAlmostEqual(
                    statistics.mean(values), expected[f"{split}_mse_mean"], places=10
                )
                self.assertAlmostEqual(
                    statistics.stdev(values), expected[f"{split}_mse_std"], places=10
                )


if __name__ == "__main__":
    unittest.main()
