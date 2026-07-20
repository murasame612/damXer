from __future__ import annotations

import unittest

import torch

from examples.generate_synthetic_data import generate
from scripts.build_engineered_inputs import build_engineered_env, numeric_fill
from scripts.paper_protocol import load_config
from scripts.train_damxer import build_token_specs, masked_point_loss


class DamXerCoreTests(unittest.TestCase):
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
            "per_epoch_test",
        )


if __name__ == "__main__":
    unittest.main()
