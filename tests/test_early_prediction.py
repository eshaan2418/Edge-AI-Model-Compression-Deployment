from __future__ import annotations

import numpy as np
import pytest

from edge_ai_compression.analysis.early_prediction import (
    ablation,
    build_dataset,
    feature_columns,
    load_tables,
    plot_vs_fraction,
    prediction_vs_fraction,
)
from tests.synthetic_db import write_early_prediction_db


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    return write_early_prediction_db(tmp_path_factory.mktemp("ep"))


def test_dataset_rows_and_features(db):
    df = build_dataset(*load_tables(db), "w4a8", 0.1)
    assert len(df) == 24 and df["config"].nunique() == 8
    assert (df["signal_step"] == 100).all()  # nearest logged step to 0.1 * 1000
    cols = feature_columns(df)
    assert "kurtosis_max" in cols and "variant_kurtosis" in cols and "target" not in cols


def test_recovers_planted_signal_and_beats_baselines(db):
    table = prediction_vs_fraction(
        db,
        "w4a8",
        predictors=("mean", "params_only", "gbm", "rf"),
        fractions=(0.1, 1.0),
        n_boot=200,
    )
    gbm = table[(table.predictor == "gbm") & (table.fraction == 1.0)].iloc[0]
    assert gbm["spearman"] > 0.7 and gbm["spearman_lo"] > 0.3
    mean = table[(table.predictor == "mean") & (table.fraction == 1.0)].iloc[0]
    assert mean["r2"] <= 0.0 < gbm["r2"]
    assert set(table.columns) >= {"spearman_lo", "spearman_hi", "r2_lo", "mae_hi", "n_configs"}


def test_ablation_assigns_information_to_weight_statistics(db):
    df = build_dataset(*load_tables(db), "w4a8", 1.0)
    ab = ablation(df, "gbm", n_boot=100).set_index(["group", "setting"])
    assert ab.loc[("weights", "only"), "spearman"] > ab.loc[("sharpness", "only"), "spearman"]
    assert ab.loc[("weights", "without"), "spearman"] < ab.loc[("weights", "only"), "spearman"]


def test_plot_and_group_count_guard(db, tmp_path):
    table = prediction_vs_fraction(db, "w4a8", predictors=("gbm",), fractions=(0.3,), n_boot=50)
    assert plot_vs_fraction(table, tmp_path / "ep.png").is_file()
    import pandas as pd

    runs = pd.read_csv(db / "training_runs.csv")
    runs = runs[runs["model"] == "model0"]
    runs.to_csv(tmp_path / "training_runs.csv", index=False)
    for name in ("training_signals.csv", "experiments.csv"):
        (tmp_path / name).write_text((db / name).read_text())
    with pytest.raises(ValueError, match=">= 3 configurations"):
        prediction_vs_fraction(tmp_path, "w4a8", predictors=("gbm",), fractions=(0.3,))
    assert np.isfinite(table["mae"]).all()
