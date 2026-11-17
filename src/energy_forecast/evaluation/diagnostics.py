"""Diagnosticos auditaveis dos erros fora da amostra, sem ajustar hiperparametros."""

from __future__ import annotations

import numpy as np
import pandas as pd

from energy_forecast.evaluation.metrics import calculate_mae, calculate_mape


def error_diagnostics(predictions: pd.DataFrame) -> dict:
    """Recalcula erros dos vetores originais; percentis nao sao intervalos preditivos."""
    actual = predictions["y_true"]
    model = predictions["y_pred_lgbm"]
    baseline = predictions["y_pred_baseline"]
    mae = calculate_mae(actual, model)
    mape = calculate_mape(actual, model)
    base_mae = calculate_mae(actual, baseline)
    base_mape = calculate_mape(actual, baseline)
    ape = (actual - model).abs() / actual * 100
    absolute = (actual - model).abs()
    folds = (
        pd.DataFrame(
            {
                "origin": predictions["origin_date"],
                "model": absolute,
                "baseline": (actual - baseline).abs(),
            }
        )
        .groupby("origin")
        .mean()
    )
    # Paired bootstrap of whole forecast origins preserves dependence within each week.
    # Adjacent weeks may still be dependent: this is descriptive, not a deployment gate.
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(folds), size=(5000, len(folds)))
    reduction = (folds["baseline"] - folds["model"]).to_numpy()
    interval = np.quantile(reduction[indices].mean(axis=1), [0.025, 0.975])
    return {
        "count": len(predictions),
        "origins": len(folds),
        "mae_lgbm": mae,
        "mape_lgbm": mape,
        "mae_baseline": base_mae,
        "mape_baseline": base_mape,
        "bias_mwmed": float((model - actual).mean()),
        "rmse_mwmed": float(np.sqrt(((model - actual) ** 2).mean())),
        "median_ape_pct": float(ape.median()),
        "p90_ape_pct": float(ape.quantile(0.90)),
        "p95_ape_pct": float(ape.quantile(0.95)),
        "max_ape_pct": float(ape.max()),
        "days_above_5pct": int((ape > 5).sum()),
        "days_above_10pct": int((ape > 10).sum()),
        "winning_origins_mae": int((folds["model"] < folds["baseline"]).sum()),
        "winning_predictions": int((absolute < (actual - baseline).abs()).sum()),
        "mae_reduction_bootstrap_95_mwmed": interval.tolist(),
    }


def monthly_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    """Exibe a variacao mensal, incluindo meses parciais com sua contagem de previsoes."""
    frame = pd.DataFrame(
        {
            "month": pd.to_datetime(predictions["target_date"]).dt.to_period("M").astype(str),
            "mape_lgbm": (predictions["y_true"] - predictions["y_pred_lgbm"]).abs()
            / predictions["y_true"]
            * 100,
            "mape_baseline": (predictions["y_true"] - predictions["y_pred_baseline"]).abs()
            / predictions["y_true"]
            * 100,
        }
    )
    return (
        frame.groupby("month")
        .agg(
            mape_lgbm=("mape_lgbm", "mean"),
            mape_baseline=("mape_baseline", "mean"),
            count=("mape_lgbm", "size"),
        )
        .reset_index()
    )
