"""Regressoes para diagnosticos e para estados incoerentes no painel."""

import numpy as np
import pandas as pd
import pytest

from energy_forecast.evaluation.diagnostics import error_diagnostics, monthly_errors


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "origin_date": pd.to_datetime(["2026-01-30", "2026-01-30"]),
            "target_date": pd.to_datetime(["2026-01-31", "2026-02-01"]),
            "horizon": [1, 2],
            "y_true": [100.0, 200.0],
            "y_pred_lgbm": [105.0, 180.0],
            "y_pred_baseline": [110.0, 240.0],
        }
    )


def test_diagnostics_recompute_errors_instead_of_trusting_saved_columns() -> None:
    predictions = _predictions().assign(ape_lgbm_pct=0, abs_error_lgbm=0)
    result = error_diagnostics(predictions)
    assert result["mape_lgbm"] == pytest.approx(7.5)
    assert result["mae_lgbm"] == pytest.approx(12.5)
    assert result["bias_mwmed"] == pytest.approx(-7.5)
    assert result["max_ape_pct"] == pytest.approx(10)
    assert result["days_above_10pct"] == 0
    assert result["days_above_5pct"] == 1
    assert result["winning_origins_mae"] == 1
    assert result["mae_reduction_bootstrap_95_mwmed"] == [12.5, 12.5]


def test_monthly_errors_include_sample_sizes_of_partial_months() -> None:
    monthly = monthly_errors(_predictions())
    assert monthly["count"].tolist() == [1, 1]
    assert monthly["mape_lgbm"].tolist() == [5.0, 10.0]


def test_dashboard_rejects_summary_and_gold_mismatches() -> None:
    from dashboard.app import _prepare_dashboard_data

    predictions = _predictions()
    diagnostics = error_diagnostics(predictions)
    summary = {"summary": diagnostics.copy()}
    gold = pd.DataFrame(
        {"data_referencia": predictions["target_date"], "carga_mwmed": predictions["y_true"]}
    )
    _prepare_dashboard_data(gold, predictions, summary)
    summary["summary"]["mape_lgbm"] = 0
    with pytest.raises(ValueError, match="métricas salvas"):
        _prepare_dashboard_data(gold, predictions, summary)
    summary["summary"]["mape_lgbm"] = diagnostics["mape_lgbm"]
    gold.loc[0, "carga_mwmed"] = np.nan
    with pytest.raises(ValueError, match="versão atual"):
        _prepare_dashboard_data(gold, predictions, summary)
