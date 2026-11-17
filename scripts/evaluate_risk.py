"""Atualiza faixas, cobertura, alertas locais e cortes criticos sem alterar o modelo."""

from __future__ import annotations

import json

import pandas as pd

from energy_forecast.evaluation.risk import (
    add_prediction_intervals,
    degradation_monitor,
    interval_coverage,
    load_risk_policy,
    operational_limits,
)
from energy_forecast.features.critical_calendar import critical_period_report
from energy_forecast.features.engineering import load_gold_series
from energy_forecast.forecasting.monitoring import archive_forecast, reconcile_issued_forecasts
from energy_forecast.utils.config import load_settings


def main() -> int:
    settings = load_settings()
    policy = load_risk_policy(settings.paths.root_dir)
    gold = load_gold_series(settings)
    folder = settings.paths.artifacts_dir / "risk"
    folder.mkdir(parents=True, exist_ok=True)
    history = pd.read_parquet(settings.paths.predictions_dir / "backtest_predictions.parquet")
    # Out-of-sample historical coverage, each row calibrated only from earlier known errors.
    scored = add_prediction_intervals(history, history, policy)
    scored.to_parquet(folder / "prequential_intervals.parquet", index=False)
    interval_coverage(scored).to_csv(folder / "coverage_by_horizon.csv", index=False)
    operational_limits(history, policy).to_csv(folder / "operational_limits.csv", index=False)
    critical_period_report(history).to_csv(folder / "critical_periods.csv", index=False)
    latest_path = settings.paths.predictions_dir / "latest_forecast_7d.parquet"
    if latest_path.exists():
        latest = pd.read_parquet(latest_path)
        latest = latest.drop(
            columns=[
                c
                for c in (
                    "lower_bound",
                    "upper_bound",
                    "calibration_n",
                    "nominal_coverage",
                    "calibration_last_target",
                    "interval_status",
                )
                if c in latest
            ]
        )
        latest = add_prediction_intervals(latest, history, policy)
        latest.to_parquet(latest_path, index=False)
        archive = archive_forecast(latest, settings.paths.predictions_dir)
        print(f"Emissao preservada: {archive}")
    live = reconcile_issued_forecasts(settings.paths.predictions_dir, gold)
    if not live.empty:
        live.to_parquet(folder / "operational_scored.parquet", index=False)
    payload = {
        "historical_simulation": degradation_monitor(scored, policy),
        "operational": degradation_monitor(live, policy),
        "policy_status": policy.status,
    }
    (folder / "monitoring.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
