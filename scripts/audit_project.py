"""Verifica dados/artefatos e opcionalmente reproduz o backtest sem alterar o MLflow."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from energy_forecast.evaluation.backtesting import run_walk_forward_backtest
from energy_forecast.evaluation.diagnostics import error_diagnostics, monthly_errors
from energy_forecast.evaluation.metrics import calculate_mae, calculate_mape
from energy_forecast.features.engineering import load_gold_series
from energy_forecast.utils.config import load_settings


def main() -> int:
    """Audita os artefatos existentes e salva evidencias em artifacts/review/."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrain", action="store_true", help="Reexecuta as 52 origens.")
    args = parser.parse_args()
    settings = load_settings()
    gold = load_gold_series(settings)
    summary_path = settings.paths.artifacts_dir / "backtest_summary.json"
    predictions_path = settings.paths.predictions_dir / "backtest_predictions.parquet"
    stored = json.loads(summary_path.read_text(encoding="utf-8"))
    predictions = pd.read_parquet(predictions_path)
    latest = pd.read_parquet(settings.paths.predictions_dir / "latest_forecast_7d.parquet")
    checks: dict[str, bool] = {}
    diagnostics = error_diagnostics(predictions)
    for key in ("mae_lgbm", "mape_lgbm", "mae_baseline", "mape_baseline"):
        checks[f"summary_{key}"] = bool(np.isclose(diagnostics[key], stored["summary"][key]))
    actual = gold.set_index("data_referencia")["carga_mwmed"]
    checks["actuals_match_gold"] = bool(
        np.allclose(predictions["y_true"], pd.to_datetime(predictions["target_date"]).map(actual))
    )
    checks["baseline_matches_lag7"] = bool(
        np.allclose(
            predictions["y_pred_baseline"],
            (pd.to_datetime(predictions["target_date"]) - pd.Timedelta(days=7)).map(actual),
        )
    )
    checks["unique_origin_horizon"] = not predictions.duplicated(["origin_date", "horizon"]).any()
    checks["causal_target_dates"] = bool(
        (pd.to_datetime(predictions["target_date"]) - pd.to_datetime(predictions["origin_date"]))
        .dt.days.eq(predictions["horizon"])
        .all()
    )
    checks["complete_folds"] = bool(
        predictions.groupby("origin_date")["horizon"]
        .apply(lambda h: sorted(h) == list(range(1, 8)))
        .all()
    )
    checks["latest_origin_matches_gold"] = bool(
        pd.to_datetime(latest["origin_date"]).eq(gold["data_referencia"].max()).all()
    )
    checks["latest_complete"] = sorted(latest["horizon"]) == list(range(1, 8))
    for row in stored["horizon_metrics"]:
        selected = predictions.loc[predictions["horizon"] == row["horizon"]]
        for model in ("baseline", "lgbm"):
            for metric, calculate in (("mae", calculate_mae), ("mape", calculate_mape)):
                checks[f"h{row['horizon']}_{metric}_{model}"] = bool(
                    np.isclose(
                        calculate(selected["y_true"], selected[f"y_pred_{model}"]),
                        row[f"{metric}_{model}"],
                    )
                )
    out_dir = settings.paths.artifacts_dir / "review"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.retrain:
        result = run_walk_forward_backtest(
            gold,
            min_train_days=settings.backtesting.min_train_days,
            step_days=settings.backtesting.step_days,
            horizon_days=7,
            max_origins=52,
        )
        result.predictions.to_parquet(out_dir / "reproduced_predictions.parquet", index=False)
        keys = ["origin_date", "target_date", "horizon"]
        reproduced = result.predictions.sort_values(keys).reset_index(drop=True)
        previous = predictions.sort_values(keys).reset_index(drop=True)
        checks["reproduced_keys"] = reproduced[keys].equals(previous[keys])
        checks["reproduced_predictions"] = bool(
            np.allclose(reproduced["y_pred_lgbm"], previous["y_pred_lgbm"], rtol=1e-7, atol=1e-5)
        )
    monthly = monthly_errors(predictions)
    monthly.to_csv(out_dir / "monthly_errors.csv", index=False)
    report = {
        "reviewed_at": datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat(),
        "gold_rows": len(gold),
        "gold_last_date": str(gold["data_referencia"].max().date()),
        "diagnostics": diagnostics,
        "checks": {k: bool(v) for k, v in checks.items()},
        "all_checks_pass": bool(all(checks.values())),
        "backtest_retrained": args.retrain,
        "prediction_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        "monthly_errors": monthly.to_dict(orient="records"),
        "limitations": [
            "Sem tolerancia operacional formal definida; MAPE nao e margem de confianca.",
            "Sem holdout intocado ou registro que exclua tuning previo no periodo avaliado.",
            "Validacao usa a versao atual dos dados; revisoes ONS e atraso de publicacao "
            "nao sao simulados por snapshots historicos.",
            "Bootstrap pareado por origem e descritivo; semanas adjacentes podem ser dependentes.",
        ],
    }
    output = out_dir / "project_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(output),
                "checks_pass": report["all_checks_pass"],
                "diagnostics": diagnostics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
