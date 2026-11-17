"""Compara candidatos predefinidos, congela selecao no desenvolvimento e reserva 13 semanas."""

from __future__ import annotations

import json

import pandas as pd

from energy_forecast.evaluation.backtesting import run_walk_forward_backtest
from energy_forecast.evaluation.metrics import calculate_mae, calculate_mape
from energy_forecast.evaluation.risk import load_risk_policy, weighted_error
from energy_forecast.features.critical_calendar import critical_period_report
from energy_forecast.features.engineering import load_gold_series
from energy_forecast.utils.config import load_settings


def main() -> int:
    settings = load_settings()
    policy = load_risk_policy(settings.paths.root_dir)
    gold = load_gold_series(settings)
    original = pd.read_parquet(settings.paths.predictions_dir / "backtest_predictions.parquet")
    original["origin_date"] = pd.to_datetime(original["origin_date"])
    original["target_date"] = pd.to_datetime(original["target_date"])
    origins = sorted(original["origin_date"].unique())
    if len(origins) != 52:
        raise ValueError("O protocolo exige 52 origens: 39 desenvolvimento / 13 verificacao.")
    development = original.loc[original["origin_date"].isin(origins[:-13])]
    reserved = original.loc[original["origin_date"].isin(origins[-13:])]
    cutoff = development["target_date"].max()
    development_gold = gold.loc[gold["data_referencia"] <= cutoff]
    folder = settings.paths.artifacts_dir / "candidates"
    folder.mkdir(parents=True, exist_ok=True)
    weather_path = settings.paths.root_dir / "data/weather/temperature_snapshots.parquet"
    weather = pd.read_parquet(weather_path) if weather_path.exists() else None
    training_origins = (
        tuple(pd.to_datetime(weather["origin_date"].unique())) if (weather is not None) else None
    )

    def metrics(frame: pd.DataFrame) -> dict:
        return {
            "mae": calculate_mae(frame["y_true"], frame["y_pred_lgbm"]),
            "mape": calculate_mape(frame["y_true"], frame["y_pred_lgbm"]),
            "weighted_error": weighted_error(frame, policy),
            "count": len(frame),
        }

    def backtest(data: pd.DataFrame, count: int, **kwargs) -> pd.DataFrame:
        return run_walk_forward_backtest(
            data,
            min_train_days=settings.backtesting.min_train_days,
            step_days=7,
            max_origins=count,
            **kwargs,
        ).predictions

    candidates = {"national": development}
    candidates["critical"] = backtest(
        development_gold, len(origins) - 13, calendar_profile="critical"
    )
    if weather is not None:
        # Same weekly training rows isolate the incremental contribution of temperature.
        candidates["weather_control"] = backtest(
            development_gold,
            len(origins) - 13,
            training_origins=training_origins,
        )
        candidates["weather"] = backtest(
            development_gold,
            len(origins) - 13,
            training_origins=training_origins,
            weather_snapshots=weather,
        )
    development_scores = {name: metrics(frame) for name, frame in candidates.items()}
    eligible = ["national", "critical"] + (["weather"] if weather is not None else [])
    selected = min(eligible, key=lambda name: development_scores[name]["weighted_error"])
    # Require a meaningful development gain and no aggregate MAE regression.
    if (
        development_scores[selected]["weighted_error"]
        > development_scores["national"]["weighted_error"] * 0.98
        or development_scores[selected]["mae"] > development_scores["national"]["mae"]
    ):
        selected = "national"
    selection = {
        "selected": selected,
        "development_last_target": str(cutoff.date()),
        "reserved_first_origin": str(pd.Timestamp(origins[-13]).date()),
        "development": development_scores,
        "selection_rule": "lowest weighted_error; >=2% gain and no MAE regression",
        "promoted": False,
    }
    selection_path = folder / "frozen_selection.json"
    selection_path.write_text(json.dumps(selection, indent=2), encoding="utf-8")
    # Only after persisting the frozen choice do we evaluate its reserved period.
    weather_test = None
    if weather is not None:
        complete_reserved = tuple(o for o in origins[-13:] if o in training_origins)
        weather_test = backtest(
            gold,
            13,
            training_origins=training_origins,
            weather_snapshots=weather,
            evaluation_origins=complete_reserved,
        )
        weather_control_test = backtest(
            gold,
            13,
            training_origins=training_origins,
            evaluation_origins=complete_reserved,
        )
        selection["weather_reserved"] = {
            "temperature": metrics(weather_test),
            "matched_control": metrics(weather_control_test),
            "available_origins": len(complete_reserved),
            "requested_origins": 13,
            "excluded_origins": [
                str(pd.Timestamp(o).date()) for o in origins[-13:] if o not in complete_reserved
            ],
        }
    if selected == "critical":
        test = backtest(gold, 13, calendar_profile="critical")
    elif selected == "weather":
        test = weather_test
    else:
        test = reserved
    selection["reserved_selected"] = metrics(test)
    selection["reserved_champion"] = metrics(
        reserved.loc[reserved["origin_date"].isin(test["origin_date"])]
    )
    selection["weather_incremental_development"] = (
        {
            "control": development_scores["weather_control"],
            "temperature": development_scores["weather"],
        }
        if weather is not None
        else {"status": "missing_archived_forecasts"}
    )
    selection["limitations"] = [
        "Verificacao retrospectiva; historico ja havia sido consultado no projeto.",
        "Nao e holdout prospectivo intocado; nenhum candidato e promovido automaticamente.",
        "Temperatura e proxy exploratoria: media simples de SP, RJ, BH e Brasilia.",
        "Disponibilidade meteorologica assume atraso conservador de 6h apos inicializacao.",
        "Rodadas antigas podem ser hindcasts/reprocessamentos; requer confirmacao de proveniencia.",
    ]
    for name, frame in candidates.items():
        frame.to_parquet(folder / f"development_{name}.parquet", index=False)
        critical_period_report(frame).to_csv(folder / f"critical_{name}.csv", index=False)
    test.to_parquet(folder / "reserved_selected.parquet", index=False)
    critical_period_report(test).to_csv(folder / "reserved_critical.csv", index=False)
    (folder / "comparison.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps(selection, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
