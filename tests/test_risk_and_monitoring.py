"""Protecoes contra vazamento na calibracao, meteorologia e acompanhamento real."""

from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from energy_forecast.evaluation.metrics import EvaluationError
from energy_forecast.evaluation.risk import (
    RiskPolicy,
    add_prediction_intervals,
    degradation_monitor,
    interval_coverage,
    operational_limits,
    weighted_error,
)
from energy_forecast.features.critical_calendar import critical_calendar
from energy_forecast.features.weather import join_weather
from energy_forecast.forecasting.monitoring import archive_forecast, reconcile_issued_forecasts


def _history() -> pd.DataFrame:
    rows = []
    for i, origin in enumerate(pd.date_range("2024-01-01", periods=6, freq="7D")):
        for h in range(1, 8):
            rows.append(
                {
                    "origin_date": origin,
                    "target_date": origin + pd.Timedelta(days=h),
                    "horizon": h,
                    "y_true": 100.0,
                    "y_pred_lgbm": 100.0 + i * 10,
                    "y_pred_baseline": 95.0,
                }
            )
    return pd.DataFrame(rows)


def test_intervals_ignore_future_actuals_and_own_target() -> None:
    history = _history()
    forecast = history.loc[history.origin_date == pd.Timestamp("2024-01-22")].copy()
    forecast["y_pred_lgbm"] = 100.0
    policy = RiskPolicy(coverage=0.75, calibration_min_samples=3)
    clean = add_prediction_intervals(forecast, history, policy)
    corrupted = history.copy()
    corrupted.loc[corrupted.target_date > pd.Timestamp("2024-01-22"), "y_true"] = 99999.0
    after = add_prediction_intervals(forecast, corrupted, policy)
    pd.testing.assert_frame_equal(clean, after)
    assert clean.lower_bound.tolist() == [80.0] * 7
    assert clean.upper_bound.tolist() == [120.0] * 7
    assert (clean.calibration_last_target <= clean.origin_date).all()
    assert interval_coverage(clean).coverage_pct.eq(100).all()


def test_intervals_do_not_invent_finite_quantile_without_enough_data() -> None:
    history = _history()
    forecast = history.loc[history.origin_date == pd.Timestamp("2024-01-22")]
    result = add_prediction_intervals(
        forecast, history, RiskPolicy(coverage=0.95, calibration_min_samples=3)
    )
    assert result.lower_bound.isna().all()
    assert result.interval_status.eq("insufficient_history").all()


def test_limits_and_asymmetric_loss_are_distinct_from_mean_error() -> None:
    frame = _history().iloc[:2].copy()
    frame["y_pred_lgbm"] = [90.0, 110.0]
    assert weighted_error(frame, RiskPolicy()) == 15.0
    assert weighted_error(frame, replace(RiskPolicy(), underprediction_weight=1.0)) == 10.0
    limits = operational_limits(frame, RiskPolicy())
    assert limits.breach_rate_pct.eq(100).all()
    assert limits.status.eq("attention").all()


def test_monitor_requires_sample_and_flags_baseline_and_error_growth() -> None:
    dates = pd.date_range("2024-01-01", periods=30)
    frame = pd.DataFrame(
        {
            "origin_date": dates - pd.Timedelta(days=1),
            "target_date": dates,
            "horizon": 1,
            "y_true": 100.0,
            "y_pred_lgbm": 101.0,
            "y_pred_baseline": 98.0,
        }
    )
    policy = RiskPolicy(monitor_recent_days=7, monitor_reference_days=21, monitor_min_samples=3)
    frame.loc[frame.index[-7:], "y_pred_lgbm"] = 110.0
    report = degradation_monitor(frame, policy)
    assert {"baseline_better", "error_increase", "daily_limit_breaches"}.issubset(report["alerts"])
    assert degradation_monitor(frame.tail(2), policy)["status"] == "waiting"
    assert degradation_monitor(pd.DataFrame(), policy)["status"] == "waiting"


def test_emitted_forecasts_do_not_backdate_performance(tmp_path) -> None:
    frame = _history().iloc[:7].drop(columns="y_true")
    issued = datetime(2024, 1, 5, 12, tzinfo=ZoneInfo("America/Sao_Paulo"))
    first = archive_forecast(frame, tmp_path, issued)
    same = archive_forecast(frame, tmp_path, issued)
    assert first == same
    assert len(list((tmp_path / "issued_forecasts").glob("*.parquet"))) == 1
    gold = pd.DataFrame(
        {"data_referencia": pd.date_range("2024-01-01", periods=7), "carga_mwmed": 100.0}
    )
    scored = reconcile_issued_forecasts(tmp_path, gold)
    assert scored.target_date.tolist() == [pd.Timestamp("2024-01-06"), pd.Timestamp("2024-01-07")]


def test_critical_calendar_encodes_events_without_changing_national_holidays() -> None:
    result = critical_calendar(
        pd.Series(
            pd.to_datetime(
                ["2026-02-16", "2026-02-17", "2026-02-18", "2026-06-04", "2026-06-05", "2026-12-26"]
            )
        )
    )
    assert result.is_carnival.tolist() == [1, 1, 0, 0, 0, 0]
    assert result.loc[2, "is_ash_wednesday"] == 1
    assert result.loc[3, "is_corpus_christi"] == 1
    assert result.loc[4, "is_bridge_day"] == 1
    assert result.loc[5, "is_year_end"] == 1


def test_weather_uses_forecasts_available_at_origin_only() -> None:
    frame = _history().iloc[:1].copy()
    weather = frame[["origin_date", "target_date"]].copy()
    weather["temperature_mean"] = 25.0
    weather["available_at"] = "2024-01-01T06:00:00Z"
    result = join_weather(frame, weather, required=True)
    assert result.cooling_degrees.iloc[0] == 3.0
    weather["available_at"] = "2024-01-03T00:00:00Z"
    with pytest.raises(ValueError, match="indisponivel"):
        join_weather(frame, weather, required=True)
    weather["available_at"] = "2024-01-01T06:00:00Z"
    weather["temperature_mean"] = np.nan
    with pytest.raises(ValueError, match="invalida"):
        join_weather(frame, weather, required=True)


def test_calendar_candidate_fit_and_inference_remain_causal() -> None:
    from energy_forecast.features.critical_calendar import CRITICAL_COLUMNS
    from energy_forecast.forecasting.lightgbm_model import LightGBMForecaster

    dates = pd.date_range("2024-01-01", periods=120)
    frame = pd.DataFrame({"data_referencia": dates, "carga_mwmed": 40000.0 + np.arange(120) * 5})
    params = {"n_estimators": 10, "n_jobs": 1, "verbose": -1, "random_state": 42}
    model = LightGBMForecaster(calendar_profile="critical", model_params=params)
    origin = dates[100]
    model.fit(frame, cutoff_date=origin)
    clean = model.predict_from_origin(frame, origin)
    frame.loc[frame.data_referencia > origin, "carga_mwmed"] = 999999.0
    pd.testing.assert_frame_equal(clean, model.predict_from_origin(frame, origin))
    assert set(CRITICAL_COLUMNS).issubset(model.get_feature_importance().feature)


def test_weather_rejects_run_later_than_declared_availability() -> None:
    frame = _history().iloc[:1]
    weather = frame[["origin_date", "target_date"]].copy()
    weather["temperature_mean"] = 25.0
    weather["available_at"] = "2024-01-01T06:00:00Z"
    weather["run_time"] = "2024-01-02T00:00:00Z"
    with pytest.raises(ValueError, match="posterior"):
        join_weather(frame, weather, required=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"coverage": 1.0},
        {"calibration_min_samples": 0},
        {"underprediction_weight": -1.0},
        {"daily_error_limit_pct": {1: 5.0}},
        {"degradation_ratio": 1.0},
    ],
)
def test_invalid_policies_are_rejected(kwargs) -> None:
    with pytest.raises(EvaluationError):
        RiskPolicy(**kwargs)
