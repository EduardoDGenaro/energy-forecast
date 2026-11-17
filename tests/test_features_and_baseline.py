"""Testes unitarios e de integracao para o Baseline Sazonal e Engenharia de Atributos."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy_forecast.features.engineering import (
    FEATURE_COLUMNS,
    FeatureEngineeringError,
    build_calendar_features,
    build_inference_features_for_origin,
    build_supervised_horizon_dataset,
    load_gold_series,
)
from energy_forecast.forecasting.baseline import (
    BaselineForecastError,
    SeasonalNaiveBaseline,
)
from energy_forecast.utils.config import load_settings


def _make_synthetic_series(num_days: int = 90) -> pd.DataFrame:
    """Gera uma serie sintetica deterministica para validacao matematica exata."""
    dates = pd.date_range("2024-01-01", periods=num_days, freq="D")
    values = [30000.0 + float(i * 10) for i in range(num_days)]
    return pd.DataFrame({"data_referencia": dates, "carga_mwmed": values})


def test_calendar_and_brazil_holiday_features() -> None:
    """Verifica codificacao de calendario, fim de semana e feriados nacionais brasileiros."""
    dates = pd.Series(
        pd.to_datetime(
            [
                "2024-12-24",  # Terca-feira, vespera de Natal
                "2024-12-25",  # Quarta-feira, Natal (feriado BR)
                "2024-12-26",  # Quinta-feira, dia seguinte ao Natal
                "2024-12-28",  # Sabado (fim de semana)
                "2025-01-01",  # Quarta-feira, Confraternizacao Universal (e lag_7 = Natal!)
            ]
        )
    )
    cal = build_calendar_features(dates, country_code="BR")

    # 2024-12-24: vespera de feriado
    assert cal.loc[0, "is_day_before_holiday_br"] == 1
    assert cal.loc[0, "is_holiday_br"] == 0
    assert cal.loc[0, "is_weekend"] == 0

    # 2024-12-25: feriado nacional
    assert cal.loc[1, "is_holiday_br"] == 1
    assert cal.loc[1, "month"] == 12
    assert cal.loc[1, "day_of_month"] == 25

    # 2024-12-26: dia apos feriado
    assert cal.loc[2, "is_day_after_holiday_br"] == 1

    # 2024-12-28: sabado
    assert cal.loc[3, "day_of_week"] == 5
    assert cal.loc[3, "is_weekend"] == 1

    # 2025-01-01: feriado nacional E 7 dias apos o Natal (is_holiday_lag_7 == 1)
    assert cal.loc[4, "is_holiday_br"] == 1
    assert cal.loc[4, "is_holiday_lag_7"] == 1


def test_seasonal_naive_baseline_exact_values_and_no_leakage() -> None:
    """Comprova que o Baseline usa exatamente y(tau - 7) e ignora observacoes > origin_date."""
    df = _make_synthetic_series(60)
    origin_date = pd.Timestamp("2024-01-31")
    baseline = SeasonalNaiveBaseline(horizon_days=7, seasonal_period_days=7)

    pred_clean = baseline.predict_from_origin(df, origin_date=origin_date)
    assert len(pred_clean) == 7
    assert list(pred_clean["horizon"]) == [1, 2, 3, 4, 5, 6, 7]

    # Para h=1 (2024-02-01), a referencia deve ser 2024-01-25
    # Para h=7 (2024-02-07), a referencia deve ser 2024-01-31 (a propria origem)
    lookup = df.set_index("data_referencia")["carga_mwmed"]
    for _, row in pred_clean.iterrows():
        expected_ref = row["target_date"] - pd.Timedelta(days=7)
        assert row["reference_date"] == expected_ref
        assert row["reference_date"] <= origin_date
        assert row["y_pred_baseline"] == pytest.approx(lookup.loc[expected_ref])

    # Corrompe todos os valores futuros (> origin_date) e verifica que a previsao nao muda
    df_corrupted = df.copy()
    df_corrupted.loc[df_corrupted["data_referencia"] > origin_date, "carga_mwmed"] = -999999.0
    pred_corrupted = baseline.predict_from_origin(df_corrupted, origin_date=origin_date)

    pd.testing.assert_frame_equal(pred_clean, pred_corrupted)


def test_baseline_rejects_invalid_period() -> None:
    """Garante que periodo sazonal menor que o horizonte seja rejeitado."""
    with pytest.raises(BaselineForecastError, match="periodo sazonal"):
        SeasonalNaiveBaseline(horizon_days=7, seasonal_period_days=3)


def test_feature_engineering_rejects_leaky_lags() -> None:
    """Impede configuracao de lags menores que o horizonte de 7 dias."""
    df = _make_synthetic_series(60)
    with pytest.raises(FeatureEngineeringError, match="devem ser >="):
        build_supervised_horizon_dataset(df, horizon_days=7, lags=(1, 7, 14))


def test_features_reject_gaps_in_daily_windows() -> None:
    """Uma janela de sete observacoes nao pode representar oito dias por falta de dado."""
    df = _make_synthetic_series(90).drop(index=40)
    with pytest.raises(FeatureEngineeringError, match="continuidade diaria"):
        build_supervised_horizon_dataset(df)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 0.0, -1.0])
def test_features_reject_invalid_loads(value: float) -> None:
    df = _make_synthetic_series(90)
    df.loc[40, "carga_mwmed"] = value
    with pytest.raises(FeatureEngineeringError, match="finitos e positivos"):
        build_supervised_horizon_dataset(df)


def test_inference_features_exact_lags_rolling_and_zero_future_leakage() -> None:
    """Valida calculo exato de lags e medias moveis na origem e ausencia de vazamento futuro."""
    df = _make_synthetic_series(90)
    origin_date = pd.Timestamp("2024-02-15")

    feats_clean = build_inference_features_for_origin(df, origin_date=origin_date)
    assert len(feats_clean) == 7
    assert set(FEATURE_COLUMNS).issubset(set(feats_clean.columns))

    lookup = df.set_index("data_referencia")["carga_mwmed"]
    expected_roll_7 = lookup.loc[origin_date - pd.Timedelta(days=6) : origin_date].mean()
    expected_roll_14 = lookup.loc[origin_date - pd.Timedelta(days=13) : origin_date].mean()
    expected_roll_28 = lookup.loc[origin_date - pd.Timedelta(days=27) : origin_date].mean()

    for _, row in feats_clean.iterrows():
        target_ts = row["target_date"]
        assert row["origin_load"] == pytest.approx(lookup.loc[origin_date])
        assert row["rolling_mean_7"] == pytest.approx(expected_roll_7)
        assert row["rolling_mean_14"] == pytest.approx(expected_roll_14)
        assert row["rolling_mean_28"] == pytest.approx(expected_roll_28)
        assert row["lag_7"] == pytest.approx(lookup.loc[target_ts - pd.Timedelta(days=7)])
        assert row["lag_14"] == pytest.approx(lookup.loc[target_ts - pd.Timedelta(days=14)])
        assert row["lag_28"] == pytest.approx(lookup.loc[target_ts - pd.Timedelta(days=28)])

    # Corrompe todos os dados futuros (> origin_date) com valores extremos
    df_corrupted = df.copy()
    df_corrupted.loc[df_corrupted["data_referencia"] > origin_date, "carga_mwmed"] = 1e12
    feats_corrupted = build_inference_features_for_origin(df_corrupted, origin_date=origin_date)

    pd.testing.assert_frame_equal(feats_clean, feats_corrupted)


def test_supervised_dataset_cutoff_strictly_prevents_test_leakage() -> None:
    """Comprova que cutoff_date impede que alvos ou features futuros entrem no treino."""
    df = _make_synthetic_series(90)
    cutoff_date = pd.Timestamp("2024-02-20")

    sup_clean = build_supervised_horizon_dataset(df, cutoff_date=cutoff_date)
    assert not sup_clean.empty
    assert (sup_clean["origin_date"] <= cutoff_date).all()
    assert (sup_clean["target_date"] <= cutoff_date).all()

    df_corrupted = df.copy()
    df_corrupted.loc[df_corrupted["data_referencia"] > cutoff_date, "carga_mwmed"] = np.nan
    sup_corrupted = build_supervised_horizon_dataset(df_corrupted, cutoff_date=cutoff_date)

    pd.testing.assert_frame_equal(sup_clean, sup_corrupted)


def test_gold_series_loads_and_generates_features_and_baseline() -> None:
    """Teste de integracao sobre a serie real da camada Gold do subsistema SE."""
    settings = load_settings()
    gold_df = load_gold_series(settings, source="parquet")
    gold_duck_df = load_gold_series(settings, source="duckdb")

    assert len(gold_df) == 3240
    assert len(gold_duck_df) == 3240

    origin_date = pd.Timestamp("2025-12-31")
    baseline = SeasonalNaiveBaseline(horizon_days=settings.forecasting.horizon_days)
    base_preds = baseline.predict_from_origin(gold_df, origin_date=origin_date)
    infer_feats = build_inference_features_for_origin(gold_df, origin_date=origin_date)

    # A coluna lag_7 das features deve coincidir exatamente com a previsao do Baseline Sazonal
    np.testing.assert_allclose(
        infer_feats["lag_7"].to_numpy(),
        base_preds["y_pred_baseline"].to_numpy(),
    )
