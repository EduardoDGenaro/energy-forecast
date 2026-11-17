"""Motor de backtesting temporal com origens de previsao sucessivas (Expanding Window)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from energy_forecast.evaluation.metrics import (
    EvaluationError,
    calculate_improvement_pct,
    calculate_mae,
    calculate_mape,
)
from energy_forecast.forecasting.baseline import SeasonalNaiveBaseline
from energy_forecast.forecasting.lightgbm_model import (
    LightGBMForecaster,
    get_default_lgbm_params,
)
from energy_forecast.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class BacktestSummary:
    """Metricas agregadas consolidadas do backtesting temporal."""

    num_origins: int
    total_evaluated_days: int
    first_origin_date: str
    last_origin_date: str
    first_target_date: str
    last_target_date: str
    mae_baseline: float
    mae_lgbm: float
    mae_improvement_pct: float
    lgbm_beats_baseline_mae: bool
    status_mae: str
    mape_baseline: float
    mape_lgbm: float
    mape_improvement_pct: float
    lgbm_beats_baseline_mape: bool
    status_mape: str

    def to_dict(self) -> dict[str, Any]:
        """Converte o resumo para dicionario serializavel em JSON."""
        return asdict(self)


@dataclass(frozen=True)
class BacktestResult:
    """Resultado completo do backtesting contendo resumo, tabelas e previsoes."""

    summary: BacktestSummary
    horizon_metrics: pd.DataFrame
    fold_metrics: pd.DataFrame
    predictions: pd.DataFrame
    final_forecaster: LightGBMForecaster


def generate_backtest_origins(
    df: pd.DataFrame,
    min_train_days: int = 730,
    step_days: int = 7,
    horizon_days: int = 7,
    max_origins: int | None = 52,
    date_col: str = "data_referencia",
) -> list[pd.Timestamp]:
    """Gera a lista cronologica de datas de origem (t_0) para validacao temporal.

    Garante que para cada origem t_0:
    1. Existem pelo menos min_train_days observacoes historicas <= t_0;
    2. Todas as datas alvo do horizonte [t_0 + 1, ..., t_0 + horizon_days] possuem
       valores reais observados na serie para avaliacao.

    Args:
        df: Serie temporal diaria ordenada.
        min_train_days: Quantidade minima de dias de historico antes da primeira origem.
        step_days: Passo em dias entre origens sucessivas (ex: 7 dias).
        horizon_days: Horizonte de previsao em dias (padrao: 7).
        max_origins: Quantidade maxima de origens mais recentes a avaliar (None para todas).
        date_col: Nome da coluna de data.

    Returns:
        list[pd.Timestamp]: Datas de origem em ordem cronologica crescente.
    """
    if min_train_days < 60:
        raise EvaluationError("min_train_days deve ser >= 60 para acomodar lags e janelas moveis.")
    if step_days <= 0 or horizon_days <= 0:
        raise EvaluationError("step_days e horizon_days devem ser estritamente positivos.")

    dates = pd.Series(pd.to_datetime(df[date_col])).dt.normalize().sort_values().unique()
    if len(dates) < min_train_days + horizon_days:
        raise EvaluationError(
            f"Serie com {len(dates)} dias e insuficiente para min_train_days={min_train_days} "
            f"e horizon_days={horizon_days}."
        )

    date_set = set(dates)
    last_possible_origin = pd.Timestamp(dates[-1]) - pd.Timedelta(days=horizon_days)
    first_possible_origin = pd.Timestamp(dates[min_train_days - 1])

    # Caminha de tras para frente a partir da ultima origem possivel em passos de step_days
    # para garantir cobertura ate o final mais recente da serie, e depois ordena crescente.
    candidate_origins: list[pd.Timestamp] = []
    current = last_possible_origin
    while current >= first_possible_origin:
        horizon_targets = [current + pd.Timedelta(days=h) for h in range(1, horizon_days + 1)]
        if current in date_set and all(t in date_set for t in horizon_targets):
            candidate_origins.append(current)
        current -= pd.Timedelta(days=step_days)

    candidate_origins.sort()

    if max_origins is not None and max_origins > 0 and len(candidate_origins) > max_origins:
        candidate_origins = candidate_origins[-max_origins:]

    if not candidate_origins:
        raise EvaluationError("Nenhuma data de origem valida encontrada para o backtesting.")

    return candidate_origins


def run_walk_forward_backtest(
    df: pd.DataFrame,
    min_train_days: int = 730,
    step_days: int = 7,
    horizon_days: int = 7,
    lags: tuple[int, ...] = (7, 14, 28),
    rolling_windows: tuple[int, ...] = (7, 14, 28),
    country_code: str = "BR",
    max_origins: int | None = 52,
    lgbm_params: dict[str, Any] | None = None,
    date_col: str = "data_referencia",
    target_col: str = "carga_mwmed",
    calendar_profile: str = "national",
    evaluation_origins: tuple[pd.Timestamp, ...] | None = None,
    weather_snapshots: pd.DataFrame | None = None,
    training_origins: tuple[pd.Timestamp, ...] | None = None,
) -> BacktestResult:
    """Executa o backtesting temporal por janelas expansivas comparando Baseline e LightGBM.

    Em cada origem de previsao t_0:
    1. Recorta estritamente os dados historicos disponiveis ate t_0 (<= t_0);
    2. Ajusta um modelo LightGBM do zero somente com os dados <= t_0;
    3. Gera a previsao de 7 dias do LightGBM e do Baseline Sazonal para [t_0 + 1 .. t_0 + 7];
    4. Compara ambas as previsoes com os valores reais do periodo de teste (MAE e MAPE).

    Args:
        df: Serie diaria da camada Gold.
        min_train_days: Minimo de dias de treino antes da primeira origem.
        step_days: Passo em dias entre origens sucessivas.
        horizon_days: Horizonte de previsao (7 dias).
        lags: Lags sazonais (7, 14, 28).
        rolling_windows: Janelas moveis na origem (7, 14, 28).
        country_code: Codigo ISO de feriados ('BR').
        max_origins: Numero de origens sucessivas avaliadas (padrao: 52 origens semanais = 1 ano).
        lgbm_params: Hiperparametros opcionais do LightGBM.
        date_col: Coluna de data.
        target_col: Coluna de carga.

    Returns:
        BacktestResult: Objeto contendo resumo agregado, metricas por horizonte, folds e previsoes.
    """
    clean_df = df[[date_col, target_col]].copy()
    clean_df[date_col] = pd.to_datetime(clean_df[date_col]).dt.normalize()
    clean_df[target_col] = clean_df[target_col].astype(float)
    clean_df = clean_df.sort_values(date_col, ascending=True).reset_index(drop=True)

    origins = generate_backtest_origins(
        clean_df,
        min_train_days=min_train_days,
        step_days=step_days,
        horizon_days=horizon_days,
        max_origins=max_origins,
        date_col=date_col,
    )

    if evaluation_origins is not None:
        origins = [o for o in origins if o in set(pd.to_datetime(evaluation_origins))]
        if not origins:
            raise EvaluationError("Nenhuma origem de avaliacao disponivel.")
    resolved_params = lgbm_params if lgbm_params is not None else get_default_lgbm_params()
    baseline_model = SeasonalNaiveBaseline(
        horizon_days=horizon_days,
        seasonal_period_days=7,
        date_col=date_col,
        target_col=target_col,
    )
    actual_lookup = clean_df.set_index(date_col)[target_col]

    all_predictions: list[pd.DataFrame] = []
    fold_records: list[dict[str, Any]] = []
    last_forecaster: LightGBMForecaster | None = None

    logger.info(
        "Iniciando backtesting temporal: %d origens sucessivas (%s a %s), horizonte=%d dias.",
        len(origins),
        origins[0].strftime("%Y-%m-%d"),
        origins[-1].strftime("%Y-%m-%d"),
        horizon_days,
    )

    for fold_idx, origin_ts in enumerate(origins, start=1):
        # Recorte causal estrito: somente observacoes <= origin_ts sao repassadas aos modelos
        train_slice = clean_df.loc[clean_df[date_col] <= origin_ts].copy()

        forecaster = LightGBMForecaster(
            horizon_days=horizon_days,
            lags=lags,
            rolling_windows=rolling_windows,
            country_code=country_code,
            date_col=date_col,
            target_col=target_col,
            model_params=resolved_params,
            calendar_profile=calendar_profile,
            weather_snapshots=weather_snapshots,
            training_origins=training_origins,
        )
        forecaster.fit(train_slice, cutoff_date=origin_ts)
        lgbm_preds = forecaster.predict_from_origin(train_slice, origin_date=origin_ts)
        base_preds = baseline_model.predict_from_origin(train_slice, origin_date=origin_ts)

        merged = pd.merge(
            base_preds[["origin_date", "target_date", "horizon", "y_pred_baseline"]],
            lgbm_preds[["origin_date", "target_date", "horizon", "y_pred_lgbm"]],
            on=["origin_date", "target_date", "horizon"],
            how="inner",
        )
        merged["fold_id"] = fold_idx
        merged["y_true"] = merged["target_date"].map(actual_lookup).astype(float)
        merged["abs_error_baseline"] = (merged["y_true"] - merged["y_pred_baseline"]).abs()
        merged["abs_error_lgbm"] = (merged["y_true"] - merged["y_pred_lgbm"]).abs()
        merged["ape_baseline_pct"] = (merged["abs_error_baseline"] / merged["y_true"]) * 100.0
        merged["ape_lgbm_pct"] = (merged["abs_error_lgbm"] / merged["y_true"]) * 100.0

        fold_mae_base = calculate_mae(merged["y_true"], merged["y_pred_baseline"])
        fold_mae_lgbm = calculate_mae(merged["y_true"], merged["y_pred_lgbm"])
        fold_mape_base = calculate_mape(merged["y_true"], merged["y_pred_baseline"])
        fold_mape_lgbm = calculate_mape(merged["y_true"], merged["y_pred_lgbm"])

        fold_records.append(
            {
                "fold_id": fold_idx,
                "origin_date": origin_ts.strftime("%Y-%m-%d"),
                "train_days": len(train_slice),
                "train_supervised_rows": forecaster.train_rows_,
                "mae_baseline": fold_mae_base,
                "mae_lgbm": fold_mae_lgbm,
                "mae_improvement_pct": calculate_improvement_pct(fold_mae_base, fold_mae_lgbm),
                "mape_baseline": fold_mape_base,
                "mape_lgbm": fold_mape_lgbm,
                "mape_improvement_pct": calculate_improvement_pct(fold_mape_base, fold_mape_lgbm),
            }
        )

        all_predictions.append(merged)
        last_forecaster = forecaster

    predictions_df = pd.concat(all_predictions, axis=0, ignore_index=True)
    fold_metrics_df = pd.DataFrame(fold_records)

    horizon_records: list[dict[str, Any]] = []
    for h in range(1, horizon_days + 1):
        h_slice = predictions_df.loc[predictions_df["horizon"] == h]
        h_mae_base = calculate_mae(h_slice["y_true"], h_slice["y_pred_baseline"])
        h_mae_lgbm = calculate_mae(h_slice["y_true"], h_slice["y_pred_lgbm"])
        h_mae_imp = calculate_improvement_pct(h_mae_base, h_mae_lgbm)

        h_mape_base = calculate_mape(h_slice["y_true"], h_slice["y_pred_baseline"])
        h_mape_lgbm = calculate_mape(h_slice["y_true"], h_slice["y_pred_lgbm"])
        h_mape_imp = calculate_improvement_pct(h_mape_base, h_mape_lgbm)

        beats_mae = h_mae_lgbm < h_mae_base
        beats_mape = h_mape_lgbm < h_mape_base

        horizon_records.append(
            {
                "horizon": int(h),
                "count": int(len(h_slice)),
                "mae_baseline": h_mae_base,
                "mae_lgbm": h_mae_lgbm,
                "mae_improvement_pct": h_mae_imp,
                "lgbm_beats_baseline_mae": beats_mae,
                "status_mae": (
                    "LightGBM supera Baseline" if beats_mae else "Desempenho inferior ao Baseline"
                ),
                "mape_baseline": h_mape_base,
                "mape_lgbm": h_mape_lgbm,
                "mape_improvement_pct": h_mape_imp,
                "lgbm_beats_baseline_mape": beats_mape,
                "status_mape": (
                    "LightGBM supera Baseline" if beats_mape else "Desempenho inferior ao Baseline"
                ),
            }
        )

    horizon_metrics_df = pd.DataFrame(horizon_records)

    agg_mae_base = calculate_mae(predictions_df["y_true"], predictions_df["y_pred_baseline"])
    agg_mae_lgbm = calculate_mae(predictions_df["y_true"], predictions_df["y_pred_lgbm"])
    agg_mae_imp = calculate_improvement_pct(agg_mae_base, agg_mae_lgbm)
    agg_beats_mae = agg_mae_lgbm < agg_mae_base

    agg_mape_base = calculate_mape(predictions_df["y_true"], predictions_df["y_pred_baseline"])
    agg_mape_lgbm = calculate_mape(predictions_df["y_true"], predictions_df["y_pred_lgbm"])
    agg_mape_imp = calculate_improvement_pct(agg_mape_base, agg_mape_lgbm)
    agg_beats_mape = agg_mape_lgbm < agg_mape_base

    summary = BacktestSummary(
        num_origins=len(origins),
        total_evaluated_days=len(predictions_df),
        first_origin_date=origins[0].strftime("%Y-%m-%d"),
        last_origin_date=origins[-1].strftime("%Y-%m-%d"),
        first_target_date=predictions_df["target_date"].min().strftime("%Y-%m-%d"),
        last_target_date=predictions_df["target_date"].max().strftime("%Y-%m-%d"),
        mae_baseline=agg_mae_base,
        mae_lgbm=agg_mae_lgbm,
        mae_improvement_pct=agg_mae_imp,
        lgbm_beats_baseline_mae=agg_beats_mae,
        status_mae=(
            "LightGBM supera Baseline" if agg_beats_mae else "Desempenho inferior ao Baseline"
        ),
        mape_baseline=agg_mape_base,
        mape_lgbm=agg_mape_lgbm,
        mape_improvement_pct=agg_mape_imp,
        lgbm_beats_baseline_mape=agg_beats_mape,
        status_mape=(
            "LightGBM supera Baseline" if agg_beats_mape else "Desempenho inferior ao Baseline"
        ),
    )

    assert last_forecaster is not None
    logger.info(
        "Backtesting concluido: MAE Baseline=%.2f MWmed vs LightGBM=%.2f MWmed (melhoria=%.2f%%) | "
        "MAPE Baseline=%.2f%% vs LightGBM=%.2f%% (melhoria=%.2f%%)",
        summary.mae_baseline,
        summary.mae_lgbm,
        summary.mae_improvement_pct,
        summary.mape_baseline,
        summary.mape_lgbm,
        summary.mape_improvement_pct,
    )

    return BacktestResult(
        summary=summary,
        horizon_metrics=horizon_metrics_df,
        fold_metrics=fold_metrics_df,
        predictions=predictions_df,
        final_forecaster=last_forecaster,
    )
