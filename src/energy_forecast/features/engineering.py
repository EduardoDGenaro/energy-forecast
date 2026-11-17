"""Engenharia de atributos temporais, calendario e janelas moveis sem vazamento temporal."""

from __future__ import annotations

import duckdb
import holidays
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from energy_forecast.features.critical_calendar import CRITICAL_COLUMNS, critical_calendar
from energy_forecast.utils.config import AppSettings

FEATURE_COLUMNS: tuple[str, ...] = (
    "horizon",
    "day_of_week",
    "day_of_month",
    "day_of_year",
    "month",
    "week_of_year",
    "is_weekend",
    "is_holiday_br",
    "is_day_before_holiday_br",
    "is_day_after_holiday_br",
    "is_holiday_lag_7",
    "day_of_year_sin",
    "day_of_year_cos",
    "lag_7",
    "lag_14",
    "lag_28",
    "lag_7_diff_14",
    "lag_7_ratio_14",
    "origin_load",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "rolling_std_14",
    "rolling_std_28",
    "lag_7_vs_rolling_mean_7",
)


class FeatureEngineeringError(ValueError):
    """Excecao lancada quando os dados de entrada sao insuficientes ou invalidos para features."""


def load_gold_series(
    settings: AppSettings,
    source: str = "parquet",
) -> pd.DataFrame:
    """Carrega a serie diaria certificada do subsistema SE a partir da camada Gold.

    Args:
        settings: Configuracoes globais do projeto.
        source: Fonte de leitura ('parquet' para data/gold/fct_carga_diaria_se.parquet
            ou 'duckdb' para a tabela main_gold.fct_carga_diaria_se).

    Returns:
        pd.DataFrame: DataFrame ordenado por data_referencia contendo no minimo
            as colunas ['data_referencia', 'id_subsistema', 'carga_mwmed'].

    Raises:
        FeatureEngineeringError: Se a camada Gold nao existir ou estiver vazia.
    """
    if source == "parquet":
        gold_file = settings.paths.gold_dir / "fct_carga_diaria_se.parquet"
        if not gold_file.is_file():
            raise FeatureEngineeringError(
                f"Arquivo Parquet da camada Gold nao encontrado em {gold_file}. "
                "Execute a etapa dbt antes de carregar a serie."
            )
        df = pq.read_table(gold_file).to_pandas()
    elif source == "duckdb":
        if not settings.paths.duckdb_path.is_file():
            raise FeatureEngineeringError(
                f"Banco DuckDB nao encontrado em {settings.paths.duckdb_path}."
            )
        with duckdb.connect(str(settings.paths.duckdb_path), read_only=True) as con:
            df = (
                con.sql("SELECT * FROM main_gold.fct_carga_diaria_se ORDER BY data_referencia ASC")
                .to_arrow_table()
                .to_pandas()
            )
    else:
        raise FeatureEngineeringError(f"Fonte de leitura Gold desconhecida: {source}")

    if df.empty:
        raise FeatureEngineeringError("A serie carregada da camada Gold esta vazia.")

    df = df.copy()
    df["data_referencia"] = pd.to_datetime(df["data_referencia"]).dt.normalize()
    df["carga_mwmed"] = df["carga_mwmed"].astype(float)
    df = df.sort_values("data_referencia", ascending=True).reset_index(drop=True)
    _prepare_clean_series(df)
    return df


def _build_holiday_date_set(
    min_year: int,
    max_year: int,
    country_code: str = "BR",
) -> set[pd.Timestamp]:
    """Retorna o conjunto de datas de feriados nacionais para o intervalo de anos informado."""
    years = list(range(min_year - 1, max_year + 2))
    cal = holidays.country_holidays(country_code, years=years)
    return {pd.Timestamp(dt).normalize() for dt in cal}


def build_calendar_features(
    dates: pd.Series | pd.DatetimeIndex,
    country_code: str = "BR",
) -> pd.DataFrame:
    """Constroi atributos deterministicos de calendario e feriados para um vetor de datas.

    Atributos de calendario dependem apenas da data alvo e nao introduzem vazamento temporal.

    Args:
        dates: Serie ou indice de datas.
        country_code: Codigo ISO do pais para calendario de feriados (padrao: 'BR').

    Returns:
        pd.DataFrame: Atributos de calendario alinhados posicionalmente a entrada.
    """
    dt_series = pd.Series(pd.to_datetime(dates)).dt.normalize().reset_index(drop=True)
    if dt_series.empty:
        raise FeatureEngineeringError("A serie de datas para atributos de calendario esta vazia.")

    min_year = int(dt_series.dt.year.min())
    max_year = int(dt_series.dt.year.max())
    holiday_dates = _build_holiday_date_set(min_year, max_year, country_code=country_code)

    day_of_week = dt_series.dt.day_of_week.astype(int)
    day_of_month = dt_series.dt.day.astype(int)
    day_of_year = dt_series.dt.day_of_year.astype(int)
    month = dt_series.dt.month.astype(int)
    week_of_year = dt_series.dt.isocalendar().week.astype(int)
    is_weekend = (day_of_week >= 5).astype(int)

    is_holiday_br = dt_series.isin(holiday_dates).astype(int)
    next_day = dt_series + pd.Timedelta(days=1)
    prev_day = dt_series - pd.Timedelta(days=1)
    is_day_before_holiday_br = next_day.isin(holiday_dates).astype(int)
    is_day_after_holiday_br = prev_day.isin(holiday_dates).astype(int)
    lag_7_date = dt_series - pd.Timedelta(days=7)
    is_holiday_lag_7 = lag_7_date.isin(holiday_dates).astype(int)

    angle = 2.0 * np.pi * (day_of_year / 365.25)
    day_of_year_sin = np.sin(angle).astype(float)
    day_of_year_cos = np.cos(angle).astype(float)

    return pd.DataFrame(
        {
            "day_of_week": day_of_week,
            "day_of_month": day_of_month,
            "day_of_year": day_of_year,
            "month": month,
            "week_of_year": week_of_year,
            "is_weekend": is_weekend,
            "is_holiday_br": is_holiday_br,
            "is_day_before_holiday_br": is_day_before_holiday_br,
            "is_day_after_holiday_br": is_day_after_holiday_br,
            "is_holiday_lag_7": is_holiday_lag_7,
            "day_of_year_sin": day_of_year_sin,
            "day_of_year_cos": day_of_year_cos,
        }
    )


def _prepare_clean_series(
    df: pd.DataFrame,
    date_col: str = "data_referencia",
    target_col: str = "carga_mwmed",
    cutoff_date: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Valida, ordena e aplica corte temporal estrito (<= cutoff_date) na serie."""
    if date_col not in df.columns or target_col not in df.columns:
        raise FeatureEngineeringError(
            f"O DataFrame deve conter as colunas '{date_col}' e '{target_col}'."
        )

    clean = df[[date_col, target_col]].copy()
    clean[date_col] = pd.to_datetime(clean[date_col]).dt.normalize()
    clean[target_col] = clean[target_col].astype(float)
    clean = clean.sort_values(date_col, ascending=True)

    if cutoff_date is not None:
        cutoff_ts = pd.Timestamp(cutoff_date).normalize()
        clean = clean.loc[clean[date_col] <= cutoff_ts]

    if clean.empty:
        raise FeatureEngineeringError(
            "Nenhum registro disponivel apos aplicacao do filtro temporal."
        )

    if clean[date_col].duplicated().any():
        raise FeatureEngineeringError(
            "Foram detectadas datas duplicadas na serie de entrada para engenharia de atributos."
        )

    if clean[date_col].isna().any():
        raise FeatureEngineeringError("A serie nao pode conter datas nulas.")
    if not np.isfinite(clean[target_col]).all() or (clean[target_col] <= 0).any():
        raise FeatureEngineeringError("A carga deve conter apenas valores finitos e positivos.")
    if not clean[date_col].diff().dropna().eq(pd.Timedelta(days=1)).all():
        raise FeatureEngineeringError(
            "A serie deve ter continuidade diaria para calcular janelas moveis em dias."
        )

    return clean.reset_index(drop=True)


def _compute_origin_table(
    clean_series: pd.DataFrame,
    date_col: str,
    target_col: str,
    rolling_windows: tuple[int, ...] = (7, 14, 28),
) -> pd.DataFrame:
    """Calcula estatisticas moveis ancoradas estritamente na data de origem t_0.

    Para uma data de origem t_0, rolling(W) utiliza apenas [t_0 - W + 1, ..., t_0].
    Nenhuma observacao > t_0 entra no calculo.
    """
    origin_df = clean_series.rename(
        columns={date_col: "origin_date", target_col: "origin_load"}
    ).copy()

    for window in rolling_windows:
        rolled = origin_df["origin_load"].rolling(window=window, min_periods=window)
        origin_df[f"rolling_mean_{window}"] = rolled.mean()
        origin_df[f"rolling_std_{window}"] = rolled.std(ddof=1)

    return origin_df


def build_supervised_horizon_dataset(
    df: pd.DataFrame,
    horizon_days: int = 7,
    lags: tuple[int, ...] = (7, 14, 28),
    rolling_windows: tuple[int, ...] = (7, 14, 28),
    country_code: str = "BR",
    date_col: str = "data_referencia",
    target_col: str = "carga_mwmed",
    cutoff_date: pd.Timestamp | str | None = None,
    calendar_profile: str = "national",
) -> pd.DataFrame:
    """Constroi o dataset supervisionado multi-horizonte (h = 1..H) sem vazamento temporal.

    Cada linha representa um par (origin_date = t_0, horizon = h) prevendo target_date = t_0 + h:
    - Quando cutoff_date e informado, tanto origin_date quanto target_date sao estritamente
      <= cutoff_date (impedindo que qualquer dado do periodo de teste vaze para o treino).
    - Os lags (7, 14, 28 dias) sao relativos a target_date (tau - L = t_0 + h - L <= t_0,
      pois L >= 7 >= h).
    - As medias e desvios moveis (7, 14, 28 dias) sao calculados exclusivamente ate
      origin_date (t_0).

    Args:
        df: Serie temporal diaria contendo data_referencia e carga_mwmed.
        horizon_days: Horizonte maximo de previsao em dias (padrao: 7).
        lags: Lags sazonais relativos a data alvo (todos devem ser >= horizon_days).
        rolling_windows: Tamanhos das janelas moveis ancoradas na data de origem.
        country_code: Codigo do pais para feriados (padrao: 'BR').
        date_col: Nome da coluna de data.
        target_col: Nome da coluna de carga.
        cutoff_date: Data limite superior opcional para recorte causal de treinamento.

    Returns:
        pd.DataFrame: Dataset supervisionado livre de nulos contendo origin_date, target_date,
            horizon, todas as colunas de FEATURE_COLUMNS e target (y_true).
    """
    if min(lags) < horizon_days:
        raise FeatureEngineeringError(
            f"Todos os lags {lags} devem ser >= horizon_days ({horizon_days}) "
            "para evitar vazamento temporal."
        )

    clean = _prepare_clean_series(
        df, date_col=date_col, target_col=target_col, cutoff_date=cutoff_date
    )
    load_lookup = clean.set_index(date_col)[target_col]
    origin_table = _compute_origin_table(
        clean, date_col=date_col, target_col=target_col, rolling_windows=rolling_windows
    )

    frames: list[pd.DataFrame] = []
    for h in range(1, horizon_days + 1):
        step_df = origin_table.copy()
        step_df["horizon"] = int(h)
        step_df["target_date"] = step_df["origin_date"] + pd.Timedelta(days=h)
        step_df["y_true"] = step_df["target_date"].map(load_lookup)

        for lag in lags:
            lag_dates = step_df["target_date"] - pd.Timedelta(days=lag)
            step_df[f"lag_{lag}"] = lag_dates.map(load_lookup)

        frames.append(step_df)

    combined = pd.concat(frames, axis=0, ignore_index=True)

    cal_df = build_calendar_features(combined["target_date"], country_code=country_code)
    for col in cal_df.columns:
        combined[col] = cal_df[col].to_numpy()

    combined["lag_7_diff_14"] = combined["lag_7"] - combined["lag_14"]
    combined["lag_7_ratio_14"] = combined["lag_7"] / combined["lag_14"]
    combined["lag_7_vs_rolling_mean_7"] = combined["lag_7"] - combined["rolling_mean_7"]

    feature_columns = get_feature_columns(calendar_profile)
    if calendar_profile == "critical":
        extra = critical_calendar(combined["target_date"])
        for col in extra:
            combined[col] = extra[col].to_numpy()
    required_non_null = [*feature_columns, "y_true"]
    combined = combined.dropna(subset=required_non_null).copy()
    combined = combined.sort_values(["origin_date", "horizon"], ascending=[True, True]).reset_index(
        drop=True
    )

    ordered_cols = ["origin_date", "target_date", *feature_columns, "y_true"]
    return combined[ordered_cols]


def build_inference_features_for_origin(
    df: pd.DataFrame,
    origin_date: pd.Timestamp | str,
    horizon_days: int = 7,
    lags: tuple[int, ...] = (7, 14, 28),
    rolling_windows: tuple[int, ...] = (7, 14, 28),
    country_code: str = "BR",
    date_col: str = "data_referencia",
    target_col: str = "carga_mwmed",
    calendar_profile: str = "national",
) -> pd.DataFrame:
    """Gera a matriz de atributos para os proximos 7 dias (t_0 + 1 .. t_0 + 7) a partir de t_0.

    Filtra estritamente a serie historica para data_referencia <= origin_date antes de
    qualquer operacao, garantindo impossibilidade fisica de vazamento de dados futuros.

    Args:
        df: Serie historica diaria.
        origin_date: Data de origem da previsao (t_0).
        horizon_days: Horizonte de previsao (padrao: 7).
        lags: Lags relativos a data alvo (padrao: 7, 14, 28).
        rolling_windows: Janelas moveis calculadas ate origin_date (padrao: 7, 14, 28).
        country_code: Codigo ISO de feriados (padrao: 'BR').
        date_col: Coluna de data.
        target_col: Coluna de carga.

    Returns:
        pd.DataFrame: DataFrame com 7 linhas (h = 1..7) contendo origin_date, target_date
            e todas as colunas de FEATURE_COLUMNS.

    Raises:
        FeatureEngineeringError: Se origin_date nao existir na serie ou nao houver historico
            suficiente (minimo de max(lags + rolling_windows) dias anteriores).
    """
    if min(lags) < horizon_days:
        raise FeatureEngineeringError(
            f"Todos os lags {lags} devem ser >= horizon_days ({horizon_days})."
        )

    origin_ts = pd.Timestamp(origin_date).normalize()
    clean = _prepare_clean_series(
        df, date_col=date_col, target_col=target_col, cutoff_date=origin_ts
    )

    if origin_ts not in set(clean[date_col]):
        raise FeatureEngineeringError(
            f"A data de origem {origin_ts.strftime('%Y-%m-%d')} nao foi encontrada na serie."
        )

    origin_table = _compute_origin_table(
        clean, date_col=date_col, target_col=target_col, rolling_windows=rolling_windows
    )
    origin_row = origin_table.loc[origin_table["origin_date"] == origin_ts]
    if origin_row.empty:
        raise FeatureEngineeringError(
            f"Nao foi possivel localizar estatisticas na origem {origin_ts.strftime('%Y-%m-%d')}."
        )

    load_lookup = clean.set_index(date_col)[target_col]

    rows: list[dict[str, float | int | pd.Timestamp]] = []
    base_stats = origin_row.iloc[0]

    for h in range(1, horizon_days + 1):
        target_ts = origin_ts + pd.Timedelta(days=h)
        row_dict: dict[str, float | int | pd.Timestamp] = {
            "origin_date": origin_ts,
            "target_date": target_ts,
            "horizon": int(h),
            "origin_load": float(base_stats["origin_load"]),
        }
        for window in rolling_windows:
            row_dict[f"rolling_mean_{window}"] = float(base_stats[f"rolling_mean_{window}"])
            row_dict[f"rolling_std_{window}"] = float(base_stats[f"rolling_std_{window}"])

        for lag in lags:
            lag_ts = target_ts - pd.Timedelta(days=lag)
            if lag_ts not in load_lookup.index:
                raise FeatureEngineeringError(
                    f"Historico insuficiente na origem {origin_ts.strftime('%Y-%m-%d')}: "
                    f"data de lag {lag_ts.strftime('%Y-%m-%d')} ausente."
                )
            row_dict[f"lag_{lag}"] = float(load_lookup.loc[lag_ts])

        rows.append(row_dict)

    infer_df = pd.DataFrame(rows)
    cal_df = build_calendar_features(infer_df["target_date"], country_code=country_code)
    for col in cal_df.columns:
        infer_df[col] = cal_df[col].to_numpy()

    infer_df["lag_7_diff_14"] = infer_df["lag_7"] - infer_df["lag_14"]
    infer_df["lag_7_ratio_14"] = infer_df["lag_7"] / infer_df["lag_14"]
    infer_df["lag_7_vs_rolling_mean_7"] = infer_df["lag_7"] - infer_df["rolling_mean_7"]

    feature_columns = get_feature_columns(calendar_profile)
    if calendar_profile == "critical":
        extra = critical_calendar(infer_df["target_date"])
        for col in extra:
            infer_df[col] = extra[col].to_numpy()
    if infer_df[list(feature_columns)].isna().any().any():
        raise FeatureEngineeringError(
            f"Historico insuficiente na origem {origin_ts.strftime('%Y-%m-%d')} "
            "para preencher todas as janelas moveis e lags sem valores nulos."
        )

    ordered_cols = ["origin_date", "target_date", *feature_columns]
    return infer_df[ordered_cols]


def get_feature_columns(calendar_profile: str = "national") -> tuple[str, ...]:
    if calendar_profile not in {"national", "critical"}:
        raise FeatureEngineeringError("Perfil de calendario desconhecido.")
    return FEATURE_COLUMNS + (CRITICAL_COLUMNS if calendar_profile == "critical" else ())
