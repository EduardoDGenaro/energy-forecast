"""Modelo Baseline Sazonal (repeticao do mesmo dia da semana anterior, t-7)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class BaselineForecastError(ValueError):
    """Excecao lancada quando nao ha historico suficiente ate a origem para o Baseline."""


@dataclass(frozen=True)
class SeasonalNaiveBaseline:
    """Previsor ingenuo sazonal baseado no valor observado no mesmo dia da semana anterior.

    Para uma data de origem t_0 e um horizonte h in {1, ..., 7}, a previsao para a data
    alvo tau = t_0 + h e dada por:
        y_hat(tau) = y(tau - 7) = y(t_0 + h - 7)

    Como h <= 7, o instante t_0 + h - 7 e sempre <= t_0, garantindo que apenas valores
    ja observados ate a data de origem sejam utilizados.
    """

    horizon_days: int = 7
    seasonal_period_days: int = 7
    date_col: str = "data_referencia"
    target_col: str = "carga_mwmed"

    def __post_init__(self) -> None:
        if self.horizon_days <= 0:
            raise BaselineForecastError("O horizonte (horizon_days) deve ser positivo.")
        if self.seasonal_period_days < self.horizon_days:
            raise BaselineForecastError(
                f"O periodo sazonal ({self.seasonal_period_days}) deve ser >= horizon_days "
                f"({self.horizon_days}) para garantir uso exclusivo de dados <= origin_date."
            )

    def predict_from_origin(
        self,
        df: pd.DataFrame,
        origin_date: pd.Timestamp | str,
    ) -> pd.DataFrame:
        """Gera a previsao de 7 dias a partir de origin_date usando apenas dados <= origin_date.

        Args:
            df: DataFrame contendo a serie historica.
            origin_date: Data de origem da previsao (t_0).

        Returns:
            pd.DataFrame: Previsoes para h = 1..horizon_days com colunas
                ['origin_date', 'target_date', 'horizon', 'reference_date', 'y_pred_baseline'].

        Raises:
            BaselineForecastError: Caso faltem colunas ou observacoes historicas necessarias.
        """
        if self.date_col not in df.columns or self.target_col not in df.columns:
            raise BaselineForecastError(
                f"O DataFrame deve conter as colunas '{self.date_col}' e '{self.target_col}'."
            )

        origin_ts = pd.Timestamp(origin_date).normalize()

        # Filtro causal estrito: descarta qualquer observacao posterior a data de origem
        history = df[[self.date_col, self.target_col]].copy()
        history[self.date_col] = pd.to_datetime(history[self.date_col]).dt.normalize()
        history[self.target_col] = history[self.target_col].astype(float)
        history = history.loc[history[self.date_col] <= origin_ts]

        if history.empty:
            raise BaselineForecastError(
                "Nenhuma observacao disponivel ate a data de origem "
                f"{origin_ts.strftime('%Y-%m-%d')}."
            )

        if history[self.date_col].duplicated().any():
            raise BaselineForecastError("A serie historica contem datas duplicadas.")

        lookup = history.set_index(self.date_col)[self.target_col]

        records: list[dict[str, object]] = []
        for h in range(1, self.horizon_days + 1):
            target_ts = origin_ts + pd.Timedelta(days=h)
            ref_ts = target_ts - pd.Timedelta(days=self.seasonal_period_days)

            # Verificacao explicita de causalidade: ref_ts nunca pode ultrapassar origin_ts
            assert ref_ts <= origin_ts

            if ref_ts not in lookup.index:
                raise BaselineForecastError(
                    f"Observacao historica de referencia {ref_ts.strftime('%Y-%m-%d')} "
                    f"ausente para prever {target_ts.strftime('%Y-%m-%d')} "
                    f"(origem {origin_ts.strftime('%Y-%m-%d')})."
                )

            records.append(
                {
                    "origin_date": origin_ts,
                    "target_date": target_ts,
                    "horizon": int(h),
                    "reference_date": ref_ts,
                    "y_pred_baseline": float(lookup.loc[ref_ts]),
                }
            )

        return pd.DataFrame(records)
