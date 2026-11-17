"""Modelo LightGBM para previsao multi-horizonte (7 dias) de carga eletrica diaria."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightgbm as lgb
import pandas as pd

from energy_forecast.features.engineering import (
    build_inference_features_for_origin,
    build_supervised_horizon_dataset,
    get_feature_columns,
)
from energy_forecast.features.weather import WEATHER_COLUMNS, join_weather


class ModelTrainingError(RuntimeError):
    """Excecao lancada quando ocorre erro no treinamento ou inferencia do modelo LightGBM."""


def get_default_lgbm_params() -> dict[str, Any]:
    """Retorna os hiperparametros padrao do modelo LightGBM para previsao de carga diaria."""
    return {
        "objective": "regression_l1",
        "n_estimators": 350,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "max_depth": 6,
        "min_child_samples": 20,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }


@dataclass
class LightGBMForecaster:
    """Wrapper de treinamento e previsao de 7 dias com LightGBM livre de vazamento temporal."""

    horizon_days: int = 7
    lags: tuple[int, ...] = (7, 14, 28)
    rolling_windows: tuple[int, ...] = (7, 14, 28)
    country_code: str = "BR"
    date_col: str = "data_referencia"
    target_col: str = "carga_mwmed"
    model_params: dict[str, Any] = field(default_factory=get_default_lgbm_params)
    calendar_profile: str = "national"
    weather_snapshots: pd.DataFrame | None = field(default=None, repr=False)
    training_origins: tuple[pd.Timestamp, ...] | None = field(default=None, repr=False)
    model_: lgb.LGBMRegressor | None = field(default=None, init=False, repr=False)
    train_Origin_cutoff_: pd.Timestamp | None = field(default=None, init=False)
    train_rows_: int = field(default=0, init=False)

    def fit(
        self,
        df: pd.DataFrame,
        cutoff_date: pd.Timestamp | str | None = None,
    ) -> LightGBMForecaster:
        """Ajusta o modelo LightGBM utilizando apenas observacoes <= cutoff_date.

        Args:
            df: Serie temporal diaria historica.
            cutoff_date: Data limite superior (origem de previsao). Tanto origin_date
                quanto target_date no conjunto supervisionado serao <= cutoff_date.

        Returns:
            LightGBMForecaster: Instancia ajustada.

        Raises:
            ModelTrainingError: Caso nao haja amostras supervisionadas suficientes.
        """
        sup_df = build_supervised_horizon_dataset(
            df=df,
            horizon_days=self.horizon_days,
            lags=self.lags,
            rolling_windows=self.rolling_windows,
            country_code=self.country_code,
            date_col=self.date_col,
            target_col=self.target_col,
            cutoff_date=cutoff_date,
            calendar_profile=self.calendar_profile,
        )

        if self.training_origins is not None:
            sup_df = sup_df.loc[sup_df["origin_date"].isin(self.training_origins)]
        if self.weather_snapshots is not None:
            sup_df = join_weather(sup_df, self.weather_snapshots, required=False)
        if len(sup_df) < 30:
            raise ModelTrainingError(
                f"Amostras supervisionadas insuficientes para treinamento ({len(sup_df)} linhas)."
            )

        x_train = sup_df[list(self.feature_columns)]
        y_train = sup_df["y_true"].to_numpy(dtype=float)

        regressor = lgb.LGBMRegressor(**self.model_params)
        regressor.fit(x_train, y_train)

        self.model_ = regressor
        self.train_rows_ = len(sup_df)
        self.train_Origin_cutoff_ = (
            pd.Timestamp(cutoff_date).normalize()
            if cutoff_date is not None
            else pd.Timestamp(sup_df["target_date"].max()).normalize()
        )
        return self

    def predict_from_origin(
        self,
        df: pd.DataFrame,
        origin_date: pd.Timestamp | str,
    ) -> pd.DataFrame:
        """Gera previsao de 7 dias a partir de origin_date usando apenas dados <= origin_date.

        Args:
            df: Serie historica diaria.
            origin_date: Data de origem da previsao (t_0).

        Returns:
            pd.DataFrame: Previsoes para h = 1..horizon_days com colunas
                ['origin_date', 'target_date', 'horizon', 'y_pred_lgbm'].

        Raises:
            ModelTrainingError: Se o modelo ainda nao tiver sido treinado.
        """
        if self.model_ is None:
            raise ModelTrainingError("O modelo LightGBM deve ser ajustado (fit) antes de prever.")

        if (
            self.train_Origin_cutoff_ is not None
            and pd.Timestamp(origin_date).normalize() < self.train_Origin_cutoff_
        ):
            raise ModelTrainingError(
                "A origem da previsao nao pode anteceder o corte de treinamento do modelo."
            )

        infer_df = build_inference_features_for_origin(
            df=df,
            origin_date=origin_date,
            horizon_days=self.horizon_days,
            lags=self.lags,
            rolling_windows=self.rolling_windows,
            country_code=self.country_code,
            date_col=self.date_col,
            target_col=self.target_col,
            calendar_profile=self.calendar_profile,
        )

        if self.weather_snapshots is not None:
            infer_df = join_weather(infer_df, self.weather_snapshots, required=True)
        x_infer = infer_df[list(self.feature_columns)]
        preds = self.model_.predict(x_infer)

        result = infer_df[["origin_date", "target_date", "horizon"]].copy()
        result["y_pred_lgbm"] = preds.astype(float)
        return result.reset_index(drop=True)

    def get_feature_importance(self) -> pd.DataFrame:
        """Retorna a importancia das features do modelo ajustado ordenada decrescentemente."""
        if self.model_ is None:
            raise ModelTrainingError("O modelo LightGBM ainda nao foi ajustado.")

        importances = self.model_.feature_importances_
        imp_df = pd.DataFrame(
            {
                "feature": list(self.feature_columns),
                "importance": importances.astype(float),
            }
        )
        return imp_df.sort_values("importance", ascending=False).reset_index(drop=True)

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return get_feature_columns(self.calendar_profile) + (
            WEATHER_COLUMNS if self.weather_snapshots is not None else ()
        )
