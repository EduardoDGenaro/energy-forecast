"""Faixas empiricas causais e limites configuraveis, sem prometer cobertura futura."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from energy_forecast.evaluation.metrics import EvaluationError, calculate_mape


@dataclass(frozen=True)
class RiskPolicy:
    status: str = "reference"
    coverage: float = 0.90
    calibration_min_samples: int = 26
    calibration_window: int = 52
    daily_error_limit_pct: dict[int, float] = field(
        default_factory=lambda: dict.fromkeys(range(1, 8), 5.0)
    )
    max_breach_rate: float = 0.20
    underprediction_weight: float = 2.0
    overprediction_weight: float = 1.0
    monitor_recent_days: int = 28
    monitor_reference_days: int = 84
    monitor_min_samples: int = 14
    degradation_ratio: float = 1.25

    def __post_init__(self) -> None:
        if self.status not in {"reference", "approved"}:
            raise EvaluationError("Status de politica desconhecido.")
        if not 0 < self.coverage < 1 or not 0 <= self.max_breach_rate <= 1:
            raise EvaluationError("Cobertura e taxa de violacao invalidas.")
        if not 1 <= self.calibration_min_samples <= self.calibration_window:
            raise EvaluationError("Janela de calibracao invalida.")
        if set(self.daily_error_limit_pct) != set(range(1, 8)) or any(
            not np.isfinite(v) or v <= 0 for v in self.daily_error_limit_pct.values()
        ):
            raise EvaluationError("Defina limites positivos para os sete horizontes.")
        if (
            any(
                not np.isfinite(v) or v <= 0
                for v in (
                    self.underprediction_weight,
                    self.overprediction_weight,
                    self.degradation_ratio,
                )
            )
            or self.degradation_ratio <= 1
        ):
            raise EvaluationError("Pesos e razao de degradacao invalidos.")
        if min(self.monitor_recent_days, self.monitor_reference_days, self.monitor_min_samples) < 1:
            raise EvaluationError("Janelas de monitoramento invalidas.")


def load_risk_policy(root: Path) -> RiskPolicy:
    path = root / "config" / "risk_policy.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if not isinstance(raw, dict):
        raise EvaluationError("A politica deve conter um mapeamento YAML.")
    if "daily_error_limit_pct" in raw:
        raw["daily_error_limit_pct"] = {
            int(k): float(v) for k, v in raw["daily_error_limit_pct"].items()
        }
    return RiskPolicy(**raw)


def validate_predictions(frame: pd.DataFrame, actuals: bool = True) -> pd.DataFrame:
    result = frame.copy()
    for col in ("origin_date", "target_date"):
        result[col] = pd.to_datetime(result[col]).dt.normalize()
    numeric = ["y_pred_lgbm", "y_pred_baseline"] + (["y_true"] if actuals else [])
    if result.empty or result[["origin_date", "target_date"]].isna().any().any():
        raise EvaluationError("Previsoes vazias ou com datas invalidas.")
    if not np.isfinite(result[numeric].to_numpy(dtype=float)).all():
        raise EvaluationError("Previsoes e valores reais devem ser finitos.")
    if actuals and (result["y_true"] <= 0).any():
        raise EvaluationError("Valores reais devem ser positivos.")
    if result.duplicated(["origin_date", "horizon"]).any():
        raise EvaluationError("Previsoes duplicadas por origem e horizonte.")
    if (
        not result["horizon"].isin(range(1, 8)).all()
        or not (result["target_date"] - result["origin_date"]).dt.days.eq(result["horizon"]).all()
    ):
        raise EvaluationError("Horizonte e datas inconsistentes.")
    return result


def add_prediction_intervals(
    forecast: pd.DataFrame, historical: pd.DataFrame, policy: RiskPolicy
) -> pd.DataFrame:
    """Quantil de erros absolutos anteriores, por h, com correcao de amostra finita.

    Apenas alvos conhecidos ate a origem entram na calibracao. Com poucos dados,
    retorna NaN e status insuficiente; nao usa um quantil interpolado artificial.
    A dependencia temporal impede interpretar a meta nominal como garantia.
    """
    output = validate_predictions(forecast, actuals=False)
    history = validate_predictions(historical)
    history["residual"] = (history["y_true"] - history["y_pred_lgbm"]).abs()
    records = []
    for row in output.itertuples():
        prior = (
            history.loc[
                (history["horizon"] == row.horizon)
                & (history["target_date"] <= row.origin_date)
                & (history["origin_date"] < row.origin_date)
            ]
            .sort_values("target_date")
            .tail(policy.calibration_window)
        )
        n = len(prior)
        rank = math.ceil((n + 1) * policy.coverage)
        ready = n >= policy.calibration_min_samples and rank <= n
        radius = float(np.sort(prior["residual"])[rank - 1]) if ready else np.nan
        records.append(
            {
                "lower_bound": max(0.0, row.y_pred_lgbm - radius) if ready else np.nan,
                "upper_bound": row.y_pred_lgbm + radius if ready else np.nan,
                "calibration_n": n,
                "nominal_coverage": policy.coverage,
                "calibration_last_target": prior["target_date"].max() if n else pd.NaT,
                "interval_status": "calibrated" if ready else "insufficient_history",
            }
        )
    return pd.concat([output.reset_index(drop=True), pd.DataFrame(records)], axis=1)


def interval_coverage(scored: pd.DataFrame) -> pd.DataFrame:
    """Avalia apenas faixas que existiriam antes de observar o alvo, sem usar o futuro."""
    ready = scored.loc[scored["interval_status"] == "calibrated"].copy()
    if ready.empty:
        return pd.DataFrame(columns=["horizon", "count", "coverage_pct", "width_mwmed"])
    ready["covered"] = ready["y_true"].between(ready["lower_bound"], ready["upper_bound"])
    ready["width"] = ready["upper_bound"] - ready["lower_bound"]
    return (
        ready.groupby("horizon")
        .agg(
            count=("covered", "size"),
            coverage_pct=("covered", lambda s: s.mean() * 100),
            width_mwmed=("width", "mean"),
        )
        .reset_index()
    )


def weighted_error(frame: pd.DataFrame, policy: RiskPolicy, column: str = "y_pred_lgbm") -> float:
    residual = frame["y_true"] - frame[column]
    return float(
        (
            residual.clip(lower=0) * policy.underprediction_weight
            + (-residual).clip(lower=0) * policy.overprediction_weight
        ).mean()
    )


def operational_limits(frame: pd.DataFrame, policy: RiskPolicy) -> pd.DataFrame:
    clean = validate_predictions(frame)
    rows = []
    for h, group in clean.groupby("horizon"):
        limit = policy.daily_error_limit_pct[int(h)]
        errors = (group["y_true"] - group["y_pred_lgbm"]).abs() / group["y_true"] * 100
        rate = float((errors > limit).mean())
        rows.append(
            {
                "horizon": int(h),
                "count": len(group),
                "limit_pct": limit,
                "mape_pct": float(errors.mean()),
                "breach_rate_pct": rate * 100,
                "within_limit_pct": 100 * (1 - rate),
                "weighted_error_mwmed": weighted_error(group, policy),
                "status": "attention" if rate > policy.max_breach_rate else "within_reference",
            }
        )
    return pd.DataFrame(rows)


def degradation_monitor(frame: pd.DataFrame, policy: RiskPolicy) -> dict:
    """Janelas por data alvo; escolhe menor antecedencia se houver previsoes sobrepostas."""
    if frame.empty:
        return {
            "status": "waiting",
            "alerts": [],
            "recent_count": 0,
            "message": "Aguardando valores reais de previsoes emitidas.",
        }
    clean = validate_predictions(frame).sort_values(["target_date", "horizon"])
    daily = clean.drop_duplicates("target_date", keep="first")
    end = daily["target_date"].max()
    recent_start = end - pd.Timedelta(days=policy.monitor_recent_days)
    recent = daily.loc[daily["target_date"] > recent_start]
    reference = daily.loc[
        (daily["target_date"] <= recent_start)
        & (daily["target_date"] > recent_start - pd.Timedelta(days=policy.monitor_reference_days))
    ]
    result = {
        "status": "waiting",
        "alerts": [],
        "recent_count": len(recent),
        "reference_count": len(reference),
        "last_target": str(end.date()),
        "message": "Amostra recente insuficiente para alertas.",
    }
    if len(recent) < policy.monitor_min_samples:
        return result
    model = calculate_mape(recent["y_true"], recent["y_pred_lgbm"])
    baseline = calculate_mape(recent["y_true"], recent["y_pred_baseline"])
    result.update(status="ok", recent_mape=model, baseline_mape=baseline, message="Sem alertas.")
    if model > baseline:
        result["alerts"].append("baseline_better")
    if len(reference) >= policy.monitor_min_samples:
        ref = calculate_mape(reference["y_true"], reference["y_pred_lgbm"])
        result["reference_mape"] = ref
        if model > ref * policy.degradation_ratio:
            result["alerts"].append("error_increase")
    else:
        result["reference_status"] = "insufficient_history"
    limits = operational_limits(recent, policy)
    if (limits["status"] == "attention").any():
        result["alerts"].append("daily_limit_breaches")
    if {"lower_bound", "upper_bound", "interval_status"}.issubset(recent.columns):
        ready = recent.loc[recent["interval_status"] == "calibrated"]
        if len(ready) >= policy.monitor_min_samples:
            coverage = float(
                ready["y_true"].between(ready["lower_bound"], ready["upper_bound"]).mean()
            )
            result["observed_coverage_pct"] = coverage * 100
            if coverage < policy.coverage:
                result["alerts"].append("coverage_below_target")
    if result["alerts"]:
        result.update(status="attention", message="Revisar desempenho nas janelas recentes.")
    return result
