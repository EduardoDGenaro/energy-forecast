"""Arquivo imutavel de previsoes emitidas e reconciliacao com valores reais posteriores."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from energy_forecast.evaluation.risk import validate_predictions


def archive_forecast(
    forecast: pd.DataFrame, predictions_dir: Path, issued_at: datetime | None = None
) -> Path:
    """Nao reescreve emissoes antigas; carimba a hora real, nunca a data da origem passada."""
    frame = validate_predictions(forecast, actuals=False)
    issued_at = issued_at or datetime.now(ZoneInfo("America/Sao_Paulo"))
    if issued_at.tzinfo is None:
        raise ValueError("A emissao deve informar fuso horario.")
    frame["issued_at"] = issued_at.isoformat()
    frame["model_profile"] = "national"
    raw = frame.sort_values(["origin_date", "horizon"]).to_csv(index=False).encode()
    folder = predictions_dir / "issued_forecasts"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{issued_at:%Y%m%dT%H%M%S}_{hashlib.sha256(raw).hexdigest()[:16]}.parquet"
    if not path.exists():
        frame.to_parquet(path, index=False)
    return path


def reconcile_issued_forecasts(predictions_dir: Path, gold: pd.DataFrame) -> pd.DataFrame:
    """So pontua dias inteiros posteriores a emissao, com alvo ja observado.

    Usa a primeira emissao por origem/horizonte para impedir cherry-picking de reemissoes.
    O backtest nunca e importado neste arquivo de operacao.
    """
    files = sorted((predictions_dir / "issued_forecasts").glob("*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(path) for path in files]
    archive = pd.concat(frames, ignore_index=True)
    issued = pd.to_datetime(archive["issued_at"], utc=True)
    archive["issued_day"] = (
        issued.dt.tz_convert("America/Sao_Paulo").dt.tz_localize(None).dt.normalize()
    )
    archive["issued_at_sort"] = issued
    archive["target_date"] = pd.to_datetime(archive["target_date"])
    archive = archive.sort_values("issued_at_sort").drop_duplicates(["origin_date", "horizon"])
    archive = archive.loc[archive["target_date"] > archive["issued_day"]].copy()
    actuals = gold.assign(data_referencia=pd.to_datetime(gold["data_referencia"])).set_index(
        "data_referencia"
    )["carga_mwmed"]
    archive["y_true"] = archive["target_date"].map(actuals)
    return archive.dropna(subset=["y_true"]).reset_index(drop=True)
