"""Temperatura de rodadas arquivadas; nunca usa temperatura realizada no futuro."""

from __future__ import annotations

import numpy as np
import pandas as pd

WEATHER_COLUMNS = ("temperature_mean", "cooling_degrees", "heating_degrees")


def join_weather(frame: pd.DataFrame, snapshots: pd.DataFrame, *, required: bool) -> pd.DataFrame:
    weather = snapshots.copy()
    for col in ("origin_date", "target_date"):
        weather[col] = pd.to_datetime(weather[col]).dt.normalize()
    if weather.duplicated(["origin_date", "target_date"]).any():
        raise ValueError("Snapshots meteorologicos duplicados.")
    available = pd.to_datetime(weather["available_at"], utc=True)
    deadline = weather["origin_date"].dt.tz_localize("America/Sao_Paulo") + pd.Timedelta(days=1)
    if available.isna().any() or (available >= deadline).any():
        raise ValueError("Temperatura indisponivel na origem da previsao.")
    if "run_time" in weather:
        run = pd.to_datetime(weather["run_time"], utc=True)
        if run.isna().any() or (run > available).any():
            raise ValueError("Inicializacao meteorologica posterior a disponibilidade declarada.")
    if (
        not np.isfinite(weather["temperature_mean"]).all()
        or not weather["temperature_mean"].between(-60, 60).all()
    ):
        raise ValueError("Temperatura invalida.")
    weather["cooling_degrees"] = (weather["temperature_mean"] - 22).clip(lower=0)
    weather["heating_degrees"] = (18 - weather["temperature_mean"]).clip(lower=0)
    joined = frame.merge(
        weather[["origin_date", "target_date", *WEATHER_COLUMNS]],
        on=["origin_date", "target_date"],
        how="left",
        validate="many_to_one",
    )
    if required and joined[list(WEATHER_COLUMNS)].isna().any().any():
        raise ValueError("Sem snapshot meteorologico completo para a origem.")
    return joined.dropna(subset=list(WEATHER_COLUMNS))
