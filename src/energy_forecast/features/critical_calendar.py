"""Eventos de calendario para um candidato, sem alterar o modelo operacional atual."""

from __future__ import annotations

from datetime import timedelta

import holidays
import pandas as pd
from dateutil.easter import easter

CRITICAL_COLUMNS = (
    "is_carnival",
    "is_ash_wednesday",
    "is_corpus_christi",
    "is_bridge_day",
    "is_year_end",
)


def critical_calendar(dates: pd.Series) -> pd.DataFrame:
    dates = pd.Series(pd.to_datetime(dates)).dt.normalize().reset_index(drop=True)
    years = range(int(dates.dt.year.min()) - 1, int(dates.dt.year.max()) + 2)
    carnival, ash, corpus = set(), set(), set()
    for year in years:
        day = easter(year)
        carnival.update(pd.Timestamp(day - timedelta(days=k)) for k in (48, 47))
        ash.add(pd.Timestamp(day - timedelta(days=46)))
        corpus.add(pd.Timestamp(day + timedelta(days=60)))
    national = {pd.Timestamp(d) for d in holidays.country_holidays("BR", years=years)}
    events = national | carnival | ash | corpus
    bridges = ((dates.dt.day_of_week == 0) & (dates + pd.Timedelta(days=1)).isin(events)) | (
        (dates.dt.day_of_week == 4) & (dates - pd.Timedelta(days=1)).isin(events)
    )
    return pd.DataFrame(
        {
            "is_carnival": dates.isin(carnival).astype(int),
            "is_ash_wednesday": dates.isin(ash).astype(int),
            "is_corpus_christi": dates.isin(corpus).astype(int),
            "is_bridge_day": bridges.astype(int),
            "is_year_end": (
                ((dates.dt.month == 12) & (dates.dt.day >= 24))
                | ((dates.dt.month == 1) & (dates.dt.day <= 2))
            ).astype(int),
        }
    )


def critical_period_report(predictions: pd.DataFrame) -> pd.DataFrame:
    """Cortes predefinidos; grupos podem se sobrepor e contagens sao explicitas."""
    frame = predictions.reset_index(drop=True)
    dates = pd.to_datetime(frame["target_date"])
    events = critical_calendar(dates)
    national = holidays.country_holidays("BR", years=dates.dt.year.unique().tolist())
    masks = {
        "Todos os dias": pd.Series(True, index=frame.index),
        "Feriados nacionais": dates.isin(pd.to_datetime(list(national))),
        "Carnaval": events["is_carnival"].eq(1),
        "Quarta de Cinzas": events["is_ash_wednesday"].eq(1),
        "Corpus Christi": events["is_corpus_christi"].eq(1),
        "Pontes de feriado": events["is_bridge_day"].eq(1),
        "Fim/inicio de ano": events["is_year_end"].eq(1),
        "Dezembro": dates.dt.month.eq(12),
        "Fevereiro": dates.dt.month.eq(2),
    }
    rows = []
    for name, mask in masks.items():
        group = frame.loc[mask]
        if group.empty:
            continue
        base = (group["y_true"] - group["y_pred_baseline"]).abs() / group["y_true"] * 100
        model = (group["y_true"] - group["y_pred_lgbm"]).abs() / group["y_true"] * 100
        rows.append(
            {
                "period": name,
                "count": len(group),
                "mape_lgbm": float(model.mean()),
                "mape_baseline": float(base.mean()),
                "max_ape_pct": float(model.max()),
            }
        )
    return pd.DataFrame(rows)
