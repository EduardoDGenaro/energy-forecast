"""Baixa rodadas semanais de temperatura ECMWF para experimento isolado e reproduzivel."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from energy_forecast.features.engineering import load_gold_series
from energy_forecast.utils.config import load_settings


def main() -> int:
    settings = load_settings()
    gold = load_gold_series(settings)
    end = gold["data_referencia"].max() - pd.Timedelta(days=7)
    # Same Saturday origins as the energy backtest; archive begins in March 2024.
    # Anchor backwards to retain the exact weekday of the original backtest.
    origins = pd.date_range(
        end=end, periods=((end - pd.Timestamp("2024-03-16")).days // 7 + 1), freq="7D"
    )
    folder = settings.paths.root_dir / "data" / "weather"
    folder.mkdir(parents=True, exist_ok=True)

    def fetch(origin: pd.Timestamp) -> list[dict]:
        path = folder / f"ecmwf_{origin:%Y-%m-%d}.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            response = requests.get(
                "https://single-runs-api.open-meteo.com/v1/forecast",
                params={
                    "latitude": "-23.55,-19.92,-22.91,-15.79",
                    "longitude": "-46.63,-43.94,-43.17,-47.88",
                    "hourly": "temperature_2m",
                    "models": "ecmwf_ifs",
                    "run": origin.strftime("%Y-%m-%dT00:00"),
                    "forecast_days": 9,
                    "timezone": "America/Sao_Paulo",
                },
                timeout=45,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, list) or len(payload) != 4:
                raise ValueError("Resposta meteorologica incompleta.")
            path.write_text(json.dumps(payload), encoding="utf-8")
        stations = []
        for item in payload:
            hourly = pd.DataFrame(
                {
                    "time": pd.to_datetime(item["hourly"]["time"]),
                    "temperature": item["hourly"]["temperature_2m"],
                }
            )
            daily = hourly.groupby(hourly["time"].dt.normalize()).agg(
                mean=("temperature", "mean"), count=("temperature", "count")
            )
            stations.append(daily.loc[daily["count"] == 24, "mean"])
        average = pd.concat(stations, axis=1).dropna().mean(axis=1)
        rows = []
        for h in range(1, 8):
            target = origin + pd.Timedelta(days=h)
            if target not in average.index:
                return []
            rows.append(
                {
                    "origin_date": origin,
                    "target_date": target,
                    "temperature_mean": float(average.loc[target]),
                    "available_at": f"{origin:%Y-%m-%d}T06:00:00+00:00",
                    "run_time": f"{origin:%Y-%m-%d}T00:00:00+00:00",
                }
            )
        return rows

    # Incomplete runs are excluded as entire origins, never imputed or partly scored.
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows = [row for result in pool.map(fetch, origins) for row in result]
    result = pd.DataFrame(rows)
    result.to_parquet(folder / "temperature_snapshots.parquet", index=False)
    complete = pd.to_datetime(result["origin_date"]).unique()
    coverage = {
        "requested_origins": len(origins),
        "complete_origins": len(complete),
        "excluded_origins": [str(o.date()) for o in origins if o not in complete],
        "method": "4-city mean; complete 24h/day at all 4 stations; no imputation",
    }
    (folder / "coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    print(f"Snapshots: {len(origins)} origens, {len(rows)} alvos. Fonte: Open-Meteo/ECMWF.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
