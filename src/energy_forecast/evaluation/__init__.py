"""Modulo de backtesting temporal e avaliacao comparativa (MAE e MAPE)."""

from energy_forecast.evaluation.backtesting import (
    BacktestResult,
    BacktestSummary,
    generate_backtest_origins,
    run_walk_forward_backtest,
)
from energy_forecast.evaluation.metrics import (
    EvaluationError,
    calculate_improvement_pct,
    calculate_mae,
    calculate_mape,
)

__all__ = [
    "BacktestResult",
    "BacktestSummary",
    "EvaluationError",
    "calculate_improvement_pct",
    "calculate_mae",
    "calculate_mape",
    "generate_backtest_origins",
    "run_walk_forward_backtest",
]
