"""Funcoes de calculo de metricas de erro (MAE, MAPE) e ganho percentual sobre o Baseline."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


class EvaluationError(ValueError):
    """Excecao lancada quando entradas de avaliacao sao invalidas, vazias ou inconsistentes."""


def _validate_arrays(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    y_pred: Sequence[float] | np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Converte e valida vetores de valores reais e previstos."""
    true_arr = np.asarray(y_true, dtype=float)
    pred_arr = np.asarray(y_pred, dtype=float)

    if true_arr.ndim != 1 or pred_arr.ndim != 1:
        raise EvaluationError("Os vetores y_true e y_pred devem ser unidimensionais.")
    if true_arr.size == 0:
        raise EvaluationError("Os vetores de avaliacao nao podem estar vazios.")
    if true_arr.size != pred_arr.size:
        raise EvaluationError(
            f"Tamanhos incompativeis entre y_true ({true_arr.size}) e y_pred ({pred_arr.size})."
        )
    if not np.isfinite(true_arr).all() or not np.isfinite(pred_arr).all():
        raise EvaluationError("Os vetores y_true e y_pred devem conter apenas valores finitos.")

    return true_arr, pred_arr


def calculate_mae(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    y_pred: Sequence[float] | np.ndarray | pd.Series,
) -> float:
    """Calcula o Erro Absoluto Medio (MAE).

    Args:
        y_true: Valores reais observados.
        y_pred: Valores previstos pelo modelo.

    Returns:
        float: MAE na mesma unidade da variavel alvo (MWmed).
    """
    true_arr, pred_arr = _validate_arrays(y_true, y_pred)
    return float(np.mean(np.abs(true_arr - pred_arr)))


def calculate_mape(
    y_true: Sequence[float] | np.ndarray | pd.Series,
    y_pred: Sequence[float] | np.ndarray | pd.Series,
) -> float:
    """Calcula o Erro Percentual Absoluto Medio (MAPE) em escala percentual (0 a 100%).

    Args:
        y_true: Valores reais observados (devem ser estritamente positivos).
        y_pred: Valores previstos pelo modelo.

    Returns:
        float: MAPE expresso em percentual (%).

    Raises:
        EvaluationError: Caso y_true contenha valores menores ou iguais a zero.
    """
    true_arr, pred_arr = _validate_arrays(y_true, y_pred)
    if np.any(true_arr <= 0.0):
        raise EvaluationError(
            "O calculo de MAPE requer valores reais (y_true) estritamente positivos (> 0)."
        )
    return float(np.mean(np.abs((true_arr - pred_arr) / true_arr)) * 100.0)


def calculate_improvement_pct(baseline_error: float, model_error: float) -> float:
    """Calcula o percentual de melhoria (reducao de erro) do modelo em relacao ao Baseline.

    Valores positivos indicam que o modelo reduziu o erro em relacao ao Baseline.
    Valores negativos indicam que o modelo apresentou desempenho inferior (erro maior).

    Args:
        baseline_error: Metrica de erro (MAE ou MAPE) do Baseline.
        model_error: Metrica de erro (MAE ou MAPE) do modelo avaliado.

    Returns:
        float: Percentual de melhoria (%).
    """
    if not np.isfinite(baseline_error) or not np.isfinite(model_error) or model_error < 0:
        raise EvaluationError("Os erros devem ser finitos e nao negativos.")
    if baseline_error <= 0.0:
        raise EvaluationError("O erro do baseline deve ser estritamente positivo (> 0).")
    return float(((baseline_error - model_error) / baseline_error) * 100.0)
