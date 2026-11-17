"""Testes unitarios para os utilitarios de linguagem acessivel e formatacao do dashboard."""

from __future__ import annotations

import pandas as pd
from dashboard.app import (
    GLOSSARY_ENTRIES,
    build_horizon_comparison_table,
    build_operational_forecast_table,
    forecast_age_days,
    format_horizon_label,
    format_pt_number,
    translate_feature_name,
)

from energy_forecast.features.engineering import FEATURE_COLUMNS
from energy_forecast.utils.config import get_project_root


def test_format_pt_number_and_horizon_labels() -> None:
    """Valida formatacao numerica brasileira e rotulos acessiveis de antecedencia."""
    assert format_pt_number(1701.53, decimals=0) == "1.702"
    assert format_pt_number(2322.70, decimals=2, suffix=" MWmed") == "2.322,70 MWmed"
    assert format_horizon_label(1) == "1 dia a frente"
    assert format_horizon_label(5) == "5 dias a frente"


def test_all_model_features_have_accessible_portuguese_labels() -> None:
    """Garante que todas as features do modelo possuem traducao para linguagem nao tecnica."""
    for feature in FEATURE_COLUMNS:
        translated = translate_feature_name(feature)
        assert translated != feature, f"Feature sem traducao amigavel: {feature}"


def test_operational_and_horizon_tables_produce_readable_columns() -> None:
    """Verifica construcao das tabelas amigaveis exibidas no dashboard."""
    latest_df = pd.DataFrame(
        {
            "origin_date": pd.to_datetime(["2026-11-14", "2026-11-14"]),
            "target_date": pd.to_datetime(["2026-11-15", "2026-11-16"]),
            "horizon": [1, 2],
            "y_pred_baseline": [42726.5, 44313.3],
            "y_pred_lgbm": [39053.2, 46037.8],
            "diff_lgbm_minus_baseline": [-3673.3, 1724.5],
        }
    )
    op_table = build_operational_forecast_table(latest_df)
    assert list(op_table["Data"]) == ["15/11/2026", "16/11/2026"]
    assert list(op_table["Dia da semana"]) == ["Domingo", "Segunda-feira"]
    assert op_table.loc[0, "Antecedencia"] == "1 dia a frente"

    horizon_df = pd.DataFrame(
        {
            "horizon": [1],
            "mae_baseline": [2106.04],
            "mae_lgbm": [1342.11],
            "mae_improvement_pct": [36.27],
            "mape_baseline": [5.31],
            "mape_lgbm": [3.37],
            "lgbm_beats_baseline_mae": [True],
            "lgbm_beats_baseline_mape": [True],
        }
    )
    h_table = build_horizon_comparison_table(horizon_df)
    assert h_table.loc[0, "Resultado"] == "Modelo vence"
    assert len(GLOSSARY_ENTRIES) >= 8


def test_streamlit_app_renders_without_exceptions() -> None:
    """Executa a interface Streamlit em modo de teste para validar graficos e tabelas reais."""
    from streamlit.testing.v1 import AppTest

    app_path = get_project_root() / "dashboard" / "app.py"
    at = AppTest.from_file(app_path, default_timeout=20)
    at.run()
    assert not at.exception
    assert at.radio[0].options == ["Previsão", "Validação", "Histórico", "Guia"]
    assert not at.number_input


def test_forecast_age_is_relative_to_origin_not_target_day() -> None:
    """Uma previsao antiga nao e apresentada como se tivesse sido produzida hoje."""
    assert forecast_age_days("2026-11-14", today="2026-11-17") == 3
    assert forecast_age_days("2026-11-17", today="2026-11-17") == 0


def test_dashboard_filters_remain_functional() -> None:
    """Exercita filtros reais em todas as abas sem alterar artefatos locais."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(get_project_root() / "dashboard" / "app.py", default_timeout=20)
    at.run()
    at.radio[0].set_value("Validação do modelo").run()
    at.selectbox[0].select("t+3").run()
    at.radio[0].set_value("Histórico e fatores").run()
    at.selectbox[0].select("2026").run()
    assert not at.exception
