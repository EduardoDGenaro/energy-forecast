"""Painel analitico opcional em Streamlit, com linguagem acessivel para operacao e gestao."""

from __future__ import annotations

import json
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from energy_forecast.evaluation.diagnostics import error_diagnostics, monthly_errors
from energy_forecast.evaluation.metrics import EvaluationError
from energy_forecast.evaluation.risk import RiskPolicy, add_prediction_intervals, load_risk_policy
from energy_forecast.utils.config import load_settings

WEEKDAY_NAMES_PT: dict[int, str] = {
    0: "Segunda-feira",
    1: "Terca-feira",
    2: "Quarta-feira",
    3: "Quinta-feira",
    4: "Sexta-feira",
    5: "Sabado",
    6: "Domingo",
}

FEATURE_LABELS_PT: dict[str, str] = {
    "horizon": "Dias de antecedencia (1 a 7)",
    "day_of_week": "Dia da semana",
    "day_of_month": "Dia do mes",
    "day_of_year": "Dia do ano (1 a 366)",
    "month": "Mes do ano",
    "week_of_year": "Semana do ano",
    "is_weekend": "Indicador de fim de semana",
    "is_holiday_br": "Feriado nacional brasileiro",
    "is_day_before_holiday_br": "Vespera de feriado nacional",
    "is_day_after_holiday_br": "Dia seguinte a feriado nacional",
    "is_holiday_lag_7": "Semana passada foi feriado (7 dias atras)",
    "day_of_year_sin": "Ciclo sazonal anual (onda seno)",
    "day_of_year_cos": "Ciclo sazonal anual (onda cosseno)",
    "lag_7": "Consumo na semana passada (7 dias atras)",
    "lag_14": "Consumo ha 2 semanas (14 dias atras)",
    "lag_28": "Consumo ha 4 semanas (28 dias atras)",
    "lag_7_diff_14": "Tendencia recente (7 dias vs. 14 dias atras)",
    "lag_7_ratio_14": "Variacao proporcional (7 dias vs. 14 dias)",
    "origin_load": "Ultimo consumo real registrado antes da previsao",
    "rolling_mean_7": "Media de consumo dos ultimos 7 dias",
    "rolling_mean_14": "Media de consumo dos ultimos 14 dias",
    "rolling_mean_28": "Media de consumo dos ultimos 28 dias",
    "rolling_std_7": "Oscilacao do consumo nos ultimos 7 dias",
    "rolling_std_14": "Oscilacao do consumo nos ultimos 14 dias",
    "rolling_std_28": "Oscilacao do consumo nos ultimos 28 dias",
    "lag_7_vs_rolling_mean_7": "Semana passada comparada a media recente",
}

GLOSSARY_ENTRIES: tuple[dict[str, str], ...] = (
    {
        "Termo no painel": "MWmed (Megawatt medio)",
        "O que significa": "Potencia eletrica media consumida ao longo das 24 horas de um dia.",
        "Por que e usado aqui": (
            "E a unidade oficial usada pelo ONS para medir a carga diaria do sistema eletrico "
            "sem distorcoes de picos instantaneos de alguns minutos."
        ),
    },
    {
        "Termo no painel": "Regiao Sudeste / Centro-Oeste (Codigo SE)",
        "O que significa": "Subsistema eletrico que abrange Sudeste e Centro-Oeste do Brasil.",
        "Por que e usado aqui": (
            "E o recorte definido para este projeto por concentrar a maior parcela da demanda "
            "nacional de energia eletrica."
        ),
    },
    {
        "Termo no painel": "Regra Simples (Semana Passada / Baseline)",
        "O que significa": (
            "Previsao ingenua que apenas repete o consumo observado exatamente 7 dias antes "
            "(mesmo dia da semana anterior)."
        ),
        "Por que e usado aqui": (
            "Serve como referencia minima de comparacao. Como o consumo segue ritmo semanal, um "
            "modelo avancado so se justifica se conseguir errar menos do que essa regra simples."
        ),
    },
    {
        "Termo no painel": "Modelo Inteligente (LightGBM)",
        "O que significa": (
            "Algoritmo de aprendizado de maquina baseado em arvores de decisao que cruza "
            "calendario, feriados, epoca do ano e historico recente."
        ),
        "Por que e usado aqui": (
            "Captura nuances que a regra simples nao enxerga, como feriados, recuperacao de "
            "carga pos-feriado e mudancas recentes de patamar de consumo."
        ),
    },
    {
        "Termo no painel": "Erro Medio em Energia (MAE)",
        "O que significa": "Diferenca media, em MWmed, entre o consumo previsto e o consumo real.",
        "Por que e usado aqui": (
            "Traduz a precisao para a realidade operacional: mostra diretamente quantos MWmed "
            "de energia ficam na margem de erro diaria."
        ),
    },
    {
        "Termo no painel": "Erro Percentual Medio (MAPE)",
        "O que significa": "Tamanho medio do erro expresso em porcentagem (%) do consumo do dia.",
        "Por que e usado aqui": (
            "Permite comparar a qualidade da previsao em epocas diferentes do ano, seja no "
            "verao (carga mais alta) ou no inverno (carga mais amena)."
        ),
    },
    {
        "Termo no painel": "Dias a frente (Antecedencia / Horizonte t+1 a t+7)",
        "O que significa": (
            "Quantos dias no futuro o sistema esta prevendo a partir da ultima medicao real "
            "(1 dia a frente = amanha; 7 dias a frente = daqui a uma semana)."
        ),
        "Por que e usado aqui": (
            "Prever o dia seguinte tende a ser mais facil do que prever a semana seguinte. "
            "Separar por antecedencia revela se o modelo sustenta vantagem nos 7 dias."
        ),
    },
    {
        "Termo no painel": "Teste Historico Semanal (Backtesting)",
        "O que significa": (
            "Simulacao em que o modelo viaja para datas do passado (52 semanas), treina apenas "
            "com dados anteriores e preve os 7 dias seguintes."
        ),
        "Por que e usado aqui": (
            "Garante avaliacao honesta sem usar informacao futura (sem vazamento de dados), "
            "reproduzindo fielmente como o sistema operaria no dia a dia."
        ),
    },
    {
        "Termo no painel": "Base Certificada (Camada Gold)",
        "O que significa": (
            "Serie historica do ONS ja limpa, padronizada e verificada quanto a datas faltantes "
            "ou valores invalidos."
        ),
        "Por que e usado aqui": (
            "Assegura que tanto o modelo quanto os graficos deste painel utilizem exatamente os "
            "mesmos dados diarios consistentes desde 2018."
        ),
    },
)

COLOR_ACTUAL = "#98C9F3"
COLOR_BASELINE = "#EDB64D"
COLOR_LGBM = "#2DD4BF"


def format_pt_number(value: float, decimals: int = 0, suffix: str = "") -> str:
    """Formata numeros no padrao brasileiro (ponto para milhar e virgula decimal)."""
    formatted = f"{value:,.{decimals}f}"
    pt_formatted = formatted.replace(",", "_").replace(".", ",").replace("_", ".")
    return f"{pt_formatted}{suffix}"


def format_error_delta(improvement_pct: float, suffix: str) -> str:
    """Formata a variacao percentual do erro para st.metric com delta_color='inverse'."""
    error_change_pct = -improvement_pct
    sign = "+" if error_change_pct > 0 else ""
    return f"{sign}{format_pt_number(error_change_pct, decimals=1)}%{suffix}"


def format_horizon_label(horizon: int) -> str:
    """Converte o numero do horizonte (1..7) em texto claro em portugues."""
    if horizon == 1:
        return "1 dia a frente"
    return f"{horizon} dias a frente"


def translate_feature_name(feature_name: str) -> str:
    """Traduz nomes tecnicos de atributos do modelo para linguagem de negocio."""
    return FEATURE_LABELS_PT.get(feature_name, feature_name)


def build_operational_forecast_table(latest_df: pd.DataFrame) -> pd.DataFrame:
    """Prepara a tabela da previsao dos proximos 7 dias com colunas e numeros legiveis."""
    df = latest_df.copy()
    target_dates = pd.to_datetime(df["target_date"])
    horizons = df["horizon"].astype(int)

    return pd.DataFrame(
        {
            "Data": target_dates.dt.strftime("%d/%m/%Y"),
            "Dia da semana": target_dates.dt.day_of_week.map(WEEKDAY_NAMES_PT),
            "Antecedencia": [format_horizon_label(h) for h in horizons],
            "Modelo Inteligente (MWmed)": [
                format_pt_number(v, decimals=0) for v in df["y_pred_lgbm"]
            ],
            "Regra Simples (MWmed)": [
                format_pt_number(v, decimals=0) for v in df["y_pred_baseline"]
            ],
            "Ajuste do Modelo vs. Regra Simples": [
                f"{'+' if v >= 0 else ''}{format_pt_number(v, decimals=0)} MWmed"
                for v in (df["y_pred_lgbm"] - df["y_pred_baseline"])
            ],
        }
    )


def build_horizon_comparison_table(horizon_df: pd.DataFrame) -> pd.DataFrame:
    """Prepara a tabela comparativa por dia de antecedencia em linguagem acessivel."""
    rows: list[dict[str, str]] = []
    for _, row in horizon_df.iterrows():
        h = int(row["horizon"])
        beats_both = bool(row["lgbm_beats_baseline_mae"]) and bool(row["lgbm_beats_baseline_mape"])
        rows.append(
            {
                "Antecedencia": format_horizon_label(h),
                "Erro Regra Simples (MWmed)": format_pt_number(
                    float(row["mae_baseline"]), decimals=0
                ),
                "Erro Modelo (MWmed)": format_pt_number(float(row["mae_lgbm"]), decimals=0),
                "Reducao do Erro (%)": f"{format_pt_number(float(row['mae_improvement_pct']), 1)}%",
                "Erro % Regra Simples": f"{format_pt_number(float(row['mape_baseline']), 2)}%",
                "Erro % Modelo": f"{format_pt_number(float(row['mape_lgbm']), 2)}%",
                "Resultado": (
                    "Modelo vence"
                    if beats_both
                    else "Regra simples vence"
                    if (
                        row["mae_baseline"] < row["mae_lgbm"]
                        and row["mape_baseline"] < row["mape_lgbm"]
                    )
                    else "Empate ou resultado misto"
                ),
            }
        )
    return pd.DataFrame(rows)


def _apply_custom_styling() -> None:
    """Tokens visuais locais; sem fontes, scripts ou dependencias de rede."""
    stylesheet = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{stylesheet}</style>", unsafe_allow_html=True)


def _render_sidebar() -> str:
    """Navegacao visual das mesmas quatro secoes do painel."""
    with st.sidebar:
        st.markdown(
            '<div class="ef-brand"><svg viewBox="0 0 32 48" aria-hidden="true">'
            '<path fill="#39D2BC" d="M20 1 2 27h12L10 47l21-29H18Z"/>'
            "</svg><span>Energy<br>Forecast</span></div>",
            unsafe_allow_html=True,
        )
        names = {
            "Previsão de 7 dias": "Previsão",
            "Validação do modelo": "Validação",
            "Histórico e fatores": "Histórico",
            "Guia de leitura": "Guia",
        }
        selected = st.radio(
            "Navegação",
            list(names),
            format_func=names.get,
            label_visibility="collapsed",
            key="dashboard_navigation",
        )
        st.markdown(
            '<div class="ef-sidebar-source">ONS · Subsistema SE<br>Sudeste / Centro-Oeste</div>',
            unsafe_allow_html=True,
        )
    return selected


@st.cache_data(show_spinner=False)
def _read_parquet(path: str, modified_ns: int) -> pd.DataFrame:
    """Invalida o cache quando o arquivo muda no pipeline."""
    return pd.read_parquet(path)


def _load_parquet(path: Path) -> pd.DataFrame:
    return _read_parquet(str(path), path.stat().st_mtime_ns)


def _style_chart(fig: go.Figure, height: int = 360) -> go.Figure:
    """Uma linguagem visual e legenda curta para todos os graficos."""
    fig.update_layout(
        template="plotly_dark",
        font={"family": "Arial, sans-serif", "size": 12, "color": "#D5DDE5"},
        paper_bgcolor="#242C34",
        plot_bgcolor="#242C34",
        colorway=[COLOR_LGBM, COLOR_BASELINE, COLOR_ACTUAL],
        separators=",.",
        hovermode="x unified",
        height=height,
        margin={"l": 12, "r": 12, "t": 12, "b": 12},
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.16,
            "xanchor": "center",
            "x": 0.5,
            "traceorder": "normal",
        },
        hoverlabel={"bgcolor": "#303B46", "font_color": "#F1F4F7"},
    )
    fig.update_xaxes(title=None, showgrid=False, zeroline=False, automargin=True)
    fig.update_yaxes(
        title=None,
        gridcolor="#3C4752",
        zeroline=False,
        automargin=True,
        tickformat=",.0f",
    )
    return fig


def _plot(fig: go.Figure) -> None:
    st.plotly_chart(
        fig,
        width="stretch",
        theme=None,
        config={"displayModeBar": False, "scrollZoom": False},
    )


def forecast_age_days(origin: Any, today: Any | None = None) -> int:
    """Mede a defasagem da origem usando o fuso brasileiro, sem chamar t+1 de amanha."""
    if today is None:
        today = datetime.now(ZoneInfo("America/Sao_Paulo")).date()
    return max(0, (pd.Timestamp(today).normalize() - pd.Timestamp(origin).normalize()).days)


def _render_verdict_banner(summary: dict[str, Any]) -> None:
    beats = bool(summary["lgbm_beats_baseline_mae"]) and bool(summary["lgbm_beats_baseline_mape"])
    style = "verdict-box" if beats else "verdict-box-warning"
    message = (
        "LightGBM supera o baseline no período avaliado."
        if beats
        else "LightGBM não supera o baseline nas duas métricas."
    )
    st.markdown(
        f'<div class="{style}"><strong>{message}</strong> '
        "O resultado histórico não garante o erro de cada previsão futura.</div>",
        unsafe_allow_html=True,
    )


def _render_operational_tab(gold_df: pd.DataFrame, latest_df: pd.DataFrame | None) -> None:
    if latest_df is None or latest_df.empty:
        st.info("Sem previsão operacional disponível. Gere a previsão para consultar os sete dias.")
        with st.expander("Como gerar a previsão"):
            st.code("python scripts/train.py", language="bash")
        return

    forecast = latest_df.sort_values("target_date").copy()
    origin = pd.Timestamp(forecast["origin_date"].iloc[0])
    first = forecast["target_date"].min().strftime("%d/%m")
    last = forecast["target_date"].max().strftime("%d/%m/%Y")
    history = gold_df.loc[gold_df["data_referencia"] <= origin].tail(21)
    if history.empty:
        st.warning("A origem da previsão não possui histórico correspondente na base.")
        return
    with st.container(border=True, key="forecast_chart"):
        st.header("Demanda diária")
        st.markdown(
            f'<div class="forecast-window"><span>Janela prevista '
            f"<strong>{first} a {last}</strong></span>"
            f"<span>Histórico até <strong>{origin:%d/%m/%Y}</strong> · MWmed</span></div>",
            unsafe_allow_html=True,
        )
        last_value = float(history["carga_mwmed"].iloc[-1])
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=history["data_referencia"],
                y=history["carga_mwmed"],
                mode="lines+markers",
                name="Real ONS",
                line={"color": COLOR_ACTUAL, "width": 2.5},
                marker={"size": 4},
            )
        )
        for col, name, color, dash in (
            ("y_pred_baseline", "Baseline", COLOR_BASELINE, "dot"),
            ("y_pred_lgbm", "LightGBM", COLOR_LGBM, "solid"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=[origin, *forecast["target_date"]],
                    y=[last_value, *forecast[col]],
                    mode="lines+markers",
                    name=name,
                    line={"color": color, "width": 2.5, "dash": dash},
                    marker={"size": 5},
                    hovertemplate="%{y:,.0f} MWmed<extra>%{fullData.name}</extra>",
                )
            )
        if {"lower_bound", "upper_bound"}.issubset(forecast.columns):
            valid = forecast.dropna(subset=["lower_bound", "upper_bound"])
            if not valid.empty:
                fig.add_trace(
                    go.Scatter(
                        x=valid["target_date"],
                        y=valid["upper_bound"],
                        mode="lines",
                        line={"width": 0},
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=valid["target_date"],
                        y=valid["lower_bound"],
                        mode="lines",
                        line={"width": 0},
                        fill="tonexty",
                        fillcolor="rgba(45,212,191,0.16)",
                        name="Faixa nominal 90%",
                        hoverinfo="skip",
                    )
                )
        fig.add_vrect(
            x0=origin,
            x1=forecast["target_date"].max(),
            fillcolor="#29423F",
            opacity=0.25,
            line_width=0,
            layer="below",
        )
        fig.add_vline(x=origin, line_color="#DCE5ED", line_dash="dash", line_width=1)
        _style_chart(fig, height=300)
        fig.update_xaxes(tickformat="%d/%m")
        _plot(fig)
        peak = forecast.loc[forecast["y_pred_lgbm"].idxmax()]
        st.caption(
            f"Área sombreada: janela prevista · Maior carga prevista em "
            f"{peak['target_date']:%d/%m}: "
            f"{format_pt_number(float(peak['y_pred_lgbm']))} MWmed. "
            "O fundo delimita a previsão; a faixa acompanha a incerteza empírica. "
            "Meta nominal de 90%, sem garantia de cobertura futura."
        )

    with st.container(border=True, key="daily_table"):
        with st.container(key="daily_table_heading"):
            table_heading, table_action = st.columns([3, 1], vertical_alignment="center")
            with table_heading:
                st.header("Previsão por dia")
            with table_action, st.container(key="export_forecast"):
                st.download_button(
                    "Baixar CSV",
                    forecast.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"previsao_se_{origin:%Y-%m-%d}.csv",
                    mime="text/csv",
                )
        st.caption("Compare os modelos e consulte a diferença em MWmed para planejar a operação.")
        st.dataframe(
            build_operational_forecast_table(forecast),
            hide_index=True,
            width="stretch",
            column_config={
                "Dia da semana": "Dia",
                "Antecedencia": "Horizonte",
                "Modelo Inteligente (MWmed)": "LightGBM (MWmed)",
                "Regra Simples (MWmed)": "Baseline (MWmed)",
                "Ajuste do Modelo vs. Regra Simples": "Diferença (MWmed)",
            },
        )
        if {"lower_bound", "upper_bound"}.issubset(forecast.columns):
            st.dataframe(
                forecast[["target_date", "lower_bound", "upper_bound", "calibration_n"]],
                hide_index=True,
                width="stretch",
                column_config={
                    "target_date": st.column_config.DateColumn("Data", format="DD/MM/YYYY"),
                    "lower_bound": st.column_config.NumberColumn(
                        "Limite inferior (MWmed)", format="%.0f"
                    ),
                    "upper_bound": st.column_config.NumberColumn(
                        "Limite superior (MWmed)", format="%.0f"
                    ),
                    "calibration_n": "Amostras de calibração",
                },
            )


def _render_backtest_tab(
    bt_preds: pd.DataFrame, horizon_df: pd.DataFrame, diagnostics: dict, policy: RiskPolicy
) -> None:
    _render_verdict_banner(
        {
            "lgbm_beats_baseline_mae": diagnostics["mae_lgbm"] < diagnostics["mae_baseline"],
            "lgbm_beats_baseline_mape": diagnostics["mape_lgbm"] < diagnostics["mape_baseline"],
        }
    )
    st.header("Realizado e previsto")
    selection = st.selectbox(
        "Antecedência da previsão", ["Todos os horizontes", *[f"t+{h}" for h in range(1, 8)]]
    )
    selected = (
        bt_preds
        if selection == "Todos os horizontes"
        else bt_preds.loc[bt_preds["horizon"] == int(selection[2:])]
    )
    selected = selected.sort_values("target_date")
    with st.container(border=True, key="validation_chart"):
        fig = go.Figure()
        for col, name, color, dash in (
            ("y_true", "Real ONS", COLOR_ACTUAL, "solid"),
            ("y_pred_baseline", "Baseline", COLOR_BASELINE, "dot"),
            ("y_pred_lgbm", "LightGBM", COLOR_LGBM, "solid"),
        ):
            fig.add_trace(
                go.Scatter(
                    x=selected["target_date"],
                    y=selected[col],
                    mode="lines",
                    name=name,
                    line={"color": color, "width": 1.8, "dash": dash},
                )
            )
        _style_chart(fig)
        fig.update_xaxes(tickformat="%m/%Y")
        _plot(fig)
        st.caption("Carga média diária (MWmed). Cada previsão utiliza apenas dados até sua origem.")

    st.header("Onde o modelo acerta menos")
    st.caption(
        "MAPE é uma média dos erros percentuais; não é margem de confiança nem garantia diária."
    )
    cols = st.columns(3)
    cols[0].metric(
        "Erro em 90% das previsões", f"{format_pt_number(diagnostics['p90_ape_pct'], 2)}%"
    )
    cols[1].metric("Maior erro observado", f"{format_pt_number(diagnostics['max_ape_pct'], 2)}%")
    cols[2].metric(
        "Dias com erro acima de 10%", f"{diagnostics['days_above_10pct']} / {len(bt_preds)}"
    )
    tolerance = bt_preds["horizon"].map(policy.daily_error_limit_pct)
    ape = (bt_preds["y_true"] - bt_preds["y_pred_lgbm"]).abs() / bt_preds["y_true"] * 100
    within = int((ape <= tolerance).sum())
    st.caption(
        f"{within} de {len(ape)} previsões ({format_pt_number(within / len(ape) * 100, 1)}%) "
        "ficaram dentro da referência diária de erro do seu horizonte."
    )
    st.caption(
        "Referência padrão: erro diário de até 5% por horizonte. "
        "Peso 2 para subestimar a carga e 1 para superestimá-la. "
        "São critérios orientativos, sem garantia de erro futuro."
    )
    st.markdown("### Erro por mês")
    monthly = monthly_errors(bt_preds)
    with st.container(border=True, key="monthly_chart"):
        fig = go.Figure()
        for col, name, color in (
            ("mape_baseline", "Baseline", COLOR_BASELINE),
            ("mape_lgbm", "LightGBM", COLOR_LGBM),
        ):
            fig.add_trace(
                go.Bar(
                    x=monthly["month"],
                    y=monthly[col],
                    name=name,
                    marker_color=color,
                    customdata=monthly[["count"]],
                    hovertemplate="%{x}<br>MAPE: %{y:.2f}%<br>N: %{customdata[0]}<extra></extra>",
                )
            )
        _style_chart(fig, 290)
        fig.update_layout(barmode="group")
        fig.update_yaxes(ticksuffix="%", tickformat=".1f")
        _plot(fig)
        st.caption("Meses de início e fim podem estar incompletos. Consulte N ao passar o cursor.")

    with st.expander("Desempenho por horizonte · t+1 a t+7", expanded=True):
        st.dataframe(build_horizon_comparison_table(horizon_df), hide_index=True, width="stretch")
    with st.expander("Dias com os maiores erros"):
        worst = bt_preds.assign(erro_pct=ape).nlargest(10, "erro_pct")
        st.dataframe(
            worst[["target_date", "horizon", "y_true", "y_pred_lgbm", "erro_pct"]],
            hide_index=True,
            width="stretch",
            column_config={
                "target_date": st.column_config.DateColumn("Data", format="DD/MM/YYYY"),
                "horizon": "Horizonte",
                "y_true": st.column_config.NumberColumn("Real (MWmed)", format="%.0f"),
                "y_pred_lgbm": st.column_config.NumberColumn("LightGBM (MWmed)", format="%.0f"),
                "erro_pct": st.column_config.NumberColumn("Erro (%)", format="%.2f"),
            },
        )
    st.info(
        "Limites da avaliação: não há intervalo preditivo calibrado nem holdout intocado "
        "documentado. Revisões históricas do ONS e atrasos de publicação não são simulados. "
        "A vantagem agregada não significa que o modelo vença em todos os meses."
    )


def _render_drivers_and_history_tab(
    gold_df: pd.DataFrame, feat_imp_df: pd.DataFrame | None
) -> None:
    st.header("Fatores consultados pelo modelo")
    st.caption(
        "Frequência de uso nas divisões das árvores. Não mede efeito causal nem "
        "a contribuição de cada fator para uma previsão específica."
    )
    with st.container(border=True, key="drivers"):
        if (
            feat_imp_df is not None
            and not feat_imp_df.empty
            and feat_imp_df["importance"].sum() > 0
        ):
            top = feat_imp_df.nlargest(10, "importance").copy()
            top["Fator"] = top["feature"].map(translate_feature_name)
            top["Uso (%)"] = top["importance"] / feat_imp_df["importance"].sum() * 100
            # Wrap actual labels for narrow screens instead of hiding them in clipped ticks.
            import textwrap

            top["Rótulo"] = top["Fator"].map(
                lambda value: "<br>".join(textwrap.wrap(value, width=27))
            )
            fig = px.bar(
                top.sort_values("Uso (%)"),
                x="Uso (%)",
                y="Rótulo",
                orientation="h",
                color_discrete_sequence=[COLOR_LGBM],
                custom_data=["Fator"],
            )
            fig.update_traces(hovertemplate="%{customdata[0]}<br>%{x:.1f}%<extra></extra>")
            _style_chart(fig, 480)
            fig.update_layout(hovermode="closest")
            fig.update_xaxes(ticksuffix="%")
            _plot(fig)
        else:
            st.info(
                "Importância de fatores indisponível. A previsão pode ser consultada normalmente."
            )
    st.header("Histórico da carga")
    st.caption("Série diária do subsistema Sudeste/Centro-Oeste. Fonte: ONS · MWmed.")
    years = sorted(gold_df["data_referencia"].dt.year.unique().tolist(), reverse=True)
    year = st.selectbox("Período do histórico", ["Todo o histórico", *[str(y) for y in years]])
    history = (
        gold_df
        if year == "Todo o histórico"
        else gold_df.loc[gold_df["data_referencia"].dt.year == int(year)]
    )
    with st.container(border=True, key="history"):
        fig = px.line(
            history, x="data_referencia", y="carga_mwmed", color_discrete_sequence=[COLOR_ACTUAL]
        )
        _style_chart(fig)
        fig.update_xaxes(tickformat="%m/%Y")
        _plot(fig)
        with st.expander("Consultar dados do histórico"):
            st.dataframe(
                history[["data_referencia", "carga_mwmed"]], hide_index=True, width="stretch"
            )


def _render_glossary_section() -> None:
    st.header("Guia de leitura")
    st.caption("Conceitos para interpretar a previsão e seus limites.")
    for entry in GLOSSARY_ENTRIES:
        with st.expander(entry["Termo no painel"]):
            st.write(entry["O que significa"])
            st.caption(entry["Por que e usado aqui"])
    st.markdown("**Como ler os erros**")
    st.write(
        "Por exemplo, um MAPE de 3,79% significa erro percentual absoluto médio "
        "no teste histórico. "
        "Alguns dias erram mais, outros menos. Não equivale a 96,21% de acurácia "
        "nem a um intervalo de confiança de ±3,79%."
    )


def _prepare_dashboard_data(
    gold: pd.DataFrame, predictions: pd.DataFrame, summary: dict
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Recusa estados incoerentes em vez de mostrar indicadores desatualizados."""
    gold = gold.copy()
    predictions = predictions.copy()
    gold["data_referencia"] = pd.to_datetime(gold["data_referencia"])
    for col in ("origin_date", "target_date"):
        predictions[col] = pd.to_datetime(predictions[col])
    if gold.empty or gold["data_referencia"].duplicated().any():
        raise ValueError("Histórico vazio ou com datas duplicadas.")
    diagnostics = error_diagnostics(predictions)
    for key in ("mae_lgbm", "mape_lgbm", "mae_baseline", "mape_baseline"):
        if not np.isclose(diagnostics[key], float(summary["summary"][key])):
            raise ValueError("As métricas salvas não correspondem às previsões disponíveis.")
    actuals = predictions["target_date"].map(gold.set_index("data_referencia")["carga_mwmed"])
    if not np.allclose(actuals, predictions["y_true"]):
        raise ValueError("O teste histórico não corresponde à versão atual da base.")
    gold = gold.sort_values("data_referencia")
    return gold, predictions, diagnostics


def render_app() -> None:
    """Painel operacional com verificacao de integridade e limites visiveis."""
    st.set_page_config(page_title="Energy Forecast · Carga elétrica", layout="wide")
    _apply_custom_styling()
    selected_page = _render_sidebar()
    with st.container(key="page_heading"):
        heading, dates = st.columns([1.8, 1])
        with heading:
            st.title("Previsão de carga elétrica")
            st.markdown(
                '<p class="source-line">Sudeste / Centro-Oeste · Demanda média diária '
                "· Dados públicos ONS</p>",
                unsafe_allow_html=True,
            )
        with dates:
            date_location = st.empty()
    project_root = Path(__file__).resolve().parents[1]
    settings = load_settings(root_dir=project_root)
    summary_path = settings.paths.artifacts_dir / "backtest_summary.json"
    predictions_path = settings.paths.predictions_dir / "backtest_predictions.parquet"
    gold_path = settings.paths.gold_dir / "fct_carga_diaria_se.parquet"
    if not all(p.is_file() for p in (summary_path, predictions_path, gold_path)):
        st.info("O painel precisa da base histórica e dos resultados da avaliação para começar.")
        with st.expander("Preparar os dados"):
            st.code("python scripts/run_dbt.py --command build\npython scripts/evaluate.py")
        return
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        gold, predictions, diagnostics = _prepare_dashboard_data(
            _load_parquet(gold_path), _load_parquet(predictions_path), summary
        )
        horizon = pd.DataFrame(summary["horizon_metrics"])
        if horizon.empty:
            raise ValueError("Resultados por horizonte ausentes.")
    except (OSError, ValueError, KeyError, TypeError, EvaluationError) as exc:
        st.error(f"Não foi possível validar os dados do painel: {exc}")
        st.caption("Atualize a base e refaça a avaliação para reconstruir os resultados.")
        return
    latest_path = settings.paths.predictions_dir / "latest_forecast_7d.parquet"
    latest = None
    if latest_path.is_file():
        try:
            latest = _load_parquet(latest_path).copy()
            for col in ("origin_date", "target_date"):
                latest[col] = pd.to_datetime(latest[col])
            if (
                latest["origin_date"].nunique() != 1
                or sorted(latest["horizon"].tolist()) != list(range(1, 8))
                or not np.isfinite(latest[["y_pred_lgbm", "y_pred_baseline"]]).all().all()
                or not (latest["target_date"] - latest["origin_date"])
                .dt.days.eq(latest["horizon"])
                .all()
            ):
                raise ValueError("Previsão incompleta ou inconsistente.")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            st.warning(f"Previsão operacional indisponível: {exc}")
            latest = None
    end_date = latest["target_date"].max() if latest is not None else gold["data_referencia"].max()
    start_date = gold["data_referencia"].tail(21).min()
    date_location.markdown(
        '<div class="date-window"><svg viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="1.5" aria-hidden="true">'
        '<rect x="3" y="5" width="18" height="16" rx="2"/>'
        '<path d="M3 10h18M8 2v6m8-6v6"/></svg>'
        f"<span>{start_date:%d/%m/%Y} – {end_date:%d/%m/%Y}</span></div>"
        f'<div class="base-date">Base até {gold["data_referencia"].max():%d/%m/%Y}</div>',
        unsafe_allow_html=True,
    )
    importance = None
    importance_path = settings.paths.artifacts_dir / "feature_importance.csv"
    if importance_path.is_file():
        try:
            importance = pd.read_csv(importance_path)
            if not {"feature", "importance"}.issubset(importance.columns):
                importance = None
        except (OSError, ValueError):
            importance = None

    with st.container(key="main_metrics"):
        cols = st.columns(4)
    cols[0].metric(
        "MAPE · LightGBM",
        f"{format_pt_number(diagnostics['mape_lgbm'], 2)}%",
        help="Erro percentual absoluto médio no teste histórico. Menor é melhor.",
    )
    cols[1].metric(
        "MAE · MWmed",
        format_pt_number(diagnostics["mae_lgbm"]),
        help="Erro absoluto médio em MWmed no teste histórico.",
    )
    improvement = (
        (diagnostics["mae_baseline"] - diagnostics["mae_lgbm"]) / diagnostics["mae_baseline"] * 100
        if diagnostics["mae_baseline"] > 0
        else None
    )
    cols[2].metric(
        "Redução do MAE",
        f"{format_pt_number(improvement, 1)}%" if improvement is not None else "Indisponível",
        help="Ganho relativo no erro absoluto médio. Negativo significa piora.",
    )
    cols[3].metric(
        "Semanas vencidas",
        f"{diagnostics['winning_origins_mae']} / {diagnostics['origins']}",
        help="Origens em que o LightGBM teve menor MAE que o baseline sazonal.",
    )
    first = predictions["target_date"].min()
    last = predictions["target_date"].max()
    validation_context = (
        f"Validação histórica · {first:%d/%m/%Y} a {last:%d/%m/%Y} · "
        f"{len(predictions)} previsões · MAE em MWmed · "
        f"Baseline: MAPE {format_pt_number(diagnostics['mape_baseline'], 2)}% / "
        f"MAE {format_pt_number(diagnostics['mae_baseline'])} MWmed"
    )
    policy = load_risk_policy(settings.paths.root_dir)
    if latest is not None:
        interval_cols = [
            c
            for c in (
                "lower_bound",
                "upper_bound",
                "calibration_n",
                "nominal_coverage",
                "calibration_last_target",
                "interval_status",
            )
            if c in latest
        ]
        latest = add_prediction_intervals(latest.drop(columns=interval_cols), predictions, policy)
    if selected_page == "Previsão de 7 dias":
        _render_operational_tab(gold, latest)
    elif selected_page == "Validação do modelo":
        _render_backtest_tab(predictions, horizon, diagnostics, policy)
    elif selected_page == "Histórico e fatores":
        _render_drivers_and_history_tab(gold, importance)
    else:
        _render_glossary_section()
    st.divider()
    st.caption(validation_context)
    model = escape(str(summary.get("registered_model_name", "LightGBM")))
    version = escape(str(summary.get("model_version", "—")))
    st.caption(
        f"ONS · Gold até {gold['data_referencia'].max():%d/%m/%Y} · "
        f"Registro MLflow: {model} v{version}"
    )


if __name__ == "__main__":
    render_app()
