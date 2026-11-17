# Registro de Decisoes Arquiteturais (ADR) — Energy Forecast

Este documento registra as decisoes arquiteturais tomadas na concepcao do projeto **Energy Forecast**, seus contextos, alternativas avaliadas e consequencias tecnicas.

---

## ADR-001: Adocao do DuckDB como Banco Analitico Local

- **Contexto:** O projeto requer processamento analitico de series temporais multi-anuais (2018 ao presente), integracao nativa com arquivos Apache Parquet e compatibilidade com dbt Core sem depender de infraestrutura em nuvem paga ou servidores de banco de dados pesados.
- **Decisao:** Utilizar **DuckDB** (`data/energy_forecast.duckdb`) como engine OLAP embutida (in-process).
- **Justificativa:**
  - Leitura direta e vetorizada de arquivos Parquet na camada Bronze (`read_parquet`);
  - Execucao local em arquivo unico, facilitando reprodutibilidade em Docker e ambientes locais;
  - Integracao madura e oficial com o ecossistema dbt via `dbt-duckdb` e com `pandas`/`pyarrow` sem sobrecarga de serializacao.

---

## ADR-002: Arquitetura Medallion Hibrida (Python Parquet na Bronze + dbt na Silver/Gold)

- **Contexto:** Dados publicos governamentais podem sofrer alteracoes metodologicas, mudancas de formatacao ou indisponibilidade temporaria na origem. Ao mesmo tempo, transformacoes analiticas se beneficiam de SQL declarativo e testes automatizados de schema.
- **Decisao:**
  - Implementar a **Camada Bronze** em Python (`src/energy_forecast/ingestion/`), baixando os dados originais do ONS e persistindo-os intactos em Parquet com um manifesto JSON separado (`ingestion_manifest.json`);
  - Implementar as **Camadas Silver e Gold** em SQL declarativo via **dbt Core + dbt-duckdb**.
- **Justificativa:**
  - Mantem a camada Bronze 100% fiel ao dado bruto original, permitindo reprocessamento completo das camadas superiores sem novo download;
  - Na Silver, regras de tipagem, deduplicacao e validacao ficam auditaveis sem descartar registros silenciosamente;
  - Na Gold, os testes nativos (`not_null`, `unique`) e testes singulares do dbt garantem o contrato da serie diaria do subsistema `SE` antes que qualquer modelo de ML seja treinado.

---

## ADR-003: Definicao do Baseline Sazonal ($t-7$)

- **Contexto:** A carga eletrica diaria apresenta forte sazonalidade semanal (diferenca estrutural entre dias uteis, sabados e domingos). Qualquer modelo de Machine Learning so se justifica em producao se superar de maneira consistente uma referencia simples que ja capture essa sazonalidade.
- **Decisao:** Adotar como baseline oficial a repeticao do valor observado no mesmo dia da semana imediatamente anterior:
  $$\hat{y}_{\tau}^{\text{baseline}} = y_{\tau - 7}, \quad \forall \tau \in \{t_0 + 1, \dots, t_0 + 7\}$$
- **Justificativa:**
  - Para todo horizonte de 1 a 7 dias a partir da origem $t_0$, o instante $\tau - 7 = t_0 + h - 7 \le t_0$ ja foi observado na data de origem;
  - Elimina qualquer risco de vazamento de dados futuros e fornece um benchmark realista para series de energia eletrica.

---

## ADR-004: Engenharia de Atributos Livre de Vazamento Temporal para Horizonte de 7 Dias

- **Contexto:** Ao prever 7 dias a frente ($h = 1 \dots 7$), o uso ingenuo de lags curtos (como $y_{\tau - 1}$) exigiria valores futuros nao observados na origem $t_0$ para $h \ge 2$ ou propagacao recursiva de erros.
- **Decisao:** Estruturar os lags iniciais em **7, 14 e 28 dias** em relacao a data alvo $\tau = t_0 + h$, e calcular janelas moveis (7, 14 e 28 dias) ancoradas exclusivamente em observacoes $\le t_0$ (ou $\le \tau - 7$).
- **Justificativa:**
  - Como $h \in \{1, \dots, 7\}$, qualquer lag $L \ge 7$ relativo a data alvo $\tau$ utiliza a observacao em $\tau - L = t_0 + h - L \le t_0$, que e um dado real consolidado no momento da previsao;
  - Permite previsao multi-horizonte direta/tabular com o LightGBM sem vazamento temporal e sem acumulo de erro recursivo.

---

## ADR-005: Validacao por Backtesting Temporal (Expanding Window) em vez de Divisao Aleatoria

- **Contexto:** Series temporais possuem dependencia cronologica, tendencia e sazonalidade. Divisoes aleatorias (`train_test_split` aleatorio ou K-Fold convencional) causam vazamento temporal severo e produzem metricas artificialmente otimistas.
- **Decisao:** Proibir divisoes aleatorias e utilizar exclusivamente **validacao temporal com origens de previsao sucessivas** (walk-forward / expanding window).
- **Justificativa:**
  - Simula fielmente a operacao real do sistema: em cada data de origem $t_0$, o modelo conhece apenas o passado ($\le t_0$) e preve os proximos 7 dias ($t_0 + 1 \dots t_0 + 7$);
  - Permite avaliar a estabilidade do modelo em diferentes epocas do ano e quantificar a degradacao de acuracia do dia $t+1$ ate o dia $t+7$.

---

## ADR-006: Configuracao Centralizada e Desacoplamento do Dashboard

- **Contexto:** Caminhos hardcoded e acoplamento entre codigo de interface grafica (Streamlit) e codigo de pipeline dificultam testes automatizados e execucao em servidores headless/containers.
- **Decisao:**
  - Centralizar todos os caminhos e hiperparametros estruturais em `config/settings.yaml`, carregados por dataclasses imutaveis em `src/energy_forecast/utils/config.py` com suporte a sobrescrita por variaveis de ambiente (`.env`);
  - Isolar o Streamlit em `dashboard/app.py` e no grupo de dependencias opcionais `[project.optional-dependencies.dashboard]`, mantendo o core (`src/energy_forecast/`) livre de qualquer dependencia de UI.
- **Justificativa:**
  - Permite que os testes (`pytest`) utilizem diretorios temporarios isolados sobrescrevendo apenas `root_dir` ou `config_path`;
  - Garante que o pipeline principal (ingestao, dbt, treinamento e avaliacao) execute de forma enxuta e independente de componentes de visualizacao.

