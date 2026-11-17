# Energy Forecast — Previsao de Carga Eletrica Diaria (ONS)

Documentação revisada em **17/11/2026**, conferida com os scripts, configurações e
artefatos locais. Os números apresentados são de uma avaliação histórica específica;
novas execuções podem produzir resultados e versões de modelo diferentes.

Revisão de 17/11/2026: [verificação técnica, limites dos modelos e melhorias do dashboard](docs/project_review_2026-11-17.md).
Para recalcular os diagnósticos e reproduzir as previsões históricas sem alterar o MLflow:
`python scripts/audit_project.py --retrain`.

Faixas de previsão, monitoramento, limites de erro e experimentos com calendário e
temperatura: [implementação e resultados](docs/risk_and_candidates.md).
Execute `python scripts/evaluate_risk.py` para persistir os diagnósticos de risco.

## 1. Descricao do Projeto

O **Energy Forecast** e um projeto de Engenharia de Dados, Machine Learning e MLOps voltado a previsao de carga eletrica diaria no Sistema Interligado Nacional (SIN), utilizando dados publicos oficiais disponibilizados pelo Operador Nacional do Sistema Eletrico (ONS).

A solucao implementa uma arquitetura Medallion local e reproduzivel (**Bronze**, **Silver** e **Gold**) sobre arquivos Parquet, **DuckDB** e **dbt Core**, integrada a um pipeline de modelagem preditiva e backtesting temporal com **LightGBM** e rastreamento de experimentos via **MLflow**.

## 2. Objetivo e Problema de Negocio

O planejamento da operacao energetica de curto prazo exige estimativas precisas da demanda diaria de energia para despacho termico e hidreletrico, gestao de reservatorios e seguranca operativa.

O objetivo central deste projeto e **prever a carga diaria de energia (em MWmed) do subsistema Sudeste/Centro-Oeste (codigo `SE`) para um horizonte de 7 dias** ($t+1$ a $t+7$), comparando rigorosamente duas abordagens sob validacao temporal por origens sucessivas (walk-forward backtesting):

1. **Baseline Sazonal (Seasonal Naive):** previsao para o dia $t+h$ ($h \in \{1, \dots, 7\}$) igual ao valor real observado no mesmo dia da semana anterior ($t+h-7$), utilizando estritamente dados observados ate a data de origem $t$.
2. **Modelo de Machine Learning (LightGBM):** modelo baseado em arvores de decisao com gradient boosting utilizando variaveis de calendario (dia da semana, mes, fim de semana, feriados nacionais brasileiros), defasagens temporais (lags de 7, 14 e 28 dias respeitando o horizonte para evitar vazamento temporal) e estatisticas de janelas moveis calculadas exclusivamente com informacoes disponiveis ate a data de origem da previsao.

O criterio de sucesso consiste em verificar se o modelo LightGBM supera o baseline sazonal e quantificar a diferenca de desempenho por meio das metricas **MAE** (Mean Absolute Error, em MWmed) e **MAPE** (Mean Absolute Percentage Error, em %), tanto de forma agregada quanto estratificada por horizonte ($h=1 \dots 7$).

## 3. Fonte Oficial dos Dados

- **Organizacao:** Operador Nacional do Sistema Eletrico (ONS)
- **Portal de Dados Abertos:** https://dados.ons.org.br/dataset/carga-energia
- **Bucket S3 Publico:** `s3://ons-aws-prod-opendata/dataset/carga_energia_di/`
- **Endpoint HTTPS do Bucket:** `https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/carga_energia_di/`
- **Recorte Temporal da ingestão:** Janeiro de 2018 até a data disponível na fonte no momento da execução.
- **Recorte da base local utilizada nos resultados abaixo:** 01/01/2018 a 14/11/2026, com 3.240 dias do subsistema SE. Isso não implica atualização automática do projeto.
- **Escopo Geografico/Eletrico:** Subsistema Sudeste/Centro-Oeste (`id_subsistema = 'SE'`).
- **Unidade de Medida:** `MWmed` (Megawatt medio diario).

Detalhes adicionais sobre o schema e as regras de verificacao pre-ingestao encontram-se em [`docs/data_dictionary.md`](docs/data_dictionary.md).

## 4. Arquitetura Implementada

O projeto adota a arquitetura Medallion combinada com um ciclo estruturado de MLOps:

- **Camada Bronze (`data/bronze/`):** ingestão idempotente dos arquivos anuais do ONS, preservando os bytes e o schema do arquivo baixado. O manifesto (`ingestion_manifest.json`) registra origem, hash, período e timestamps. Arquivos alterados na fonte podem ser baixados novamente e substituir a cópia anual local; a Bronze não mantém todas as versões históricas da fonte.
- **Camada Silver (`dbt/models/silver/`):** Padronizacao de nomes e tipos de dados (datas e numericos), tratamento explicito de nulos ou inconsistencias documentadas, verificacao de duplicidades e preservacao de rastreabilidade ate os arquivos da camada Bronze no DuckDB.
- **Camada Gold (`dbt/models/gold/`):** Serie temporal diaria analitica filtrada para o subsistema `SE`, ordenada cronologicamente, com chave temporal unica, testes de qualidade automatizados (`not_null`, `unique` e verificacoes de continuidade temporal) pronta para consumo pela engenharia de atributos.
- **Engenharia de Atributos, Modelagem e Backtesting (`src/energy_forecast/`):** Construcao de features livre de vazamento temporal (data leakage), execucao de backtesting com janelas expansivas sobre multiplas origens de previsao, registro de hiperparametros, metricas e artefatos no **MLflow** (`artifacts/`), e persistencia das previsoes em `data/predictions/`.
- **Visualização opcional (`dashboard/app.py`):** aplicação Streamlit com tema escuro, navegação lateral, previsão de sete dias, faixas empíricas de incerteza, avaliação histórica, filtros, tabelas e exportação CSV. A apresentação não modifica os modelos nem as previsões salvas.
- **Diagnósticos de risco e experimentos:** módulos de calibração, acompanhamento de previsões emitidas e comparação de candidatos. Os relatórios ficam em `artifacts/risk/` e `artifacts/candidates/`; essas análises não promovem candidatos automaticamente.

A especificacao detalhada da arquitetura e as decisoes de engenharia estao documentadas em [`docs/architecture.md`](docs/architecture.md) e [`docs/decisions.md`](docs/decisions.md).

## 5. Stack Tecnologica

| Categoria | Tecnologia | Finalidade |
| :--- | :--- | :--- |
| Linguagem | Python >= 3.11 | Linguagem principal de ingestao, features, ML e orquestracao CLI |
| Armazenamento Colunar | Apache Parquet (PyArrow) | Preservação do conteúdo baixado na Bronze e persistência dos artefatos de previsão |
| Banco Analitico Local | DuckDB | Processamento OLAP local de alta performance sobre Parquet |
| Transformacao e Qualidade | dbt Core + dbt-duckdb | Modelagem declarativa e testes de integridade das camadas Silver e Gold |
| Manipulacao de Dados | pandas + PyArrow | Engenharia de atributos temporais e avaliacao de backtesting |
| Machine Learning | LightGBM + scikit-learn + holidays | Modelo preditivo tabular, calendario de feriados nacionais e metricas |
| MLOps / Tracking | MLflow >= 3.0 | Registro de experimentos, parametros, metricas, modelos e artefatos |
| Qualidade de Codigo | Ruff + pytest | Linting, formatacao padronizada e testes unitarios/integracao |
| Empacotamento | Docker + Docker Compose + Makefile | Ambiente conteinerizado e atalhos de execução; versões mínimas declaradas em `pyproject.toml` |
| Visualizacao (Opcional) | Streamlit >= 1.49 + Plotly | Painel interativo opcional e separado do pipeline |

## 6. Instrucoes de Instalacao

### Pre-requisitos

- Python 3.11 ou superior
- Git
- Docker e Docker Compose (opcional, para execucao conteinerizada)
- Acesso à internet para instalar dependências e baixar dados ONS; a experiência com temperatura também consulta Open-Meteo.
- `make` é opcional. No Windows, os comandos Python abaixo podem ser executados diretamente no PowerShell.

### Configuracao Local (Ambiente Virtual)

1. Crie e ative um ambiente virtual na raiz do projeto:

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

```powershell
# Windows / PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Instale o projeto em modo editavel com as dependencias de desenvolvimento (e opcionalmente o dashboard):

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,dashboard]"
```

3. Crie os arquivos locais de configuracao de ambiente e profile do dbt a partir dos exemplos versionados:

```bash
# Linux / macOS
cp .env.example .env
cp dbt/profiles.yml.example dbt/profiles.yml
```

```powershell
# Windows / PowerShell
Copy-Item .env.example .env
Copy-Item dbt/profiles.yml.example dbt/profiles.yml
```

Execute os comandos na raiz do projeto, com o ambiente virtual ativo. As
configurações principais ficam em `config/settings.yaml`; `.env` fornece
sobrescritas. Copie os exemplos somente se os arquivos locais ainda não existirem,
para preservar configurações já personalizadas. O projeto está preparado para
o subsistema SE e horizonte de sete dias; alterar apenas uma variável de ambiente
não garante suporte a outros recortes ou horizontes.

### Execucao via Docker

```bash
docker compose build
docker compose run --rm pipeline python scripts/ingest.py
docker compose run --rm pipeline python scripts/run_dbt.py --command build
docker compose run --rm pipeline python scripts/evaluate.py
docker compose run --rm pipeline pytest
```

O serviço `pipeline` executa `pytest` por padrão, mas os testes de integração
precisam dos dados e artefatos gerados nas etapas anteriores. Os diretórios
`data/` e `artifacts/` são montados como volumes locais.

Para os serviços opcionais:

```bash
docker compose --profile optional up dashboard
# Em outro terminal:
docker compose --profile tracking up mlflow
```

O dashboard usa a porta 8501 e o MLflow, a porta 5000. Os comandos Docker foram
conferidos com `docker-compose.yml`; o build Docker não foi executado nesta revisão.

## 7. Etapas do Pipeline

Os pontos de entrada estão centralizados em `scripts/`; as etapas principais têm
atalhos no `Makefile`. Para a primeira execução, faça ingestão, dbt e avaliação
nessa ordem. `evaluate.py` também treina o modelo final e gera a previsão operacional.

1. **Ingestao Bronze:**
   ```bash
   python scripts/ingest.py
   # ou: make ingest
   ```
2. **Transformacao e Testes de Qualidade (Silver e Gold via dbt + DuckDB):**
   ```bash
   python scripts/run_dbt.py --command build
   # ou: make dbt
   ```
3. **Treinamento operacional, quando for necessário apenas reajustar a previsão:**
   ```bash
   python scripts/train.py
   # ou: make train
   ```
   Salva a previsão de sete dias e a importância das features, preserva a emissão
   e adiciona faixas se houver backtest disponível. **Não registra uma nova versão
   no MLflow.** Não substitui a avaliação histórica necessária ao dashboard.

4. **Backtesting, previsão final e registro no MLflow:**
   ```bash
   python scripts/evaluate.py
   # ou: make evaluate
   ```
   Avalia por padrão as 52 origens semanais mais recentes, treina novamente no
   histórico completo e registra modelo, métricas e artefatos no MLflow.
   O número de origens pode ser informado com `--max-origins`; os testes e
   experimentos do snapshot documentado esperam 52.

5. **Interface do MLflow e Dashboard Opcional:**
   ```bash
   mlflow ui --backend-store-uri sqlite:///artifacts/mlflow.db --port 5000
   # Ou: make mlflow-ui
   ```
   Em outro terminal, na raiz do projeto:
   ```bash
   python -m streamlit run dashboard/app.py
   # Ou: make dashboard
   ```
   A porta padrão do dashboard é 8501. Para usar a porta 8502, utilizada durante
   a revisão visual deste projeto:
   ```bash
   python -m streamlit run dashboard/app.py --server.port 8502
   ```

6. **Diagnósticos adicionais, executados conforme a necessidade:**
   ```bash
   python scripts/audit_project.py
   python scripts/audit_project.py --retrain
   python scripts/evaluate_risk.py
   ```
   A auditoria recalcula os indicadores e verifica artefatos; `--retrain` também
   reproduz as previsões históricas em `artifacts/review/`, sem criar versão no
   MLflow. `evaluate_risk.py` persiste faixas, cobertura, limites e acompanhamento;
   também atualiza as faixas da previsão atual e arquiva uma emissão com horário
   real. Portanto, esse último comando **não é somente leitura**.

Os comandos `audit_project.py`, `evaluate_risk.py`, `fetch_weather.py` e
`evaluate_candidates.py` são executados diretamente; não possuem atalhos próprios
no `Makefile` atual. Nenhuma rotina é agendada automaticamente.

## 8. Roadmap de Implementacao

- [x] **Etapa 1 — Preparacao do Terreno e Arquitetura:** estrutura de diretorios, `pyproject.toml`, Docker, configuracao dbt/DuckDB, utilitarios centrais de configuracao/logging, testes iniciais, lint com Ruff e documentacao arquitetural.
- [x] **Etapa 2 — Inspecao do Dicionario ONS e Ingestao Bronze:** verificacao do dicionario oficial e dos arquivos no S3 do ONS (2018 ao presente), download idempotente, preservacao dos Parquets brutos sem alteracao de schema e geracao do manifesto de ingestao (`data/bronze/ingestion_manifest.json`).
- [x] **Etapa 3 — Camadas Silver e Gold com dbt:** criacao das sources sobre os Parquets da Bronze, padronizacao e auditoria na Silver (`main_silver.stg_ons_carga_diaria`), construcao da serie diaria do subsistema `SE` na Gold (`main_gold.fct_carga_diaria_se`) e 22 testes de qualidade/continuidade no dbt.
- [x] **Etapa 4 — Baseline Sazonal e Engenharia de Atributos:** implementacao do modelo Seasonal Naive ($t-7$) e do pipeline de features temporais (calendario, feriados BR, lags 7/14/28 e medias/desvios moveis ancorados na origem da previsao sem vazamento temporal).
- [x] **Etapa 5 — Modelo LightGBM, Backtesting e MLflow:** treinamento sob validacao temporal com 52 origens sucessivas (364 dias avaliados), comparacao contra o baseline (MAE e MAPE por horizonte e agregados) e rastreamento completo no MLflow (`lightgbm-se-daily-load` v1).
- [x] **Etapa 6 — Consolidacao dos Resultados e Dashboard Opcional:** documentacao final das metricas reais obtidas no backtesting e disponibilizacao do painel opcional em Streamlit (`dashboard/app.py`).

## 9. Como Executar os Testes e Verificacoes de Qualidade

Para executar a suite de testes automatizados com `pytest`:

```bash
pytest
# ou: make test
```

A suíte combina testes unitários, de interface e de integração. Parte dela lê
dados e artefatos locais, que não são versionados. Os testes de integração contêm
assertivas do snapshot documentado (por exemplo, 3.240 dias e término em
14/11/2026); uma atualização dos dados exige revisar essas expectativas.
Executar `pytest` em uma instalação vazia ou após atualizar o snapshot pode falhar
por esses contratos, sem que isso represente falha do algoritmo.

Para validar o padrao de codigo e a formatacao com `Ruff`:

```bash
ruff check src scripts tests dashboard
ruff format --check src scripts tests dashboard
# ou: make lint
```

## 10. Resultados da Avaliacao (Baseline vs. LightGBM)

Os resultados abaixo foram obtidos por **backtesting temporal com janelas expansivas (expanding window walk-forward)** sobre a serie diaria real da camada Gold do subsistema `SE` (`2018-01-01` a `2026-11-14`), avaliando **52 origens de previsao semanais sucessivas** (`2025-11-15` a `2026-11-07`, totalizando **364 dias previstos** no periodo de `2025-11-16` a `2026-11-14`) registrados no MLflow (`run_id = 1461985175154c79ac2151ff5f6acf15`, modelo `lightgbm-se-daily-load` versao `1`).

### Resposta Executiva (LightGBM supera o Baseline?)

**O LightGBM supera o baseline no agregado e na média de cada um dos sete horizontes do backtest documentado.** Isso não significa vitória em todos os dias, semanas ou meses, nem garante o mesmo resultado em novas previsões.
No consolidado das 52 janelas semanais (364 dias avaliados), o LightGBM reduziu o **MAE** de **2.322,70 MWmed** para **1.701,53 MWmed** (**melhoria de 26,74%**, ou -621,17 MWmed de erro absoluto medio) e reduziu o **MAPE** de **5,22%** para **3,79%** (**melhoria relativa de 27,38%**).
Os maiores ganhos médios ocorrem nos horizontes $t+1$ e $t+2$ (acima de **36% de redução de MAE**). As features de carga recente, calendário e lags fazem parte do modelo, mas esta comparação não isola a contribuição causal de cada feature.

### Tabela Comparativa Agregada (52 Origens / 364 Dias Avaliados)

| Modelo | MAE (MWmed) | MAPE (%) | Melhoria vs. Baseline (MAE %) | Melhoria vs. Baseline (MAPE %) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Baseline Sazonal ($t-7$) | 2.322,70 | 5,22% | — | — | Referencia |
| **LightGBM** | **1.701,53** | **3,79%** | **+26,74%** | **+27,38%** | **LightGBM supera Baseline** |

### Desempenho Real Estratificado por Horizonte de Previsao ($t+1$ a $t+7$)

| Horizonte ($h$) | Amostras ($N$) | MAE Baseline (MWmed) | MAE LightGBM (MWmed) | Melhoria MAE (%) | MAPE Baseline (%) | MAPE LightGBM (%) | Melhoria MAPE (%) | Status no Horizonte |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| $t+1$ | 52 | 2.106,04 | **1.342,11** | **+36,27%** | 5,31% | **3,37%** | **+36,49%** | LightGBM supera Baseline |
| $t+2$ | 52 | 2.647,98 | **1.664,95** | **+37,12%** | 5,94% | **3,72%** | **+37,38%** | LightGBM supera Baseline |
| $t+3$ | 52 | 2.524,62 | **1.921,87** | **+23,87%** | 5,48% | **4,15%** | **+24,34%** | LightGBM supera Baseline |
| $t+4$ | 52 | 2.062,76 | **1.831,22** | **+11,22%** | 4,39% | **3,91%** | **+11,00%** | LightGBM supera Baseline |
| $t+5$ | 52 | 2.271,75 | **1.810,03** | **+20,32%** | 4,96% | **3,95%** | **+20,33%** | LightGBM supera Baseline |
| $t+6$ | 52 | 2.443,49 | **1.679,67** | **+31,26%** | 5,37% | **3,64%** | **+32,24%** | LightGBM supera Baseline |
| $t+7$ | 52 | 2.202,26 | **1.660,88** | **+24,58%** | 5,12% | **3,82%** | **+25,39%** | LightGBM supera Baseline |

## 11. Faixas de previsão e limites de interpretação

As faixas empíricas têm meta nominal de 90% e usam erros absolutos anteriores do
mesmo horizonte, conhecidos até a origem. A calibração exige pelo menos 26 amostras
e utiliza até as 52 mais recentes. Sem histórico suficiente, os limites ficam
indisponíveis. Cada alvo avaliado é excluído da própria calibração.

Na avaliação disponível, houve cobertura de **92,31% em 182 previsões avaliáveis**.
Esse resultado não garante a mesma cobertura futura: dependência temporal, mudanças
de regime e treinamentos sucessivos limitam a interpretação da meta nominal.
MAPE de 3,79% também não equivale a acurácia de 96,21% ou intervalo de ±3,79%.

As referências padrão ficam em `config/risk_policy.yaml`: erro diário de até 5%
por horizonte, peso 2 para subestimativa e 1 para superestimativa. A rotina de
monitoramento usa ainda referência de no máximo 20% dos dias acima do limite,
janelas de 28/84 dias e razão de degradação de 1,25. São parâmetros exploratórios,
sem validação de SLA ou conversão dos pesos em custo financeiro.

As origens do backtest documentado são semanais, aos sábados. O teste utiliza
o histórico consolidado disponível no projeto; não simula integralmente revisões
do ONS e horários de publicação. Não há avaliação prospectiva intocada documentada.
Dados locais e artefatos estão no `.gitignore`; as dependências usam faixas de
versão abertas, sem lockfile. A reprodução comprovada refere-se ao ambiente e ao
snapshot revisados, e não a qualquer instalação futura.

## 12. Experimentos com calendário e temperatura

O perfil operacional continua sendo o LightGBM com calendário nacional. Há um
perfil candidato com Carnaval, Quarta de Cinzas, Corpus Christi, pontes e fim de
ano, além de um candidato com temperatura prevista por rodadas ECMWF via Open-Meteo.

```bash
python scripts/fetch_weather.py
python scripts/evaluate_candidates.py
```

O download guarda arquivos brutos e snapshots em `data/weather/` e requer rede.
A comparação usa 39 semanas para desenvolvimento e 13 reservadas para verificação
retrospectiva, persistindo a seleção antes de avaliar o período reservado.
O protocolo atual exige os artefatos de um backtest com 52 origens.

No experimento disponível, o candidato com temperatura obteve MAPE de **2,83%**,
contra **4,04%** do operacional nos mesmos **84 alvos**. Apenas 12 das 13 origens
reservadas tinham temperatura completa; a rodada de 19/09/2026 foi excluída sem
preenchimento. Esse resultado não deve ser comparado diretamente aos 3,79% do
backtest completo, pois os períodos e amostras são diferentes.

O candidato permanece **experimental**. A temperatura é média simples de quatro
localidades, a disponibilidade assume atraso de seis horas e a proveniência das
rodadas antigas precisa de confirmação. O histórico já havia sido consultado;
portanto, a verificação reservada não é um holdout prospectivo intocado. Nenhum
experimento substitui automaticamente o modelo operacional no MLflow ou no painel.

Detalhes e fontes: [faixas, acompanhamento e modelos candidatos](docs/risk_and_candidates.md).

## 13. Dashboard e artefatos disponíveis

O dashboard usa tema escuro e navegação lateral com quatro seções:

- **Previsão:** 21 dias de histórico, sete dias previstos, faixas empíricas,
  tabelas diárias e botão **Baixar CSV**.
- **Validação:** séries realizadas e previstas, filtro por horizonte, erros mensais,
  piores dias e referência padrão de erro.
- **Histórico:** fatores usados pelo modelo e consulta da série com filtro por ano.
- **Guia:** explicação dos termos, métricas e limites de interpretação.

O calendário do cabeçalho informa o período exibido, acompanhado da data da base.
A seção “Risco e acompanhamento” e seus controles por sessão foram removidos;
as rotinas técnicas e seus relatórios continuam disponíveis por CLI. Nenhuma rotina
é agendada automaticamente e não há envio de alertas externos.

| Local | Conteúdo |
| --- | --- |
| `data/bronze/` | Parquets ONS e manifesto de ingestão |
| `data/silver/`, `data/gold/` | Exportações analíticas do dbt |
| `data/predictions/` | Backtest, previsão atual e diretório `issued_forecasts/` |
| `artifacts/backtest_summary.json` | Resumo da avaliação e identificação no MLflow |
| `artifacts/mlflow.db`, `artifacts/mlruns/` | Rastreamento e modelos locais |
| `artifacts/review/` | Auditoria e reprodução do backtest |
| `artifacts/risk/` | Cobertura, limites, cortes críticos e monitoramento |
| `artifacts/candidates/` | Seleção congelada e comparação dos candidatos |
| `data/weather/` | Rodadas meteorológicas e relatório de cobertura |
| `.streamlit/config.toml`, `dashboard/styles.css` | Tema e estilos da interface |

As emissões são arquivadas com horário real. O acompanhamento só pontua dias
inteiros posteriores à emissão quando os valores reais chegam à Gold. Os dados
do backtest não são registrados como se fossem previsões operacionais emitidas.
`evaluate_risk.py` recalcula os relatórios, mas a base precisa ser atualizada
separadamente para obter novos valores reais.

Referências internas: [arquitetura](docs/architecture.md),
[dicionário de dados](docs/data_dictionary.md), [decisões](docs/decisions.md) e
[revisão técnica de 17/11/2026](docs/project_review_2026-11-17.md).

