# Dicionario de Dados e Especificacao da Fonte — ONS Carga de Energia Diaria

## 1. Identificacao do Conjunto de Dados

- **Nome Oficial no Portal ONS:** Carga de Energia Diaria
- **Pagina do Conjunto de Dados:** https://dados.ons.org.br/dataset/carga-energia
- **Endpoint de Metadados (API CKAN):** `https://dados.ons.org.br/api/3/action/package_show?id=carga-energia`
- **Caminho no Bucket S3 Publico:** `s3://ons-aws-prod-opendata/dataset/carga_energia_di/`
- **Dicionarios Oficiais Inspecionados:**
  - `https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/carga_energia_di/DicionarioDados_Carga_Energia_Diaria.pdf` (Versao 1.2, publicada em 01-05-2023)
  - `https://ons-aws-prod-opendata.s3.amazonaws.com/dataset/carga_energia_di/DicionarioDados_Carga_Energia_Diaria.json`
- **Periodicidade Temporal:** Diaria.
- **Escopo do Projeto:** Janeiro de 2018 ate a data mais recente disponivel (`2026-11-14`), filtrado na camada Gold para o subsistema **Sudeste/Centro-Oeste (`SE`)**.

---

## 2. Especificacao Oficial das Colunas (Dicionario ONS v1.2)

De acordo com o documento `DicionarioDados_Carga_Energia_Diaria.pdf` (Versao 1.2) e o recurso `DicionarioDados_Carga_Energia_Diaria.json`, o conjunto de dados possui 4 colunas obrigatorias:

| Codigo da Coluna | Descricao Oficial | Tipo Logico (Dicionario PDF) | Formato Especificado | Permite Nulo | Permite Zero | Permite Negativo |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `id_subsistema` | Codigo do Subsistema | `TEXTO` | 3 posicoes | Nao | N/A | N/A |
| `nom_subsistema` | Nome do Subsistema | `TEXTO` | 60 posicoes (ampliado de 20 para 60 na v1.2) | Nao | N/A | N/A |
| `din_instante` | Data de referencia | `DATETIME` | `YYYY-MM-DD HH:MM:SS` | Nao | N/A | N/A |
| `val_cargaenergiamwmed` | Valor da Carga de Energia, em MWmed | `FLOAT` | Separador decimal ponto (`.`) | Nao | Nao | Nao |

### Representacao Real dos Subsistemas nos Arquivos

Nos 9 arquivos anuais inspecionados (2018 a 2026), os 4 subsistemas do Sistema Interligado Nacional (SIN) sao representados exatamente pelos seguintes pares `(id_subsistema, nom_subsistema)`:

- `'SE'` -> `'Sudeste/Centro-Oeste'` **(Subsistema alvo deste projeto)**
- `'S'` -> `'Sul'`
- `'NE'` -> `'Nordeste'`
- `'N'` -> `'Norte'`

---

## 3. Alteracoes Metodologicas Documentadas pelo ONS

Conforme registrado oficialmente nos metadados do conjunto de dados no portal CKAN do ONS, a composicao da serie historica de carga diaria passou por tres marcos metodologicos:

1. **Ate fevereiro de 2021:** Os valores representam a carga atendida por usinas despachadas e/ou programadas pelo ONS, com base nos dados recebidos pelo Sistema de Supervisao e Controle do ONS.
2. **De marco de 2021 a 28 de abril de 2023:** Os valores representam a carga atendida por usinas despachadas e/ou programadas pelo ONS, acrescida da previsao de geracao de usinas nao despachadas pelo ONS.
3. **A partir de 29 de abril de 2023:** Alem das parcelas anteriores, passou a ser incorporado o valor estimado da micro e minigeracao distribuida (**MMGD**), calculado com base em dados meteorologicos previstos.
4. **Processo de Consistencia Recorrente:** O ONS destaca que os dados passam por processo de consistencia continua e podem sofrer atualizacoes retrospectivas apos a publicacao inicial. Por esse motivo, a camada Bronze registra o `ETag`, `LastModified` e `SHA-256` no manifesto para detectar eventuais reprocessamentos na fonte.

---

## 4. Inspecao Fisica dos Arquivos Parquet Disponiveis (2018–2026)

A verificacao direta no bucket `s3://ons-aws-prod-opendata/dataset/carga_energia_di/` identificou 9 arquivos Parquet anuais para o periodo do projeto (`CARGA_ENERGIA_2018.parquet` a `CARGA_ENERGIA_2026.parquet`), totalizando **12.960 registros** nos 4 subsistemas (**3.240 dias** para o subsistema `SE`), sem nenhum valor nulo (`nulls = 0` em todas as colunas).

> [!WARNING]
> **Divergencia Real de Tipo Fisico Detectada nos Parquets Oficiais do ONS:**
> Embora o dicionario logico defina `val_cargaenergiamwmed` como `FLOAT`, a inspecao dos cabecalhos Parquet revelou que:
> - Nos arquivos de **2018, 2019, 2020 e 2021**, a coluna `val_cargaenergiamwmed` foi gravada pelo ONS com tipo fisico Arrow `string` (ex.: `'30268.42412500'`).
> - Nos arquivos de **2022, 2023, 2024, 2025 e 2026**, a coluna `val_cargaenergiamwmed` foi gravada com tipo fisico Arrow `double` (`float64`).
>
> **Tratamento Arquitetural:** Na camada **Bronze**, os arquivos Parquet originais sao preservados bit a bit sem modificacao. Na camada **Silver** (Etapa 3, via dbt + DuckDB), a leitura unificada utilizara conversao explicita (`TRY_CAST(val_cargaenergiamwmed AS DOUBLE)`) e validacao de nao-nulidade/positividade.

### Inventario Auditado na Camada Bronze

| Ano | Arquivo | Linhas (Total SIN) | Linhas (`SE`) | Periodo Coberto | Tipo Fisico `din_instante` | Tipo Fisico `val_cargaenergiamwmed` | Tamanho (Bytes) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 2018 | `CARGA_ENERGIA_2018.parquet` | 1.460 | 365 | `2018-01-01` a `2018-12-31` | `timestamp[ns]` | `string` | 24.221 |
| 2019 | `CARGA_ENERGIA_2019.parquet` | 1.460 | 365 | `2019-01-01` a `2019-12-31` | `timestamp[ns]` | `string` | 24.256 |
| 2020 | `CARGA_ENERGIA_2020.parquet` | 1.464 | 366 | `2020-01-01` a `2020-12-31` | `timestamp[ns]` | `string` | 24.428 |
| 2021 | `CARGA_ENERGIA_2021.parquet` | 1.460 | 365 | `2021-01-01` a `2021-12-31` | `timestamp[ns]` | `string` | 24.384 |
| 2022 | `CARGA_ENERGIA_2022.parquet` | 1.460 | 365 | `2022-01-01` a `2022-12-31` | `timestamp[ns]` | `double` | 19.772 |
| 2023 | `CARGA_ENERGIA_2023.parquet` | 1.460 | 365 | `2023-01-01` a `2023-12-31` | `timestamp[ns]` | `double` | 19.767 |
| 2024 | `CARGA_ENERGIA_2024.parquet` | 1.464 | 366 | `2024-01-01` a `2024-12-31` | `timestamp[ns]` | `double` | 19.803 |
| 2025 | `CARGA_ENERGIA_2025.parquet` | 1.460 | 365 | `2025-01-01` a `2025-12-31` | `timestamp[ns]` | `double` | 19.524 |
| 2026 | `CARGA_ENERGIA_2026.parquet` | 1.272 | 318 | `2026-01-01` a `2026-11-14` | `timestamp[ns]` | `double` | 17.649 |

---

## 5. Contrato por Camada da Arquitetura Medallion

### 5.1 Camada Bronze (`data/bronze/*.parquet`)

- **Schema:** Identico ao arquivo bruto fornecido pelo ONS para cada ano (sem renomear colunas, sem converter tipos, sem filtrar subsistemas e sem remover linhas).
- **Metadados Externos:** Registrados em `data/bronze/ingestion_manifest.json` (URL de origem, URI S3, ETag, LastModified, ano, hash SHA-256, contagem de linhas, periodo inicial/final, lista de subsistemas, schema fisico Arrow e timestamps UTC de ingestao e verificacao).

### 5.2 Camada Silver (`silver.stg_ons_carga_diaria`)

- **Responsabilidade:** Unificacao do schema fisico entre 2018–2021 (`string`) e 2022–2026 (`double`), padronizacao de datas e auditoria de qualidade sobre todos os registros da Bronze.
- **Colunas Previstas:**
  - `id_subsistema` (`VARCHAR`, normalizado com `TRIM`/`UPPER`)
  - `nom_subsistema` (`VARCHAR`, normalizado com `TRIM`)
  - `din_instante` (`DATE`, extraido de `CAST(din_instante AS DATE)`)
  - `val_cargaenergiamwmed_raw` (`VARCHAR`, valor original preservado para rastreabilidade)
  - `val_cargaenergiamwmed` (`DOUBLE`, obtido via `TRY_CAST(val_cargaenergiamwmed AS DOUBLE)`)
  - `is_valid_load` (`BOOLEAN`: indica se `val_cargaenergiamwmed IS NOT NULL AND val_cargaenergiamwmed > 0`)
  - `duplicate_rank` (`BIGINT`: numeracao por `(id_subsistema, din_instante)` para rastrear eventuais duplicidades na fonte)
  - `_source_file` (`VARCHAR`: caminho/nome do arquivo Parquet da Bronze de onde o registro se originou)
  - `_ingested_at` (`TIMESTAMP`: instante de processamento no dbt)

### 5.3 Camada Gold (`gold.fct_carga_diaria_se`)

- **Responsabilidade:** Serie temporal diaria oficial pronta para modelagem preditiva do subsistema `SE`.
- **Regras de Filtragem e Qualidade:**
  - `id_subsistema = 'SE'`
  - `din_instante >= DATE '2018-01-01'`
  - Apenas registros validos (`is_valid_load = TRUE` e `duplicate_rank = 1`), com testes do dbt garantindo que anomalias na Silver sejam sinalizadas.
- **Colunas Previstas:**
  - `data_referencia` (`DATE`, Primary Key, `not_null`, `unique`, ordenada cronologicamente)
  - `id_subsistema` (`VARCHAR`, constante `'SE'`, `not_null`)
  - `nom_subsistema` (`VARCHAR`, `'Sudeste/Centro-Oeste'`)
  - `carga_mwmed` (`DOUBLE`, `not_null`, carga diaria em MWmed)
  - `dias_desde_observacao_anterior` (`INTEGER`, diferenca em dias em relacao a data anterior para verificacao automatica de continuidade da serie)
