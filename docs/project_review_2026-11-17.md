# Revisão do projeto — 17/11/2026

O projeto funciona no ambiente local verificado e o LightGBM apresenta vantagem
histórica sobre o baseline sazonal. Isso sustenta a escolha entre os dois modelos
avaliados, mas não demonstra que o erro será baixo em qualquer dia futuro. Falta uma
tolerância operacional definida conforme o custo de superestimar/subestimar carga.

## Evidências executadas

- 46 testes Python aprovados, incluindo proteção contra vazamento temporal,
  integridade de dados, cálculos de métricas e interação com os filtros Streamlit.
- `dbt build`: duas tabelas reconstruídas e 22 testes de qualidade aprovados.
- Ruff: lint e formatação aprovados. `pip check`: sem conflitos no ambiente instalado.
- Bronze: SHA-256 dos nove Parquets de 2018–2026 confere com o manifesto local.
- Gold: 3.240 dias, de 01/01/2018 a 14/11/2026; datas únicas, série contínua,
  cargas positivas e finitas. O nulo da observação anterior no primeiro dia é esperado.
- 52 origens semanais do backtest reexecutadas: 364 previsões reproduzidas com
  `rtol=1e-7, atol=1e-5`, sem criar nova versão no MLflow nem sobrescrever o resultado original.
- 41 verificações dos artefatos aprovadas: resumo e métricas por horizonte recalculados,
  valores reais alinhados à Gold, baseline igual ao lag de sete dias, horizontes completos,
  datas causais, unicidade e reprodução das previsões.
- MLflow local: execução `1461985175154c79ac2151ff5f6acf15` em `FINISHED`;
  modelo `lightgbm-se-daily-load`, versão 1, em `READY`.

Resultados detalhados em `artifacts/review/project_audit.json`,
`monthly_errors.csv` e `reproduced_predictions.parquet`. Os artefatos são locais e
ignorados pelo controle de versão conforme o `.gitignore` existente.

## Qualidade das previsões

| Indicador | Baseline sazonal | LightGBM |
| --- | ---: | ---: |
| MAE (MWmed) | 2.322,70 | 1.701,53 |
| MAPE | 5,22% | 3,79% |
| Redução relativa do MAE | — | 26,74% |
| Redução relativa do MAPE | — | 27,38% |

O LightGBM vence nas médias dos sete horizontes. Seus MAPEs por horizonte variam
entre 3,37% e 4,15%. Vence em MAE em 39/52 origens (75%) e em erro absoluto
em 234/364 previsões (64,3%).

| Diagnóstico do LightGBM | Resultado |
| --- | ---: |
| Mediana do erro percentual absoluto | 3,31% |
| Percentil 90 | 7,67% |
| Percentil 95 | 9,85% |
| Maior erro percentual absoluto | 14,58% |
| Dias com erro até 5% | 256/364 (70,3%) |
| Dias com erro acima de 10% | 17/364 (4,7%) |
| Viés médio (previsto menos real) | −367,93 MWmed |
| RMSE | 2.142,13 MWmed |

O maior erro ocorreu em 27/12/2025: real de 46.604,42 MWmed e previsão de
39.810,67 MWmed. O modelo perde em MAPE mensal em dezembro/2025
(5,38% contra 5,24%) e fevereiro/2026 (4,16% contra 3,56%). Novembro/2026
tem MAPE de 6,08%, mas cobre somente 14 dias. Meses parciais são identificados
pela quantidade de amostras no painel e no CSV mensal.

O bootstrap pareado de origens completas, 5.000 reamostragens com semente 42,
produziu intervalo descritivo de 95% de 405,73 a 837,98 MWmed para a redução
média do MAE. Preserva a dependência entre os sete horizontes da mesma origem;
semanas adjacentes ainda podem ser dependentes. Não é intervalo preditivo nem
certificação de significância ou de prontidão operacional.

## Correções realizadas

- Rejeição explícita de NaN/infinito em métricas e de cargas inválidas no preparo da série.
- Continuidade diária validada antes das médias móveis: sete observações devem
  representar sete dias, sem lacunas silenciosas.
- Bloqueio de inferência com origem anterior ao corte de treinamento do modelo,
  evitando uso acidental de um modelo treinado no futuro em avaliação passada.
- Validação de finitude também na Silver e no teste de carga positiva da Gold.
- Dashboard confere resumo contra previsões e valores reais contra Gold;
  inconsistências exibem mensagem de recuperação em vez de indicadores incoerentes.
- Horizonte t+1 deixa de ser chamado de “amanhã” sem relação com a data atual.
  A origem e a defasagem usam o fuso `America/Sao_Paulo`.
- Comparação por horizonte passa a distinguir vitória, derrota e resultado misto/empate.
- Mínimos declarados de Streamlit e MLflow ajustados às APIs utilizadas;
  `tzdata` explícito para disponibilidade do fuso também em Windows/contêineres.
  Não foi feita instalação limpa para testar todas as versões mínimas.

## Interface

Skill utilizada: [Impeccable, repositório oficial](https://github.com/pbakaus/impeccable),
versão 4.9.2, instalada globalmente no Codex. Aplicadas as orientações de
polimento, robustez e modo Operate. A skill anteriormente excluída não foi utilizada.

Mantidos Streamlit, Plotly, tabelas, glossário e as funções do painel. A apresentação
agora prioriza a previsão operacional, reduz textos repetidos, encurta as legendas,
separa os limites históricos e exibe a atualização da base. Inclui exportação CSV,
filtro por horizonte e ano, erro mensal, piores dias e tolerância diária exploratória.
A importância por número de divisões não é interpretada como efeito causal.

Tokens em `dashboard/styles.css` e tema em `.streamlit/config.toml`; o Dockerfile
inclui o tema. Sem fontes remotas ou novas bibliotecas de interface.

Verificação visual pelo navegador em desktop (1366 px) e celular (390 px);
as tabelas largas preservam rolagem interna e as abas usam navegação nativa.
O detector mecânico do Impeccable não apresentou ocorrências nos arquivos alterados.
Isso não equivale a uma auditoria WCAG completa. Leitor de tela, gestos em aparelho
físico e todos os navegadores não foram testados.

## Limites e próximos passos

1. Definir limites por horizonte e período crítico e o custo assimétrico dos erros.
   Um MAPE de 3,79% não representa acurácia de 96,21% nem intervalo de ±3,79%.
2. Reservar avaliação temporal intocada antes de qualquer ajuste adicional. Não há
   registro suficiente para excluir tuning prévio no período atual de teste.
3. As origens deste teste são semanais, aos sábados. Validar também origens diárias
   caso o uso operacional passe a atualizar a previsão todos os dias.
4. Simular disponibilidade real, atrasos de publicação e revisões do ONS com snapshots
   históricos. O teste atual usa o histórico consolidado disponível no projeto.
5. Calibrar intervalos preditivos em período separado e medir cobertura. Revisar
   datas especiais e avaliar variáveis meteorológicas com disponibilidade causal.
6. A base local termina em 14/11/2026, três dias antes desta revisão. O aviso do
   painel mede a defasagem; não afirma que exista arquivo mais recente no ONS.
7. Dependências ainda têm faixas abertas e não há lockfile; a reprodução comprovada
   refere-se ao ambiente instalado. Os testes de integração dependem dos dados locais.
   Build Docker, instalação limpa e varredura de CVEs não foram executados.

## Reproduzir

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check src scripts tests dashboard
.\.venv\Scripts\ruff.exe format --check src scripts tests dashboard
.\.venv\Scripts\python.exe scripts/run_dbt.py --command build
.\.venv\Scripts\python.exe scripts/audit_project.py --retrain
.\.venv\Scripts\python.exe -m streamlit run dashboard/app.py
```

`audit_project.py` sem `--retrain` somente recalcula as verificações dos artefatos;
com a opção, reexecuta o treinamento histórico e compara as previsões originais.

Referências técnicas: [validação temporal do scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html),
[API Streamlit 1.49](https://docs.streamlit.io/1.49.0/develop/api-reference/data/st.dataframe),
[migração MLflow 3](https://mlflow.org/docs/latest/ml/mlflow-3/).
