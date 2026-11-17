# Faixas, acompanhamento e melhoria dos modelos

Implementação de 17/11/2026 dos itens 2 a 5. Complementa a revisão técnica anterior.
Os resultados operacionais continuam usando o LightGBM nacional; o candidato com
temperatura permanece identificado como experimental até confirmar a proveniência
e disponibilidade histórica dos dados meteorológicos e obter avaliação prospectiva.

O dashboard apresenta a referência padrão na aba **Validação do modelo**. A seção
“Risco e acompanhamento” e os controles de ajuste por sessão foram removidos.
As faixas de previsão e as rotinas de análise continuam disponíveis.

## Faixas de previsão

Cada horizonte recebe uma faixa simétrica em MWmed, usando quantil dos erros
absolutos fora da amostra conhecidos até a origem, do mesmo horizonte. A calibração
exige ao menos 26 amostras e usa no máximo as 52 mais recentes. O índice do quantil é
`ceil((n+1) × cobertura)`; se exceder a amostra, a faixa fica indisponível, sem
interpolação ou promessa artificial de cobertura.

A avaliação histórica recalibra cada previsão só com alvos anteriores já conhecidos:
o alvo que está sendo avaliado nunca calibra sua própria faixa. Depois das 26 origens
iniciais, houve **182 previsões avaliáveis, com 168 cobertas (92,31%)** para a meta
nominal de 90%. A cobertura por horizonte variou de 88,46% a 96,15% (26 amostras em cada).
As larguras médias variaram de aproximadamente 5.846 a 8.522 MWmed. Não se trata de
uma garantia de cobertura futura: dependência temporal, mudanças de regime e
treinamentos sucessivos limitam a interpretação de cobertura nominal.

O gráfico, a tabela e o CSV operacional incluem os limites. O fundo azul marca a
janela prevista; a banda em torno da curva representa a faixa empírica. Sem amostras
suficientes, o status é `insufficient_history` e os limites ficam nulos.

## Monitoramento

O acompanhamento operacional lê somente `data/predictions/issued_forecasts/`, onde
as previsões geradas por `train.py`, `evaluate.py` e `evaluate_risk.py` são preservadas
com horário real de emissão. Não retrodata a emissão para a origem antiga.

Quando a Gold recebe novos valores reais, as previsões são reconciliadas por data.
Somente dias inteiros posteriores à emissão são pontuados. Reemissões da mesma
origem/horizonte não substituem a primeira previsão. Em datas com múltiplas origens,
o monitor diário usa o menor horizonte disponível, regra fixa que evita duplicar dias.

Janelas: 28 dias recentes e 84 anteriores, mínimo de 14 observações. Alertas locais:

- MAPE recente superior ao baseline;
- MAPE recente acima de 1,25 vez o MAPE da referência;
- taxa de dias acima do limite maior que a referência de 20%, por horizonte;
- cobertura recente abaixo da meta nominal, com amostra suficiente.

As comparações são descritivas e podem gerar alertas em amostras pequenas; não são
testes estatísticos de mudança de regime. O painel separa **previsões emitidas** de
**simulação com o backtest**. Nesta execução não há alvos operacionais maduros; o
painel informa que está aguardando reais. Na simulação histórica, os últimos 28 dias
geraram alertas por violações dos limites e cobertura de 82,14%, abaixo de 90%.

Não foram criados agendamentos nem notificações externas. O monitor é atualizado
ao abrir/reexecutar o painel e pode ser persistido por `scripts/evaluate_risk.py`.

## Limites e assimetria

`config/risk_policy.yaml` mantém referências iniciais, escolhidas após o usuário
delegar a definição: 5% de erro percentual absoluto diário em cada horizonte,
no máximo 20% dos dias acima desse limite, peso 2 para subestimativa e 1 para
superestimativa. São parâmetros exploratórios, sem validação de custo financeiro
ou SLA. O painel usa essas referências fixas; alterações do pipeline ficam centralizadas
no arquivo de configuração.

O erro ponderado é `mean(2 × max(real - previsto, 0) + max(previsto - real, 0))`,
em MWmed multiplicado por peso relativo. Não é custo monetário. O MAPE permanece
separado do critério de violação diária: um MAPE abaixo de 5% não implica que cada
dia erre menos de 5%. Nos dados atuais, todos os horizontes excedem a referência
de 20% de violações; os alertas não foram relaxados para esconder esse resultado.

## Experimentos com períodos críticos e temperatura

O perfil opcional `critical` acrescenta Carnaval, Quarta de Cinzas, Corpus Christi,
pontes de feriado e período de 24/dez a 2/jan. São eventos do calendário; o código
não os classifica todos como feriados legais nacionais. O perfil padrão `national`
e suas features continuam reproduzindo o modelo operacional. O relatório apresenta
cortes com N explícito; grupos podem se sobrepor e alguns têm amostras muito pequenas.

A avaliação reserva 39 semanas para desenvolvimento (273 previsões) e as últimas
13 para verificação. A seleção é persistida em `frozen_selection.json` **antes** de
avaliar o período reservado. A regra predefinida escolhe menor erro ponderado, com
ganho mínimo de 2% e sem regressão de MAE no desenvolvimento. O calendário adicional
teve ganho pequeno (MAPE 3,70% vs. 3,73%), insuficiente para justificar promoção.

Foram baixadas rodadas de temperatura ECMWF por inicialização, via Open-Meteo.
O proxy regional usa média simples de São Paulo, Rio de Janeiro, Belo Horizonte e
Brasília. As médias diárias exigem 24 horas presentes nas quatro localidades.
Das 139 origens consultadas, 138 estavam completas (966 alvos). A rodada de
19/09/2026 estava incompleta e foi excluída integralmente, sem preenchimento.
O aquecimento e resfriamento são derivados das temperaturas previstas, com bases
fixas de 18 °C e 22 °C. Não se usa temperatura realizada no futuro como feature.

O modelo com temperatura e seu controle usam exatamente os mesmos pares semanais
de treino, isolando a contribuição incremental das features meteorológicas. A
disponibilidade é assumida 6 horas após a inicialização UTC, coerente com o atraso
típico documentado pelo provedor; horários reais de disponibilização não foram
comprovados para cada rodada. O código rejeita snapshots declarados posteriores
ao dia da origem e rodadas posteriores à disponibilidade declarada.

| Desenvolvimento, N=273 | MAPE | MAE (MWmed) | Erro ponderado |
| --- | ---: | ---: | ---: |
| Operacional | 3,73% | 1.659,76 | 2.641,89 |
| Calendário crítico | 3,70% | 1.648,09 | 2.630,98 |
| Controle do clima, mesmas linhas de treino | 3,85% | 1.704,03 | 2.489,19 |
| Temperatura | 2,79% | 1.234,40 | 1.901,65 |

Seleção congelada: temperatura. Das 13 origens reservadas, 12 possuem temperatura
completa; a comparação abaixo usa somente os mesmos **84 alvos** em todos os modelos:

| Verificação retrospectiva | MAPE | MAE (MWmed) | Erro ponderado |
| --- | ---: | ---: | ---: |
| Operacional | 4,04% | 1.846,79 | 3.030,04 |
| Controle do clima, mesmas linhas de treino | 3,90% | 1.780,31 | 2.851,31 |
| Temperatura | 2,83% | 1.296,77 | 2.296,78 |

O resultado é promissor, mas **não demonstra desempenho prospectivo**. O histórico
já havia sido consultado anteriormente, e o arquivo meteorológico pode conter
hindcasts ou reprocessamentos. O proxy regional também não é ponderado pela carga
de cada localidade. O candidato não foi promovido, não altera as previsões atuais
e não gera faixas usando resíduos do modelo operacional.

## Arquivos e comandos

- `artifacts/risk/`: faixas prequenciais, cobertura por horizonte, limites,
  períodos críticos e `monitoring.json`.
- `artifacts/candidates/`: seleção congelada, previsões e comparação dos candidatos.
- `data/weather/`: rodadas brutas, cobertura e snapshots; dados locais ignorados no Git.
- `data/predictions/issued_forecasts/`: emissões operacionais preservadas.

```powershell
# Atualiza faixas, registra a previsão atual como emitida agora e persiste o monitor.
.\.venv\Scripts\python.exe scripts/evaluate_risk.py
# Atualiza somente o arquivo meteorológico; requer rede.
.\.venv\Scripts\python.exe scripts/fetch_weather.py
# Reexecuta os candidatos predefinidos, sem alterar o registro do modelo operacional.
.\.venv\Scripts\python.exe scripts/evaluate_candidates.py
```

O treino operacional passa a preservar emissões e calibrar faixas quando há
backtest disponível. A avaliação completa calibra o modelo final com os resíduos
da própria execução. Não use resíduos de um modelo antigo para calibrar um modelo
alterado: reexecute o backtest antes de avaliar suas faixas.

Validação inicial: 61 testes Python aprovados, Ruff e `pip check` aprovados. Testes
cobrem a imunidade das faixas a valores futuros, insuficiência de calibração,
assimetria de erro, condições de alerta, emissão sem retrodatação, calendário,
disponibilidade meteorológica. A simplificação da interface é verificada por testes
de renderização e filtros, incluindo a ausência da aba e dos controles removidos.

Referências: [Open-Meteo Single Runs](https://open-meteo.com/en/docs/single-runs-api),
[limites das hipóteses de permutabilidade em conformal prediction](https://mapie.readthedocs.io/en/stable/content/exchangeability-testing/).
