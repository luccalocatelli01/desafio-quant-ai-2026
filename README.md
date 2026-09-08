# desafio-quant-ai-2026

Estratégia quantitativa de seleção de ações na B3, construída para o Desafio Quant AI 2026 (Itaú Asset). Regressões OLS por subsetor com re-treino mensal, avaliada em 103 meses de teste.

**A estratégia não superou o CDI nem um benchmark de peso igual. Este repositório documenta esse resultado e os testes que o sustentam.**

## Resultado

Teste: decisões de 2018-01 a 2026-07, retornos de 2018-02 a 2026-08 — 103 meses. Custo de 50 bps por perna, cobrado sobre duas pernas, fixado antes de qualquer backtest existir.

| Série | Bruto (% a.a.) | Líq. 50 bps (% a.a.) | Vol (% a.a.) | DD máx (%) | Meses + (%) |
|---|---|---|---|---|---|
| V5 — modelo final (1/vol) | 8,97 | 7,22 | 19,00 | −31,01 | 52,4 |
| CDI | 9,03 | 9,03 | 1,19 | 0,00 | 100,0 |
| Peso igual, universo elegível | 6,72 | 5,91 | 21,44 | −33,90 | 51,5 |
| Índice interno (não investível) | 7,19 | 7,19 | 21,75 | −34,57 | 53,4 |

Três leituras do mesmo número:

1. **Contra o CDI** — o modelo perde no acumulado geométrico já no bruto: 8,97% contra 9,03% a.a. Em média aritmética mensal o excedente é +0,15%/mês no bruto (t de Newey-West 3 lags = 0,26) e +0,013%/mês líquido de 50 bps (t = 0,02). A diferença entre as duas leituras é arrasto de variância: 19% a.a. de volatilidade contra 1,19% do CDI.
2. **Contra o peso igual** — o excedente é indistinguível de zero: t de Newey-West (3 lags) = −0,02. A inversão contra o comparador de manchete acontece em torno de 64,5 bps de custo, acima do caso base.
3. **Contra o acaso** — teste de permutação com 2.000 réplicas (semente 20260815): p = 0,0425 testando apenas o decil inferior; p = 0,3985 corrigindo para as 10 decis; **p = 0,7350** corrigindo para 10 decis × 5 construções. Não é possível descartar sorte.

O Sharpe não é usado como critério de ordenação: com excesso negativo contra o CDI, μ/σ cresce com σ (Israelsen, 2005). A tabela completa, com a curva de custo de 0 a 100 bps e todas as construções intermediárias, está em [`TABELA_FINAL.csv`](TABELA_FINAL.csv).

O que sobrevive é mecanismo, não alfa: o score separa perdedores melhor do que separa ganhadores, e a construção de carteira bate o 1/N sobre os mesmos nomes em +0,10%/mês no bruto (t = 0,94) — positivo em 4 de 4 construções, mas também indistinguível de zero.

## A estratégia

Ranquear ações por momento e volatilidade dentro do próprio subsetor, e comprar as melhores de cada um.

- **Universo** — B3, COTAHIST de 1995 a 2026. Essa é a base; a janela de teste é 2018-01 a 2026-07, e a série de backtest começa em 2001-12, depois de satisfeitas as exigências de histórico e de janela de features. Lote padrão à vista (`CODBDI` = 02, `TPMERC` = 10), classes ON/PN/UNT. Elegibilidade exige 36 meses de histórico observado, negociação no mês de decisão e no anterior, e pelo menos 4 sessões em cada.
- **Features** — seis, fixas: momento de 1, 3, 6 e 12 meses e volatilidade de 3 e 6 meses. Cada janela só existe se os K meses forem consecutivos *e* o ativo for elegível em todos eles. Transformadas em rank percentil cross-seccional dentro de cada mês.
- **Alvo** — retorno do mês seguinte menos o retorno do índice interno do mesmo universo. O modelo ordena ativos relativamente; não prevê o mercado.
- **Arquitetura** — um OLS por subsetor (45 chaves setor+subsetor) quando há pelo menos 5 ativos elegíveis e 150 observações na janela; caso contrário o ativo cai para um modelo generalista treinado sobre tudo. Sem regularização e **sem seleção de fatores** — a seleção "por estabilidade" testada antes era indistinguível de sortear 3 de 7.
- **Walk-forward** — janela de 36 meses, re-treino mensal. Para o mês de decisão T, os meses de treino vão de T−37 a T−2 e o último alvo usado é T−1.
- **Carteira** — 3 ativos por especialista (2 se o terceiro score for menor que metade do primeiro), pesos iguais, capital não alocado no CDI, rebalanceamento mensal. Sem teto de peso, sem piso, sem vol targeting.

Os parâmetros foram congelados em [`decisoes/MODEL_SPEC_2026-08-15.md`](decisoes/MODEL_SPEC_2026-08-15.md) **depois** de medir o IC fora da amostra e **antes** de existir qualquer backtest de carteira. As 26 decisões estruturais, cada uma datada e com o mecanismo que a justifica, estão em [`decisoes/DECISOES_D01_D26.md`](decisoes/DECISOES_D01_D26.md).

**Sobre o histórico deste repositório.** O projeto foi desenvolvido fora de controle de versão, em agosto de 2026, e publicado aqui em commit único em 08/09/2026. O histórico do git não corrobora a cronologia descrita acima — o registro cronológico são os arquivos datados em `decisoes/` e o carimbo de data e hora no cabeçalho do MODEL_SPEC. Esses arquivos foram escritos por mim e datados por mim; não são prova no sentido forte. O que é verificável de fato é outra coisa: o MODEL_SPEC registra o hash SHA-256 do arquivo de previsões, e quem rodar o pipeline pode conferir se o modelo que este repositório descreve é o mesmo que gerou os resultados reportados.

## Como rodar

Requisitos: Python 3.11+.

```
git clone https://github.com/luccalocatelli01/desafio-quant-ai-2026.git
cd desafio-quant-ai-2026
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/macOS
pip install -r requirements.txt
```

Os dados **não estão neste repositório** — são 640 MB e vêm todos de fontes públicas. O próprio código os baixa:

- **COTAHIST (B3)** — `dados.baixar_cotahist(anos, destino)` baixa os ZIPs anuais e `dados.parse_cotahist_ano` os converte para parquet.
- **CDI, série SGS-12 (Banco Central)** — `dados.baixar_cdi_sgs12(data_inicio, data_fim)`.
- **Classificação Setorial (B3)** — baixe manualmente a planilha em [b3.com.br](https://www.b3.com.br/pt_br/produtos-e-servicos/negociacao/renda-variavel/acoes/consultas/classificacao-setorial/) e salve em `dados/`.

Com `dados/` preenchido, abra `notebooks/00_pipeline.ipynb` e execute as células em ordem. Cada célula imprime um bloco de sanidade com as contagens e checagens que ela precisa passar antes da seguinte; a série de decisão é reprodutível a partir do hash registrado no MODEL_SPEC.

## Estrutura

```
src/dados.py       download, parsing do COTAHIST, ajuste societário, elegibilidade
src/motor.py       aplicação de pesos, turnover, custos, backtest
src/metricas.py    retorno, vol, drawdown, VaR/CVaR, tracking error, turnover, HHI
notebooks/         pipeline completo, célula a célula, com os blocos de sanidade
decisoes/          MODEL_SPEC congelado, as 26 decisões datadas, log de uso de IA
figuras/           as 18 figuras do relatório
TABELA_FINAL.csv   tabela consolidada de resultados, com a curva de custo inteira
```

## Limitações

- A série é **price-only**. O ajuste societário cobre grupamento, bonificação e `FATCOT`, mas não proventos em dinheiro: 6,32% do universo carrega marcador de provento não neutralizado. É por isso que o benchmark do alvo também é price-only.
- **Não há partição de holdout reservada.** O split treino/teste é único e foi fixado em [`decisoes/D03_sem_holdout_2026-08-15.md`](decisoes/D03_sem_holdout_2026-08-15.md); os resultados de teste não devem ser lidos como fora da amostra em sentido estrito.
- Os 50 bps de custo são o caso conservador. O custo efetivo típico de ações líquidas na B3 fica entre 3 e 15 bps. Trocar a manchete para 10 bps foi considerado e rejeitado: compraria 1,40 pp de retorno e nenhuma conclusão nova.
- O modelo foi testado em um único mercado, em um único período, com um único conjunto de seis fatores.

## Uso de IA generativa

O desafio exigia uso de IA generativa em pelo menos uma etapa. O log completo, célula a célula e com os erros da IA documentados, está em [`decisoes/USO_DE_IA.md`](decisoes/USO_DE_IA.md).

Em resumo: um orquestrador escrevia os prompts de cada célula e mantinha o dossiê sem executar código; Claude Code implementava uma célula por vez e o humano conferia o bloco de sanidade antes de liberar a próxima; e quatro rodadas de conselho adversarial com revisores cegos atacaram o modelo. O conselho derrubou o achado central — foi ele que endureceu o teste de permutação até p = 0,7350. Três erros da IA estão registrados no log, com data e correção. Nenhum LLM entra na estratégia: a IA construiu, mediu e auditou, mas não decide alocação.

## Contexto e créditos

Projeto do Desafio Quant AI 2026 (Itaú Asset), competição acadêmica de estratégias quantitativas. A equipe ficou entre os 5% melhores na etapa de relatório final e foi eliminada nas quartas de final.

O código deste repositório foi escrito por Lucca Locatelli, com contribuições da equipe do desafio nas decisões de modelagem, na revisão dos resultados e na consolidação do relatório final.

## Licença

MIT — ver [LICENSE](LICENSE).
