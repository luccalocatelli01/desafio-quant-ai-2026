# MODEL_SPEC — Desafio Quant AI 2026
**Congelado em:** 2026-08-15 18:09:30
**Hash SHA-256 de `intermediario\b4b5_previsoes.parquet`:** `5f67e19fa21e1b22f8d29064fca75ba06c498e2b62ac6b87ae67f332eb385e67`

---

## 1. UNIVERSO
- Inclusão: `CODBDI`='02' (string), `TPMERC`=10 (Int64) — medidos em A1.
- Classes de `ESPECI`: raiz ∈ {ON, PN, PNA, PNB, PNC, PND, PNE, PNF, PNG, UNT} — raiz = primeiro token de `ESPECI`, maiúsculo, com todos os caracteres NÃO alfabéticos removidos do FIM (regex `[^A-Z]+$`). Regra corrigida em A1b.

## 2. AJUSTE SOCIETÁRIO
- Escopo: marcadores G (grupamento) e B (bonificação) + `FATCOT`.
- NÃO cobre proventos D/J/R/S (dividendo/juros/rendimento/subscrição) — 6,32% do universo carrega marcador de provento não neutralizado (L08). A série é price-only; é por isso que o benchmark do target também é price-only (D12).

## 3. GRADE MENSAL
- Uma linha por (CODISI, mês) EXISTENTE = última sessão negociada do mês (preço), com VOLTOT/QUATOT/TOTNEG somados no mês e `n_sessoes`.
- Buraco (mês sem negócio) = AUSÊNCIA DE LINHA. Nunca preço herdado, nunca `ffill`/`fillna`.

## 4. ELEGIBILIDADE — QUATRO CAMADAS (mesma família; cada uma descoberta depois que a anterior já estava congelada)
- **D08** — ≥36 meses COM OBSERVAÇÃO de histórico até t; negociou em t; `n_sessoes`≥4 em t.
- **D10** — `n_sessoes`≥4 também em t−1 (perna anterior, retorno bem formado).
- **D19** — a janela de K meses de qualquer feature só existe se os K meses forem consecutivos E o ISIN for ELEGÍVEL em todos eles; senão, NaN.
- **D25** — o universo de PREVISÃO é diferente do universo de TREINO: treino exige as duas pernas (`elegivel_treino`, D09); previsão condiciona SOMENTE a perna t (`elegivel`), porque condicionar a seleção em t+1 seria look-ahead (proibido por D09).
- Mecanismo comum às quatro: um filtro de universo não é uma linha de código — é uma propriedade que precisa ser reverificada em CADA uso do dado.

## 5. SETOR
- Ponte ISIN → mapa: `raiz = CODISI[2:6]` se começa com `BR`; senão `raiz = CODISI[0:4]` (D07).
- Chave do especialista: `SETOR || SUBSETOR` (D06), **45 valores distintos**.
- Sem correspondência → `SEM_SETOR` literal → alimenta SEMPRE o generalista, nunca é descartado.

## 6. BENCHMARK (D12)
- Índice interno, peso igual, sobre o universo ELEGÍVEL (nunca sobre o arquivo inteiro).
- Reportado também em versão EX-MAIOR-CONTRIBUINTE mensal (`ret_indice_ex_max`, D16) — toda métrica de desempenho do projeto é reportada nas duas versões, dos dois lados (estratégia e benchmark).

## 7. TAXA LIVRE DE RISCO (D13)
- CDI (SGS-12), percentual ao dia útil, composto por dias úteis do mês. Unidade medida (não assumida) antes de qualquer composição.

## 8. FEATURES (D17 corrigida por D18/D19) — 6 FIXAS
`mom_1m`, `mom_3m`, `mom_6m`, `mom_12m` (produto de (1+`ret_1m`) na janela de K meses, menos 1); `vol_3m`, `vol_6m` (desvio-padrão amostral de `ret_1m` na janela). Janela exige K meses consecutivos E elegíveis (D19).
**Força relativa PROIBIDA**: `rel_1m`/`rel_3m`/`rel_6m` são, sob rank cross-seccional mensal, idênticas às `mom_K` correspondentes (identidade algébrica — mom do índice é constante na seção transversal do mês) — verificado nesta base para K=1 [F], K=3 e K=6 (D18).

## 9. TRANSFORMAÇÃO (D20)
Rank percentil cross-seccional em (0,1], método 'average', dentro de cada mês, calculado SÓ sobre ELEGÍVEIS com a feature não-nula. Nunca ao longo do tempo, nunca sobre não-elegíveis.

## 10. ALVO (D21)
`alfa_fut` = `ret_1m`(t+1) − `ret_indice`(t+1). Excesso sobre o benchmark price-only do mesmo universo — o modelo ordena ativos relativamente, não prevê o mercado.

## 11. WALK-FORWARD (D22, D26)
- Janela de 36 meses, re-treino MENSAL.
- Para o mês de decisão T: `mes(t)` ∈ **[T−37, T−2]**; último mês de ALVO usado = **T−1**.
- Leitura conservadora de uma ambiguidade do prompt original, declarada em D26; é a única compatível com a sanidade de ausência de look-ahead.
- Estrutura idêntica à de março, verificada como livre de look-ahead por 4 replicações independentes [F].

## 12. ARQUITETURA (D23)
- **D01**: um subsetor tem ESPECIALISTA no mês T se, e só se, (i) ≥5 ativos elegíveis com features completas em T E (ii) ≥150 observações de treino na janela de 36 meses.
- Não satisfeito → ativos do subsetor recebem score do GENERALISTA.
- GENERALISTA: OLS sobre TODAS as observações da janela, de todos os subsetores, inclusive SEM_SETOR; treinado em todo mês T.
- SEM_SETOR SEMPRE vai ao generalista.
- 6 fatores fixos para TODOS os modelos. OLS COM intercepto (o alvo não tem média cross-seccional exatamente zero por janela). SEM regularização (sem ridge, sem lasso).
- **PROIBIDA qualquer seleção de fatores** — por estabilidade, IC, p-valor ou qualquer critério [F: seleção "por estabilidade" de março era indistinguível de sortear 3 de 7 — erro-padrão da IC média (~0,031) maior que a diferença entre configurações (0,054 vs 0,072); validação reusava janelas com 35 de 36 meses em comum].

## 13. CARTEIRA — congelados AGORA, antes de existir qualquer backtest
- **K** = 3 ativos por especialista por padrão; **K = 2** se o terceiro score for menor que metade do primeiro.
- **Gate de confiança**: um especialista só aloca se o score MÉDIO dos seus selecionados for > 0; caso contrário, não aloca nada naquele mês.
- **Pesos**: IGUAIS entre todos os ativos selecionados de todos os especialistas ativos.
- **Capital não alocado**: CDI.
- O **GENERALISTA também seleciona**, pelas mesmas regras, tratado como um especialista.
- **Rebalanceamento**: mensal, no fechamento da última sessão negociada do mês.
- **Custo**: 50 bps por rebalanceamento; curva 0/10/25/50 bps medida em C3.
- **SEM** teto de peso, **SEM** piso, **SEM** vol targeting, **SEM** K dinâmico por rank de score [F: a máquina de pesos de março destruía 25-28 bps/mês; o teto de 15% era violado em 78% dos meses; o vol targeting dava 2,15x mais peso a papel parado].
- **Deslistagem**: congelamento/liquidação pelo motor local. PROIBIDO `fillna(0)`.
- **Reporte**: toda métrica em duas versões, com e sem o maior contribuinte mensal (D16).

---

## DECLARAÇÃO FINAL

Este MODEL_SPEC foi congelado em 2026-08-15 18:09:30, APÓS medir o IC fora da amostra
(+0,0025, t=0,24) e ANTES de qualquer backtest de carteira existir. Nenhum parâmetro
será alterado a partir deste ponto. Qualquer alteração posterior será registrada
como ESCOLHA PÓS-RESULTADO, com data, e declarada no relatório final.

---

## REFERÊNCIA DAS DECISÕES CITADAS
D01, D02, D03, D04, D05, D06, D07, D08, D09, D10, D11, D12, D13, D14, D15, D16, D17, D18, D19, D20, D21, D22, D23, D24, D25, D26 — ver DOSSIE.md seção 1 para o texto completo de cada uma.

## VERIFICAÇÃO S1 (todo parâmetro corresponde ao implementado em A1-B5)
Percorrido o código de todas as células do notebook (`notebooks\00_pipeline.ipynb`, células 1-25) contra cada linha deste documento. Confirmado, entre outros: filtro `CODBDI=='02'`/`TPMERC==10` (A1); marcadores G/B+FATCOT do ajuste societário (A2); `primeira_obs` e ausência de preenchimento de buraco (A3); `MIN_HIST, MIN_SESSOES, MIN_ATIVOS_D01, MIN_OBS_D01 = 36, 4, 5, 150` e as colunas `elegivel_hist`/`elegivel_sessoes`/`elegivel_perna_anterior` (A8); ponte `CODISI[2:6]`/`CODISI[0:4]` e `subsetor_chave` = setor+subsetor (A5); benchmark `mean(ret_1m)` sobre `elegivel & ret_valido` e `ret_indice_ex_max` (A6); fator do CDI `1 + valor/100` (A6); janela `elegivel_k` aplicada às 6 features, sem `rel_1m`/`rel_3m`/`rel_6m` na lista `FEATURES` (B1); `rank(method='average', pct=True)` por mês (B2); `alfa_fut = ret_1m_t1 - ret_indice_t1` (B3); janela `ini_jan, fim_jan = T-(JANELA+1), T-2` e `OLS` via `np.linalg.lstsq` com coluna de 1's (intercepto), sem `Ridge`/`Lasso`/`alpha=` em nenhuma célula (B4/B5). **Nenhuma divergência encontrada.**
