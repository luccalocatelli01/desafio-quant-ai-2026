"""Métricas de desempenho.

Todas as funções recebem séries já produzidas pelo motor e devolvem números.
Nenhuma delas decide nada — só mede. Nenhuma lê dado de disco, nenhuma conhece
estratégia: a mesma função mede a mínima variância, o ICVaR e a carteira 1/N,
que é o que mantém a comparação entre as três honesta.

CONVENÇÕES DESTE MÓDULO (todas verificáveis nas docstrings de cada função)
-------------------------------------------------------------------------

1. PASSO SEMANAL, ANUALIZAÇÃO POR 52. A frequência de rebalanceamento decidida
   é SEMANAL (`decisoes/01`, 11/08/2026) e o backtest roda sobre a MESMA grade
   semanal que os modelos consomem. Logo `periodos_por_ano` é **52**, e a
   constante `PERIODOS_POR_ANO_SEMANAL` existe para que ninguém digite 252 por
   reflexo. **Nunca misture 52 e 252 na mesma tabela:** a volatilidade escala
   com a raiz do número de períodos, então usar 252 sobre uma série semanal
   multiplicaria a volatilidade anualizada por ~2,2 sem que nada tivesse
   mudado nos dados. O parâmetro continua explícito em cada função — a
   constante é o valor registrado, não um default embutido.

2. EXCESSO É ARITMÉTICO, EM TODAS AS MÉTRICAS. `r − rf` e `r − b`, nunca
   `(1+r)/(1+rf) − 1`. Motivo: (a) é a definição padrão de retorno ativo, e é
   o que torna tracking error e information ratio comparáveis a qualquer
   referência externa; (b) a diferença entre as duas convenções é `r·rf`, da
   ordem de 1e-5 no passo semanal — segunda ordem; (c) o objeto deste projeto
   é a comparação ENTRE as três séries, e um viés de segunda ordem aplicado
   identicamente às três cancela. Consistência vale mais aqui do que a
   correção de segunda ordem, e a escolha fica declarada em vez de implícita.

2b. NUMERADOR DAS RAZÕES DE EXCESSO (Sharpe, Sortino, IR): MÉDIA ARITMÉTICA
   ANUALIZADA (`média(e) · p`), nunca a capitalização geométrica da série de
   excesso. CORRIGIDO EM 12/08/2026, e o histórico fica declarado: até então
   o numerador era `retorno_anualizado(e)` — a capitalização geométrica de
   uma DIFERENÇA aritmética, que não é a riqueza de posição investível
   nenhuma (`prod(1 + r_t − b_t)` não corresponde a nada negociável). O
   arrasto de variância dessa composição, `≈ −σ_e²/2` ao ano, escala com a
   volatilidade do excesso e portanto NÃO cancela entre séries de
   volatilidades diferentes — a revisão adversarial de 12/08 mediu o efeito
   no resultado real: o IR reportado de −0,882 com tracking error de 37,4%
   implicava excesso de −33,0% a.a. contra uma diferença real de retornos
   anualizados de −15,9% a.a. (fator ~2), e no Sharpe o arrasto punia a
   série mais volátil em ~0,19 contra ~0,11 da menos volátil — viés
   sistemático entre as séries comparadas, não deslocamento comum. Com a
   média aritmética no numerador, o estimador volta a ser o padrão da
   literatura (`média/desvio · sqrt(p)`) e o item 2 acima segue valendo
   inalterado: a SÉRIE de excesso continua aritmética. A justificativa da
   anualização geométrica em `retorno_anualizado()` também segue valendo —
   para RETORNO DE CARTEIRA, que é riqueza de verdade, geométrico é o número
   que corresponde à curva; o que muda é só o numerador das RAZÕES.

3. TAXA LIVRE DE RISCO = CDI, IDÊNTICA NAS TRÊS SÉRIES (`decisoes/01`,
   11/08/2026, linha "CDI no universo de otimização", papel 3). Ela é
   argumento obrigatório de quem precisa dela — este módulo não sabe onde o
   CDI mora e não vai buscá-lo.

4. SINAL DE VaR E CVaR: devolvidos como **PERDA POSITIVA**. `var_historico`
   igual a 0,04 significa "perde 4% ou mais em 5% dos períodos", não "+4%".
   Convenção declarada porque as duas existem na literatura e trocá-las
   inverte a leitura da tabela inteira.

5. DENOMINADOR ZERO DEVOLVE NaN, NUNCA INFINITO. Uma tabela de relatório que
   exibe `inf` faz o leitor pensar em erro de código; `NaN` diz "não definido
   com estes dados", que é o que de fato aconteceu.

O QUE ESTE MÓDULO NÃO FAZ
-------------------------
Não roda backtest (isso é `src/motor.py`), não calcula peso (isso é
`src/modelos.py`), não lê dado (isso é `src/dados.py`), não escolhe `alpha`,
não escolhe `participacao_max` e não decide se a montagem inicial da carteira
entra na média de turnover — todos esses são argumentos obrigatórios, porque
são decisões de quem reporta, não do medidor.
"""

import numpy as np
import pandas as pd

# Passo semanal: 52 observações por ano. Ver a convenção 1 do cabeçalho. Não é
# parâmetro de modelagem — é o valor registrado da frequência já decidida.
PERIODOS_POR_ANO_SEMANAL = 52


# --- auxiliares -------------------------------------------------------------


def _serie(x, nome: str) -> pd.Series:
    """Coage para Series e recusa ausência. Falha alto, nunca em silêncio."""
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(
                f"{nome} como DataFrame precisa ter exatamente 1 coluna, "
                f"recebido {x.shape[1]}."
            )
        x = x.iloc[:, 0]
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    if len(x) == 0:
        raise ValueError(f"{nome} está vazia: não há observação para medir.")
    if x.isna().any():
        n = int(x.isna().sum())
        raise ValueError(
            f"{nome} tem {n} observação(ões) ausente(s). Métrica não preenche "
            f"buraco de série: decida o que fazer com a ausência antes de medir, "
            f"porque preencher aqui seria imputar dado dentro do relatório."
        )
    return x.astype(float)


def _alinhar(a: pd.Series, b, nome_b: str) -> tuple:
    """Alinha `b` ao índice de `a`. Escalar vira série constante.

    Não faz interseção: exige que `b` cubra TODO o índice de `a`. Interseção
    silenciosa mudaria o período medido sem ninguém perceber, e duas séries
    medidas em períodos diferentes não são comparáveis.
    """
    if np.isscalar(b):
        return a, pd.Series(float(b), index=a.index)
    b = _serie(b, nome_b).reindex(a.index)
    if b.isna().any():
        faltam = a.index[b.isna()]
        raise ValueError(
            f"{nome_b} não cobre todo o índice da série medida; faltam "
            f"{len(faltam)} data(s), a primeira em {faltam[0]!r}. As duas "
            f"séries têm de cobrir exatamente o mesmo período — recortar por "
            f"interseção aqui mudaria o período medido em silêncio."
        )
    return a, b


def _dividir(numerador: float, denominador: float) -> float:
    """Divisão que devolve NaN quando o denominador é zero. Ver convenção 5."""
    if denominador == 0 or not np.isfinite(denominador):
        return float("nan")
    return float(numerador) / float(denominador)


def _validar_periodos(periodos_por_ano) -> int:
    if not isinstance(periodos_por_ano, (int, np.integer)) or periodos_por_ano < 1:
        raise ValueError(
            f"periodos_por_ano deve ser inteiro >= 1, recebido "
            f"{periodos_por_ano!r}. Com passo semanal o valor é "
            f"{PERIODOS_POR_ANO_SEMANAL} (ver convenção 1 do módulo)."
        )
    return int(periodos_por_ano)


def _validar_alpha(alpha) -> float:
    if not np.isscalar(alpha) or not (0.0 < float(alpha) < 1.0):
        raise ValueError(
            f"alpha deve ser o NÍVEL DE CONFIANÇA, estritamente entre 0 e 1 "
            f"(ex.: 0.95 para 95%), recebido {alpha!r}. Não há default: o nível "
            f"da cauda é decisão de quem reporta e tem de ser declarado."
        )
    return float(alpha)


# --- retorno e risco --------------------------------------------------------


def retorno_anualizado(retornos, periodos_por_ano: int):
    """Retorno anualizado da série, GEOMÉTRICO (capitalizado).

    O QUE FAZ: `(prod(1 + r))^(periodos_por_ano / n) − 1`. É a taxa constante
    que, capitalizada pelo mesmo número de períodos, chega à mesma riqueza
    final — ou seja, o número que corresponde à curva de riqueza que o
    relatório vai plotar.

    O QUE NÃO FAZ: não usa a média aritmética dos retornos. A média aritmética
    anualizada é maior que a geométrica sempre que há variância, e a diferença
    NÃO é cosmética: ela cresce com a volatilidade, então favoreceria
    sistematicamente a série mais volátil das três. Usar a geométrica remove
    esse viés da comparação.

    Args:
        retornos: Series de retornos simples da carteira, no passo declarado.
        periodos_por_ano: observações por ano. Com passo semanal, 52.

    Returns:
        Float.

    Raises:
        ValueError: se a série tiver ausência, ou se algum `1 + r` for <= 0
            (carteira zerada — a capitalização não está definida a partir daí).
    """
    r = _serie(retornos, "retornos")
    p = _validar_periodos(periodos_por_ano)

    fatores = 1.0 + r
    if (fatores <= 0).any():
        d = fatores.index[fatores <= 0][0]
        raise ValueError(
            f"há período com 1 + retorno <= 0 (em {d!r}): a carteira zerou ou "
            f"ficou negativa e o retorno geométrico não está definido a partir "
            f"daí. O motor levanta o mesmo erro pelo mesmo motivo."
        )
    return float(fatores.prod() ** (p / len(r)) - 1.0)


def volatilidade_anualizada(retornos, periodos_por_ano: int):
    """Desvio-padrão anualizado da série.

    O QUE FAZ: `std(r, ddof=1) * sqrt(periodos_por_ano)`. `ddof=1` é a
    estimativa não enviesada da variância populacional; a diferença some com
    n grande, mas declarar qual foi usada evita que dois membros da equipe
    reportem números levemente diferentes para a mesma série.

    O QUE ASSUME: retornos independentes entre períodos. A raiz do tempo é
    exata só sob independência; com autocorrelação ela subestima (positiva) ou
    superestima (negativa) a volatilidade anual. É a convenção universal e a
    usamos, mas a premissa fica declarada.

    Args:
        retornos: Series de retornos simples da carteira.
        periodos_por_ano: observações por ano. Com passo semanal, 52.

    Returns:
        Float.
    """
    r = _serie(retornos, "retornos")
    p = _validar_periodos(periodos_por_ano)
    if len(r) < 2:
        raise ValueError(
            "volatilidade exige ao menos 2 observações; com uma só não há "
            "dispersão para medir."
        )
    return float(r.std(ddof=1) * np.sqrt(p))


def excesso_sobre_benchmark(retornos, retornos_benchmark):
    """Série de retorno em excesso sobre o benchmark. ARITMÉTICO: `r − b`.

    Ver a convenção 2 do módulo para por que aritmético e não geométrico.

    O QUE NÃO FAZ: não recorta por interseção. Se o benchmark não cobre todo o
    período da carteira, levanta erro — medir a carteira num período e o
    benchmark em outro produziria um "excesso" que não é excesso de nada.

    Args:
        retornos: Series de retornos da carteira.
        retornos_benchmark: Series de retornos do benchmark (a carteira 1/N,
            `decisoes/01`, 10/08/2026), ou escalar.

    Returns:
        Series de excesso, no índice da carteira.
    """
    r = _serie(retornos, "retornos")
    r, b = _alinhar(r, retornos_benchmark, "retornos_benchmark")
    return r - b


def sharpe(retornos, taxa_livre_risco, periodos_por_ano: int):
    """Índice de Sharpe anualizado, sobre o excesso aritmético contra o CDI.

    O QUE FAZ: `média(r − rf) · p / volatilidade_anualizada(r − rf)` — o
    estimador padrão `média/desvio · sqrt(p)`. Numerador e denominador medem
    a MESMA série de excesso — misturar retorno da carteira no numerador com
    volatilidade do excesso no denominador é um erro comum que produz número
    sem interpretação.

    CONVENÇÃO DO NUMERADOR (2b do cabeçalho, corrigida em 12/08/2026): média
    ARITMÉTICA anualizada, não capitalização geométrica da série de excesso.
    A capitalização geométrica de uma diferença aritmética embute um arrasto
    de `≈ −σ_e²/2` ao ano que cresce com a volatilidade — viés sistemático
    ENTRE séries de volatilidades diferentes, que é exatamente a comparação
    deste projeto.

    O QUE ASSUME: `taxa_livre_risco` já está no MESMO passo de tempo dos
    retornos (semanal, se a série for semanal). Uma taxa anual passada aqui
    sem conversão infla o desconto por ~52 e derruba o Sharpe das três séries.

    RESSALVA REGISTRADA (`decisoes/01`, 10 e 11/08/2026): os preços do COTAHIST
    não têm provento, então o Sharpe absoluto fica SUBESTIMADO nas três séries.
    A comparação ENTRE elas não é afetada, porque o viés é simétrico.

    Args:
        retornos: Series de retornos da carteira.
        taxa_livre_risco: Series (mesmo passo) ou escalar. É o CDI, aplicado
            identicamente às três séries.
        periodos_por_ano: observações por ano. Com passo semanal, 52.

    Returns:
        Float. NaN se o excesso não tiver dispersão.
    """
    r = _serie(retornos, "retornos")
    r, rf = _alinhar(r, taxa_livre_risco, "taxa_livre_risco")
    p = _validar_periodos(periodos_por_ano)
    excesso = r - rf
    return _dividir(
        float(excesso.mean()) * p, volatilidade_anualizada(excesso, p)
    )


def sortino(retornos, taxa_livre_risco, periodos_por_ano: int):
    """Sortino anualizado: excesso sobre o CDI dividido pelo desvio DOWNSIDE.

    O QUE FAZ: no denominador, a semi-dispersão negativa
    `sqrt(mean(min(e, 0)^2)) * sqrt(periodos_por_ano)`, com `e = r − rf`.

    O QUE ASSUME, e é a escolha que precisa ficar explícita: a média do
    quadrado é dividida por **n (todas as observações)**, não pelo número de
    observações negativas. Essa é a definição padrão de semi-desvio, e é a
    única das duas que se reduz ao desvio-padrão quando a distribuição é
    simétrica. Dividir pelo número de negativas inflaria o denominador de
    séries com poucas perdas e tornaria os Sortinos das três séries
    incomparáveis entre si.

    POR QUE ELE ACOMPANHA O SHARPE AQUI: o modelo central é um ICVaR, cujo
    objetivo é explicitamente a cauda ESQUERDA. Uma métrica que pune desvio
    para cima do mesmo jeito que para baixo mede mal exatamente aquilo que o
    modelo tenta otimizar. Sortino não substitui o Sharpe — os dois vão na
    tabela.

    Args:
        retornos: Series de retornos da carteira.
        taxa_livre_risco: Series (mesmo passo) ou escalar.
        periodos_por_ano: observações por ano. Com passo semanal, 52.

    Returns:
        Float. NaN se não houver nenhum período abaixo do CDI (denominador
        zero — ver convenção 5).
    """
    r = _serie(retornos, "retornos")
    r, rf = _alinhar(r, taxa_livre_risco, "taxa_livre_risco")
    p = _validar_periodos(periodos_por_ano)

    excesso = r - rf
    abaixo = excesso.clip(upper=0.0)
    desvio_downside = float(np.sqrt((abaixo**2).mean()) * np.sqrt(p))
    # Numerador: média aritmética anualizada — convenção 2b, 12/08/2026.
    return _dividir(float(excesso.mean()) * p, desvio_downside)


def serie_drawdown(riqueza, riqueza_inicial: float):
    """Série completa de drawdown, para o gráfico do relatório.

    O QUE FAZ: `riqueza / pico_até_agora − 1`, com o pico **semeado em
    `riqueza_inicial`**.

    POR QUE `riqueza_inicial` É OBRIGATÓRIO: `serie_riqueza()` devolve a série
    já com o retorno do primeiro período aplicado — o capital inicial NÃO é
    uma observação dela. Sem semear o pico, uma carteira que cai no primeiro
    período teria drawdown 0 naquele instante, porque ela mesma seria o próprio
    pico. O erro só aparece quando a série começa perdendo, que é justamente o
    caso em que o número importa.

    Args:
        riqueza: Series de riqueza acumulada, saída de `serie_riqueza()`.
        riqueza_inicial: capital inicial passado ao motor. Sem default: é ele
            que define o primeiro pico.

    Returns:
        Series de drawdown (<= 0) indexada por data.
    """
    w = _serie(riqueza, "riqueza")
    if riqueza_inicial is None or not np.isscalar(riqueza_inicial) or riqueza_inicial <= 0:
        raise ValueError(
            f"riqueza_inicial deve ser um número > 0, recebido "
            f"{riqueza_inicial!r}. É ela que semeia o primeiro pico; sem ela o "
            f"drawdown do período inicial sairia zerado por construção."
        )
    pico = np.maximum(w.cummax(), float(riqueza_inicial))
    return w / pico - 1.0


def drawdown_maximo(riqueza, riqueza_inicial: float):
    """Maior queda percentual do pico ao vale da série de riqueza.

    Args:
        riqueza: Series de riqueza acumulada.
        riqueza_inicial: capital inicial. Ver `serie_drawdown()`.

    Returns:
        Float negativo (ou zero).
    """
    return float(serie_drawdown(riqueza, riqueza_inicial).min())


def calmar(retornos, riqueza, riqueza_inicial: float, periodos_por_ano: int):
    """Calmar: retorno anualizado dividido pelo módulo do drawdown máximo.

    O QUE ASSUME: `retornos` e `riqueza` vêm da MESMA execução do motor. A
    função confere que os índices coincidem, porque um Calmar montado com o
    retorno de uma série e o drawdown de outra é um número sem significado e
    nada no tipo dos argumentos denunciaria a troca.

    O QUE NÃO FAZ: não anualiza o drawdown — drawdown não escala com o tempo,
    é o pior caso observado na janela inteira. Isso significa que o Calmar
    **depende do comprimento da amostra**: janelas mais longas tendem a conter
    drawdowns piores, então Calmars de períodos de tamanhos diferentes não são
    comparáveis. Vale para comparar as três séries entre si (mesma janela),
    não para comparar com número de terceiro.

    Args:
        retornos: Series de retornos da carteira.
        riqueza: Series de riqueza da mesma execução.
        riqueza_inicial: capital inicial.
        periodos_por_ano: observações por ano. Com passo semanal, 52.

    Returns:
        Float. NaN se o drawdown máximo for zero.
    """
    r = _serie(retornos, "retornos")
    w = _serie(riqueza, "riqueza")
    if not r.index.equals(w.index):
        raise ValueError(
            "retornos e riqueza têm índices diferentes: as duas séries têm de "
            "vir da MESMA execução do motor. Um Calmar com o retorno de uma "
            "carteira e o drawdown de outra não significa nada."
        )
    p = _validar_periodos(periodos_por_ano)
    dd = drawdown_maximo(w, riqueza_inicial)
    return _dividir(retorno_anualizado(r, p), abs(dd))


# --- cauda ------------------------------------------------------------------


def var_historico(retornos, alpha: float):
    """VaR histórico (empírico) no nível de confiança `alpha`. PERDA POSITIVA.

    O QUE FAZ: devolve `−quantil(r, 1 − alpha)`. Com `alpha = 0,95`, é o
    quantil 5% da distribuição de retornos, trocado de sinal: "em 5% dos
    períodos a carteira perde ao menos este tanto".

    SINAL: perda positiva (convenção 4 do módulo). Um VaR de 0,04 é uma perda
    de 4%. Se a cauda de 5% for de ganhos, o valor sai negativo — e sair
    negativo é informação, não erro.

    O QUE NÃO FAZ: não ajusta para o tamanho da amostra e não supõe
    distribuição nenhuma. É o quantil empírico da série, e a `alpha` alta com
    poucas observações ele fica instável — com 152 observações e `alpha` 0,95
    há ~8 pontos na cauda, que foi exatamente o argumento registrado para a
    frequência semanal (`decisoes/01`, 11/08/2026).

    Args:
        retornos: Series de retornos.
        alpha: nível de confiança em (0, 1), ex.: 0.95. **Sem default.**

    Returns:
        Float. Perda positiva quando a cauda é de perdas.
    """
    r = _serie(retornos, "retornos")
    a = _validar_alpha(alpha)
    return float(-np.quantile(r.to_numpy(), 1.0 - a))


def cvar_historico(retornos, alpha: float):
    """CVaR histórico (Expected Shortfall) no nível `alpha`. PERDA POSITIVA.

    O QUE FAZ: média dos retornos que ficam **no ou abaixo** do quantil
    `1 − alpha`, trocada de sinal. É a perda média CONDICIONADA a se estar na
    cauda, e por construção `cvar >= var`.

    POR QUE ELE ESTÁ AQUI E NÃO SÓ O VaR: o modelo central otimiza CVaR
    (`decisoes/01`, 09/08/2026). Reportar só VaR mediria a carteira por uma
    régua diferente da que a construiu. O VaR também vai à tabela porque é a
    fronteira que define a cauda, mas o número que dialoga com o objetivo do
    modelo é este.

    O QUE NÃO FAZ: não é o CVaR do OTIMIZADOR. Este aqui é medido **ex-post**,
    sobre a série realizada da carteira; o do otimizador é **ex-ante**, sobre
    os 152 cenários da janela de estimação. Os dois números não têm de bater e
    compará-los diretamente seria erro de leitura.

    Args:
        retornos: Series de retornos.
        alpha: nível de confiança em (0, 1), ex.: 0.95. **Sem default.**

    Returns:
        Float. Perda positiva quando a cauda é de perdas.
    """
    r = _serie(retornos, "retornos")
    a = _validar_alpha(alpha)
    corte = np.quantile(r.to_numpy(), 1.0 - a)
    cauda = r[r <= corte]
    if len(cauda) == 0:
        return float("nan")
    return float(-cauda.mean())


# --- contra o benchmark -----------------------------------------------------


def tracking_error(retornos, retornos_benchmark, periodos_por_ano: int):
    """Desvio-padrão anualizado do excesso aritmético sobre o benchmark.

    LEMBRETE REGISTRADO (`decisoes/01`, linha "Benchmark"): a carteira 1/N é
    FACTÍVEL para o otimizador (pesos iguais pertencem ao simplex), então o
    excesso in-sample é >= 0 por construção. Um tracking error pequeno não é
    defeito nem virtude por si — ele mede **quanto o modelo se afastou** do
    1/N, e é justamente esse afastamento que a linha "Benchmark" manda
    reportar como evidência, em vez do sinal do excesso.

    Args:
        retornos: Series de retornos da carteira.
        retornos_benchmark: Series de retornos do benchmark (1/N).
        periodos_por_ano: observações por ano. Com passo semanal, 52.

    Returns:
        Float.
    """
    excesso = excesso_sobre_benchmark(retornos, retornos_benchmark)
    return volatilidade_anualizada(excesso, periodos_por_ano)


def information_ratio(retornos, retornos_benchmark, periodos_por_ano: int):
    """Excesso médio anualizado sobre o benchmark dividido pelo tracking error.

    Mesmo cuidado do Sharpe: numerador e denominador medem a MESMA série de
    excesso, e o numerador é a MÉDIA ARITMÉTICA anualizada (convenção 2b,
    corrigida em 12/08/2026).

    POR QUE NÃO O GEOMÉTRICO, e aqui o efeito era o maior do módulo: com
    tracking error grande, o arrasto de variância da capitalização geométrica
    da série de diferença domina o número. No resultado contaminado de 12/08,
    o IR de −0,882 com TE de 37,4% implicava excesso de −33,0% a.a. contra
    uma diferença real de retornos anualizados de −15,9% a.a. — o leitor da
    tabela receberia um número duas vezes maior que o afastamento verdadeiro.
    `prod(1 + r_t − b_t)` não é a riqueza de nenhuma posição investível; a
    média aritmética anualizada sobre o TE é o estimador padrão e é
    diretamente comparável a referências externas.

    Args:
        retornos: Series de retornos da carteira.
        retornos_benchmark: Series de retornos do benchmark (1/N).
        periodos_por_ano: observações por ano. Com passo semanal, 52.

    Returns:
        Float. NaN se o excesso não tiver dispersão (carteira idêntica ao
        benchmark — o que acontece, por exemplo, se o otimizador devolver
        pesos iguais).
    """
    p = _validar_periodos(periodos_por_ano)
    excesso = excesso_sobre_benchmark(retornos, retornos_benchmark)
    return _dividir(
        float(excesso.mean()) * p, volatilidade_anualizada(excesso, p)
    )


# --- giro -------------------------------------------------------------------


def turnover_medio(turnover, excluir_montagem_inicial: bool):
    """Giro médio por período, na convenção de duas pontas do motor.

    CONVENÇÃO HERDADA DO MOTOR: vender 10% de A e comprar 10% de B dá giro
    0,20, não 0,10. O custo de 50 bps registrado em `decisoes/01` incide sobre
    esse notional, então o giro reportado aqui é o mesmo número que gera a
    conta de custo — não há fator 2 escondido entre os dois.

    POR QUE `excluir_montagem_inicial` É OBRIGATÓRIO E NÃO TEM DEFAULT: o
    primeiro rebalanceamento parte de caixa e tem giro ~1,0, uma ordem de
    grandeza acima do giro de regime. Incluí-lo mede "giro médio incluindo a
    montagem"; excluí-lo mede "giro de regime". As duas leituras são legítimas
    e dão números bem diferentes numa série longa — então quem reporta declara
    qual está reportando, em vez de herdar a escolha de um default.

    A montagem inicial É COBRADA no backtest de qualquer forma (`decisoes/01`,
    11/08/2026): este parâmetro governa só o que entra nesta ESTATÍSTICA, não
    o que o motor cobra.

    Args:
        turnover: Series de giro por período, saída de `calcular_turnover()`
            ou de `rodar_backtest()`.
        excluir_montagem_inicial: se True, descarta a PRIMEIRA data com giro
            não nulo antes de tirar a média. **Sem default.**

    Returns:
        Float.
    """
    t = _serie(turnover, "turnover")
    if not isinstance(excluir_montagem_inicial, bool):
        raise ValueError(
            "excluir_montagem_inicial é obrigatório e tem de ser True ou False: "
            "a montagem inicial tem giro ~1,0 e desloca a média, então incluí-la "
            "ou não muda o número reportado. Declare a escolha em vez de herdar "
            "um default."
        )
    if excluir_montagem_inicial:
        nao_nulos = t[t > 0]
        if len(nao_nulos):
            t = t.drop(index=nao_nulos.index[0])
        if len(t) == 0:
            return float("nan")
    return float(t.mean())


def turnover_anualizado(turnover, periodos_por_ano: int, excluir_montagem_inicial: bool):
    """Giro médio por período multiplicado por `periodos_por_ano`.

    O QUE FAZ: `turnover_medio(...) * periodos_por_ano`. Com passo semanal e
    rebalanceamento toda semana, um giro médio de 0,10 por semana vira 5,2 ao
    ano — a carteira é renovada ~5 vezes por ano na convenção de duas pontas.

    O QUE NÃO FAZ: não usa raiz do tempo. Giro acumula linearmente (é uma soma
    de notionals), ao contrário da volatilidade — usar `sqrt` aqui seria erro
    de dimensão.

    Args:
        turnover: Series de giro por período.
        periodos_por_ano: observações por ano. Com passo semanal, 52.
        excluir_montagem_inicial: ver `turnover_medio()`. **Sem default.**

    Returns:
        Float.
    """
    p = _validar_periodos(periodos_por_ano)
    return turnover_medio(turnover, excluir_montagem_inicial) * p


# --- concentração -----------------------------------------------------------


def hhi(pesos):
    """Índice de Herfindahl-Hirschman da carteira, período a período.

    O QUE FAZ: `sum_i w_i^2` em cada data. Vai de `1/n` (n ativos com peso
    igual) a 1 (tudo em um ativo só). Para a carteira 1/N sobre 50 ativos, o
    HHI é exatamente 0,02 — o que dá uma régua imediata para ler o HHI dos
    modelos na mesma tabela.

    POR QUE ELE EXISTE AQUI: `decisoes/01` (12/08/2026) decidiu **não** impor
    teto de peso por ativo, porque um teto seria parâmetro sem justificativa.
    A contrapartida registrada dessa decisão é que a concentração seja
    **reportada**. Esta função é essa contrapartida; sem ela a decisão de não
    restringir ficaria sem o diagnóstico que a acompanha.

    O QUE ASSUME: `pesos` são os pesos EFETIVOS (já derivados), saída de
    `rodar_backtest()["pesos_efetivos"]`, e não os pesos-alvo. Medir só os
    alvos esconderia a concentração que a deriva do mercado produz entre dois
    rebalanceamentos.

    Args:
        pesos: DataFrame de pesos, uma linha por data e uma coluna por ativo.

    Returns:
        Series de HHI indexada por data.
    """
    if not isinstance(pesos, pd.DataFrame):
        raise TypeError(f"pesos precisa ser DataFrame, recebido {type(pesos).__name__}.")
    if len(pesos) == 0:
        raise ValueError("pesos está vazio: não há carteira para medir.")
    w = pesos.fillna(0.0).astype(float)
    return (w**2).sum(axis=1)


def hhi_medio(pesos):
    """Média temporal do HHI. Float. Ver `hhi()` para a definição e a ressalva."""
    return float(hhi(pesos).mean())


def n_efetivo_ativos(pesos):
    """Número EFETIVO de ativos: `1 / HHI`, período a período.

    É o HHI lido na unidade em que a equipe pensa. Uma carteira com HHI 0,25
    tem 4 ativos efetivos, independentemente de quantas colunas não-zero ela
    tenha — o que expõe a diferença entre "50 ativos na carteira" e "50 ativos
    que de fato importam". Séries: o número efetivo cai quando a carteira
    concentra, mesmo sem nenhum ativo ser vendido.

    Args:
        pesos: DataFrame de pesos efetivos.

    Returns:
        Series indexada por data.
    """
    h = hhi(pesos)
    return h.map(lambda x: _dividir(1.0, x))


# --- capacidade -------------------------------------------------------------


def capacity(pesos, volume_medio_diario, participacao_max):
    """Tamanho máximo da carteira, em R$, dada uma PREMISSA de participação.

    ============================ LEIA ANTES DE USAR ========================
    ESTE NÚMERO DEPENDE DE UMA PREMISSA QUE NÃO É ESTIMADA — É DECLARADA.

    `participacao_max` é a fração do volume diário do papel que assumimos
    poder representar sem mover o preço (ex.: 0,10 = "não passo de 10% do
    volume diário do ativo"). Não há nada nos dados que determine esse número:
    ele sai de premissa de execução, e premissas diferentes dão capacidades
    proporcionalmente diferentes. Por isso ele é ARGUMENTO OBRIGATÓRIO, SEM
    DEFAULT — este módulo não escolhe premissa por ninguém, e um default aqui
    viraria, em duas semanas, "o número que o código usa".

    O RESULTADO É ORDEM DE GRANDEZA, NÃO NÚMERO PRECISO. Ele responde "esta
    estratégia comporta milhões ou bilhões?", não "esta estratégia comporta
    R$ 412,7 milhões". Reportá-lo com precisão de centavo seria emprestar à
    premissa uma exatidão que ela não tem.
    ========================================================================

    O QUE FAZ: em cada data, `min_i (participacao_max * ADV_i / w_i)` sobre os
    ativos com peso não nulo. A leitura é: o ativo que amarra a capacidade é o
    que tem a pior razão entre liquidez e peso — não necessariamente o menos
    líquido, nem necessariamente o de maior peso.

    O QUE NÃO FAZ, e cada item destes empurra a capacidade real para BAIXO:
      - é baseada em POSIÇÃO, não em giro: mede o tamanho da posição que cabe
        em `participacao_max` de um dia de volume, e não o tamanho do TRADE de
        rebalanceamento. A versão por giro exigiria a série de turnover por
        ativo e não está implementada;
      - não modela impacto de mercado, spread, nem a possibilidade de espalhar
        a ordem por vários dias (que aumentaria a capacidade);
      - usa o volume medido no PASSADO da janela como proxy do volume
        disponível no futuro;
      - ignora que sair de uma posição em estresse costuma ser mais caro que
        entrar nela em calmaria.

    Args:
        pesos: DataFrame de pesos (datas × ativos).
        volume_medio_diario: volume financeiro diário por ativo, em R$. Ou
            DataFrame alinhado a `pesos` (datas × ativos), ou Series por ativo
            (constante no tempo). No projeto, o proxy natural é a MEDIANA de
            `VOLTOT` da janela de elegibilidade — a mesma que ordena o cap de
            50 (`decisoes/01`, 11/08/2026), o que mantém uma régua só.
        participacao_max: fração do volume diário assumida como executável, em
            (0, 1]. **OBRIGATÓRIO, SEM DEFAULT.**

    Returns:
        Series de capacidade em R$, indexada por data.

    Raises:
        ValueError: se `participacao_max` não for passado, for None, ou estiver
            fora de (0, 1].
    """
    if participacao_max is None:
        raise ValueError(
            "participacao_max é OBRIGATÓRIO e não tem default. É a premissa de "
            "execução — a fração do volume diário do papel que assumimos poder "
            "representar sem mover o preço. Nenhum dado deste projeto determina "
            "esse número: ele é declarado por quem reporta, e a capacidade "
            "escala linearmente com ele. Passe o valor explicitamente e declare-o "
            "no relatório junto com o resultado."
        )
    if not np.isscalar(participacao_max) or not (0.0 < float(participacao_max) <= 1.0):
        raise ValueError(
            f"participacao_max deve ser fração em (0, 1], recebido "
            f"{participacao_max!r}. Ex.: 0.10 para 'não passo de 10% do volume "
            f"diário'."
        )
    if not isinstance(pesos, pd.DataFrame):
        raise TypeError(f"pesos precisa ser DataFrame, recebido {type(pesos).__name__}.")
    if len(pesos) == 0:
        raise ValueError("pesos está vazio: não há carteira para medir.")

    p = float(participacao_max)
    w = pesos.fillna(0.0).astype(float)

    if isinstance(volume_medio_diario, pd.DataFrame):
        adv = volume_medio_diario.reindex(index=w.index, columns=w.columns)
    else:
        adv = _serie(volume_medio_diario, "volume_medio_diario")
        faltam = [c for c in w.columns if c not in adv.index]
        if faltam:
            raise ValueError(
                f"volume_medio_diario não cobre {len(faltam)} ativo(s) com peso, "
                f"o primeiro sendo {faltam[0]!r}. Sem volume não há como dizer "
                f"quanto daquele papel cabe na carteira, e supor um volume seria "
                f"inventar liquidez."
            )
        adv = pd.DataFrame(
            np.tile(adv.reindex(w.columns).to_numpy(), (len(w), 1)),
            index=w.index, columns=w.columns,
        )

    matriz_w = w.to_numpy()
    matriz_adv = adv.to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        limite = np.where(matriz_w > 0, p * matriz_adv / matriz_w, np.inf)

    # Ativo com peso > 0 e volume ausente/zero torna a capacidade indefinida
    # naquela data: não é "capacidade infinita", é "não medimos".
    limite = np.where(np.isnan(limite), np.nan, limite)
    capacidade = np.nanmin(limite, axis=1) if limite.size else np.array([])
    indefinida = np.isnan(limite).any(axis=1)
    capacidade = np.where(indefinida, np.nan, capacidade)
    capacidade = np.where(np.isinf(capacidade), np.nan, capacidade)

    return pd.Series(capacidade, index=w.index)


# --- consolidação -----------------------------------------------------------


def metricas_de_uma_carteira(nome, riqueza, retornos, taxa_livre_risco,
                             periodos_por_ano: int, alpha: float,
                             riqueza_inicial: float,
                             excluir_montagem_inicial: bool,
                             retornos_benchmark=None, turnover=None, pesos=None):
    """Uma linha de tabela para uma carteira. Ver `tabela_metricas()`.

    As colunas contra benchmark só aparecem se `retornos_benchmark` for
    passado; as de giro, só com `turnover`; as de concentração, só com
    `pesos`. Coluna ausente é mais honesto que coluna com NaN, porque NaN
    numa tabela de relatório se lê como "mediu e não deu", e aqui o caso é
    "não foi medido".
    """
    r = _serie(retornos, "retornos")
    w = _serie(riqueza, "riqueza")
    p = _validar_periodos(periodos_por_ano)
    a = _validar_alpha(alpha)

    linha = {
        "retorno_anualizado": retorno_anualizado(r, p),
        "volatilidade_anualizada": volatilidade_anualizada(r, p),
        "sharpe": sharpe(r, taxa_livre_risco, p),
        "sortino": sortino(r, taxa_livre_risco, p),
        "drawdown_maximo": drawdown_maximo(w, riqueza_inicial),
        "calmar": calmar(r, w, riqueza_inicial, p),
        f"var_{int(round(a * 100))}": var_historico(r, a),
        f"cvar_{int(round(a * 100))}": cvar_historico(r, a),
        "n_periodos": int(len(r)),
    }

    if retornos_benchmark is not None:
        linha["tracking_error"] = tracking_error(r, retornos_benchmark, p)
        linha["information_ratio"] = information_ratio(r, retornos_benchmark, p)

    if turnover is not None:
        linha["turnover_medio"] = turnover_medio(turnover, excluir_montagem_inicial)
        linha["turnover_anualizado"] = turnover_anualizado(
            turnover, p, excluir_montagem_inicial
        )

    if pesos is not None:
        linha["hhi_medio"] = hhi_medio(pesos)
        linha["n_efetivo_medio"] = float(n_efetivo_ativos(pesos).mean())

    return pd.DataFrame([linha], index=pd.Index([nome], name="carteira"))


def tabela_metricas(carteiras, taxa_livre_risco, periodos_por_ano: int,
                    alpha: float, riqueza_inicial: float,
                    excluir_montagem_inicial: bool, retornos_benchmark=None):
    """Consolida todas as métricas em uma tabela única para o relatório.

    O QUE FAZ: uma linha por carteira, todas medidas com EXATAMENTE os mesmos
    parâmetros — mesma taxa livre de risco, mesmo `alpha`, mesma anualização.
    É essa uniformidade que torna as linhas comparáveis entre si, e é por isso
    que os parâmetros são da tabela e não de cada carteira.

    O QUE NÃO FAZ: não escolhe o benchmark, não escolhe `alpha`, não decide se
    a montagem inicial entra no giro médio. Não formata número, não arredonda
    e não ordena por desempenho — ordenar por resultado é o tipo de gesto que
    transforma tabela em argumento.

    Args:
        carteiras: dict `nome -> dict` com as chaves `riqueza` e `retornos`
            (obrigatórias) e, opcionalmente, `turnover` e `pesos`.
        taxa_livre_risco: CDI no mesmo passo das séries, aplicado
            identicamente a todas as carteiras.
        periodos_por_ano: observações por ano. Com passo semanal, 52.
        alpha: nível de confiança de VaR/CVaR. Sem default.
        riqueza_inicial: capital inicial usado no motor.
        excluir_montagem_inicial: ver `turnover_medio()`. Sem default.
        retornos_benchmark: Series do benchmark (1/N). Se None, as colunas de
            tracking error e information ratio não aparecem.

    Returns:
        DataFrame de uma linha por carteira, indexado pelo nome.
    """
    if not isinstance(carteiras, dict) or not carteiras:
        raise ValueError(
            "carteiras precisa ser um dict não vazio no formato "
            "{nome: {'riqueza': ..., 'retornos': ..., 'turnover': ..., 'pesos': ...}}."
        )

    linhas = []
    for nome, partes in carteiras.items():
        faltando = [c for c in ("riqueza", "retornos") if c not in partes]
        if faltando:
            raise ValueError(f"carteira {nome!r} não trouxe {faltando}.")
        linhas.append(metricas_de_uma_carteira(
            nome=nome,
            riqueza=partes["riqueza"],
            retornos=partes["retornos"],
            taxa_livre_risco=taxa_livre_risco,
            periodos_por_ano=periodos_por_ano,
            alpha=alpha,
            riqueza_inicial=riqueza_inicial,
            excluir_montagem_inicial=excluir_montagem_inicial,
            retornos_benchmark=retornos_benchmark,
            turnover=partes.get("turnover"),
            pesos=partes.get("pesos"),
        ))
    return pd.concat(linhas)
