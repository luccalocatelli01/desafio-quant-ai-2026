"""Motor de backtest.

Agnóstico de estratégia: recebe uma trilha de pesos já decidida por outro
módulo, aplica sobre os retornos, desconta custos de transação e devolve a
série de riqueza. O motor não sabe — e não deve saber — se os pesos vieram de
mínima variância, de ICVaR ou de 1/N. Isso mantém a comparação entre modelos
honesta: todos passam exatamente pelo mesmo cano.

CONVENÇÕES DESTE MÓDULO (todas verificáveis nas docstrings de cada função)
-------------------------------------------------------------------------

1. SEM LOOK-AHEAD, POR CONSTRUÇÃO. Uma linha de pesos indexada na data `D`
   representa uma decisão tomada com informação disponível ATÉ `D`, inclusive.
   A EXECUÇÃO acontece ao FECHAMENTO de `D`, aos preços de `D`, e o retorno
   passa a acruar a partir do pregão seguinte: a carteira captura o retorno
   do primeiro pregão POSTERIOR a `D`, que é o movimento do fechamento de `D`
   ao fechamento de `D+1`. Quem chama passa os pesos indexados na data da
   decisão e não desloca nada — o motor cuida disso, em um lugar só.

   Ver a seção CONVENÇÃO DE EXECUÇÃO na docstring de `_trilha()` para o
   detalhe, inclusive por que o preço que precisa existir para executar é o
   de `D` e não o de `D+1`.

2. DERIVA ENTRE REBALANCEAMENTOS. Entre duas datas de rebalanceamento os pesos
   NÃO ficam fixos: eles derivam com o retorno de cada ativo. O ativo que sobe
   passa a pesar mais na carteira sem que ninguém compre nada. Manter o peso
   fixo equivaleria a rebalancear todo pregão de graça, e subestimaria o
   turnover.

3. CAIXA É UMA COLUNA COMO OUTRA QUALQUER. O ativo de caixa (será o CDI) entra
   como mais uma coluna de `retornos`, com o seu próprio retorno. Não há
   tratamento especial, não há "resíduo não investido". Por isso os pesos
   precisam somar 1: o que não está em risco tem de estar explicitamente em
   caixa.

4. CUSTO É PARÂMETRO, EM BASIS POINTS, COM DEFAULT ZERO. O valor decidido pela
   equipe — 50 bps proporcionais por transação, nas três séries (`decisoes/01`,
   11/08/2026) — é o REGISTRADO, não uma constante embutida aqui: este módulo
   continua sem default próprio, no mesmo espírito do item 5 abaixo.

5. FREQUÊNCIA DE REBALANCEAMENTO É PARÂMETRO. Nada de frequência embutida no
   código: `gerar_datas_rebalanceamento()` recebe o calendário de pregões
   REAIS e a frequência desejada. A frequência decidida pela equipe é SEMANAL
   (`decisoes/01`, 11/08/2026) — mas ela é o valor REGISTRADO, não uma
   constante embutida aqui. Este módulo continua sem default.

6. ATIVO SEM PREÇO: MARCAÇÃO CONGELADA, NUNCA IMPUTAÇÃO. Quando um ativo
   DETIDO fica sem retorno, a posição é congelada — contribui zero no período
   — e o evento é registrado. Ver `_trilha()` para a regra inteira e para a
   distinção entre "marcar posição parada" (permitido) e "imputar dado"
   (proibido). Registrado em `decisoes/01`, 11/08/2026.

O QUE ESTE MÓDULO NÃO FAZ
-------------------------
Não calcula pesos. Não conhece mínima variância nem ICVaR. Não escolhe ativo,
não filtra universo, não estima covariância, não mede desempenho (isso é
`src/metricas.py`) e não lê dado de lugar nenhum — recebe tudo por argumento.
"""

import numpy as np
import pandas as pd

# Tolerância para a checagem de que os pesos somam 1. Não é parâmetro de
# modelagem: é folga numérica para erro de ponto flutuante de solver.
TOL_SOMA_PESOS = 1e-6


def gerar_datas_rebalanceamento(indice, frequencia):
    """Deriva as datas de rebalanceamento a partir do calendário de pregões.

    O QUE FAZ: escolhe, dentro do calendário REAL informado, as datas em que a
    carteira é redecidida. Para frequência de calendário, devolve o ÚLTIMO
    pregão efetivamente existente dentro de cada período — nunca uma data de
    calendário que não foi pregão.

    O QUE ASSUME: que `indice` é o calendário de pregões efetivamente
    observados, e não um intervalo de datas corridas. O motor nunca inventa
    pregão.

    O QUE NÃO FAZ: não decide a frequência. A frequência de rebalanceamento foi
    decidida como SEMANAL em `decisoes/01_decisoes_estruturais.md` (11/08/2026),
    mas essa decisão não foi embutida nesta função por escolha deliberada:
    `frequencia` continua argumento obrigatório, sem default. Também não garante que a
    última data devolvida seja utilizável: uma decisão tomada no último pregão
    da série não tem pregão seguinte para ser executada, e `aplicar_pesos()`
    a descarta reportando quantas foram descartadas.

    Args:
        indice: índice de datas dos pregões (DatetimeIndex, ou qualquer índice
            ordenável quando `frequencia` for inteiro).
        frequencia: ou um inteiro N — rebalanceia a cada N pregões, contados a
            partir do primeiro — ou um alias de período do pandas ("ME", "QE",
            "W-FRI", "YE"...), caso em que se toma o último pregão de cada
            período. Também aceita "D" como sinônimo de "todo pregão".

    Returns:
        Índice com as datas de rebalanceamento, subconjunto de `indice`.

    Raises:
        ValueError: se `indice` for vazio, se `frequencia` for inteiro menor
            que 1, ou se o alias de período não for reconhecido pelo pandas.
        TypeError: se `frequencia` for alias de período e `indice` não for
            DatetimeIndex.
    """
    idx = pd.Index(indice)
    if len(idx) == 0:
        raise ValueError("indice vazio: não há calendário de pregões.")
    idx = idx.drop_duplicates().sort_values()

    if isinstance(frequencia, (int, np.integer)) and not isinstance(frequencia, bool):
        if frequencia < 1:
            raise ValueError(f"frequencia inteira deve ser >= 1, recebido {frequencia}.")
        return idx[:: int(frequencia)]

    if not isinstance(frequencia, str):
        raise TypeError(
            f"frequencia deve ser int (a cada N pregões) ou str (alias de "
            f"período do pandas), recebido {type(frequencia).__name__}."
        )

    rotulo = frequencia.strip()
    if rotulo.upper() in ("D", "PREGAO", "PREGÃO", "DIARIO", "DIÁRIO"):
        return idx

    if not isinstance(idx, pd.DatetimeIndex):
        raise TypeError(
            f"frequencia={rotulo!r} é alias de período e exige um DatetimeIndex; "
            f"o índice recebido é {type(idx).__name__}. Converta o calendário "
            f"para datas ou use frequência inteira (a cada N pregões)."
        )

    serie = pd.Series(idx, index=idx)
    try:
        ultimos = serie.groupby(pd.Grouper(freq=rotulo)).max().dropna()
    except (ValueError, KeyError) as exc:
        raise ValueError(
            f"frequencia={rotulo!r} não é um alias de período reconhecido pelo "
            f"pandas ({exc!r}). Exemplos válidos: 'ME' (fim de mês), 'QE' "
            f"(fim de trimestre), 'W-FRI' (sexta), 'YE' (fim de ano)."
        ) from exc

    return pd.DatetimeIndex(pd.to_datetime(ultimos.to_numpy()))


def _validar_entradas(pesos, retornos):
    """Confere pesos e retornos antes de rodar. Falha alto, nunca em silêncio.

    Regras verificadas, todas com exceção explícita:
      - `retornos` não vazio e com índice estritamente crescente;
      - toda coluna de `pesos` existe em `retornos`;
      - nenhum peso ausente (NaN);
      - cada linha de pesos soma 1, dentro de TOL_SOMA_PESOS.

    A exigência de somar 1 é consequência da convenção de caixa: se a soma
    fosse livre, a fração não alocada renderia zero por omissão, o que é uma
    decisão de modelagem tomada em silêncio pelo motor. Caixa é coluna.
    """
    if not isinstance(retornos, pd.DataFrame) or not isinstance(pesos, pd.DataFrame):
        raise TypeError("pesos e retornos precisam ser DataFrames do pandas.")
    if len(retornos) == 0:
        raise ValueError("retornos vazio: não há pregão para simular.")
    if len(pesos) == 0:
        raise ValueError("pesos vazio: não há decisão de carteira para aplicar.")

    idx_ret = retornos.index
    if not idx_ret.is_monotonic_increasing or idx_ret.has_duplicates:
        raise ValueError(
            "o índice de retornos precisa ser estritamente crescente e sem datas "
            "repetidas — ordene e remova duplicatas antes de chamar o motor."
        )
    if not pesos.index.is_monotonic_increasing or pesos.index.has_duplicates:
        raise ValueError(
            "o índice de pesos precisa ser estritamente crescente e sem datas "
            "repetidas: duas decisões na mesma data são ambíguas."
        )

    faltantes = [c for c in pesos.columns if c not in retornos.columns]
    if faltantes:
        raise ValueError(
            f"há colunas em pesos que não existem em retornos: {faltantes}. "
            f"O motor não inventa série de retorno para ativo que não recebeu."
        )

    if pesos.isna().to_numpy().any():
        linhas = pesos.index[pesos.isna().any(axis=1)].tolist()
        raise ValueError(
            f"há pesos ausentes (NaN) nas datas {linhas[:5]}"
            f"{'...' if len(linhas) > 5 else ''}. Peso ausente não é zero: se a "
            f"intenção é não alocar, passe 0.0 explicitamente."
        )

    somas = pesos.sum(axis=1)
    ruins = somas[(somas - 1.0).abs() > TOL_SOMA_PESOS]
    if len(ruins):
        d0 = ruins.index[0]
        raise ValueError(
            f"os pesos precisam somar 1 em cada data (tolerância "
            f"{TOL_SOMA_PESOS:g}); em {d0} somam {ruins.iloc[0]:.10f}. "
            f"O que não está em risco tem de estar explicitamente na coluna de "
            f"caixa — o motor não trata resíduo não alocado, porque isso seria "
            f"decidir o retorno do caixa em silêncio."
        )

    # AUSÊNCIA ESTRUTURAL, não ausência temporária. A regra de ativo sem preço
    # cobre o papel que para de negociar e pode voltar; ela NÃO cobre o papel
    # que nunca teve preço nenhum na janela. Congelar uma posição que nunca
    # teve marcação inicial seria carregar um ativo sem preço de entrada — não
    # há "último preço observado" para liquidar depois.
    for coluna in pesos.columns:
        if (pesos[coluna] != 0).any() and retornos[coluna].isna().all():
            raise ValueError(
                f"o ativo {coluna!r} recebe peso não-zero mas NÃO tem um único "
                f"retorno na janela inteira. Isso é ausência ESTRUTURAL de "
                f"dado, não ausência temporária de preço: a regra de "
                f"congelamento cobre o papel que para de negociar e pode "
                f"voltar, e pressupõe um último preço observado para liquidar. "
                f"Um ativo que nunca negociou não tem esse preço. Tire-o do "
                f"universo ou explique de onde viria a marcação inicial."
            )


def _alinhar_caixa(retornos_caixa, datas):
    """Alinha a série de caixa ao calendário do motor. Devolve array ou None.

    Aceita Series ou DataFrame de uma coluna só. Não preenche nada: data que
    a série de caixa não cobre vira NaN, e o NaN só vira erro se e quando um
    congelamento precisar daquela data (ver `_exigir_caixa`).
    """
    if retornos_caixa is None:
        return None
    if isinstance(retornos_caixa, pd.DataFrame):
        if retornos_caixa.shape[1] != 1:
            raise ValueError(
                f"retornos_caixa como DataFrame precisa ter exatamente 1 "
                f"coluna, recebido {retornos_caixa.shape[1]}. A série de caixa "
                f"é uma só — se há mais de uma, qual delas remunera a "
                f"liquidação não está declarado."
            )
        retornos_caixa = retornos_caixa.iloc[:, 0]
    if not isinstance(retornos_caixa, pd.Series):
        raise TypeError(
            f"retornos_caixa precisa ser Series ou DataFrame de 1 coluna, "
            f"recebido {type(retornos_caixa).__name__}."
        )
    return retornos_caixa.reindex(datas).to_numpy(dtype=float)


def _exigir_caixa(caixa, t, datas, ativo):
    """Falha alto se a série de caixa não cobre um período de congelamento.

    A regra de ativo sem preço termina, no pior caso, em liquidação contra
    caixa remunerado a CDI. Sem série de caixa cobrindo o período, esse
    desfecho não tem para onde ir — e inventar um retorno de caixa seria
    exatamente o tipo de decisão em silêncio que o motor não toma.
    """
    if caixa is None:
        raise ValueError(
            f"o ativo {ativo!r} entrou em congelamento em {datas[t]!r} — está "
            f"detido e sem retorno disponível. A regra de ativo sem preço "
            f"(decisoes/01, 11/08/2026) termina, se ele não voltar até o "
            f"próximo rebalanceamento, em liquidação ao último preço contra "
            f"caixa remunerado a CDI. Sem `retornos_caixa`, esse desfecho não "
            f"tem para onde ir. Passe a série de caixa ao motor."
        )
    if np.isnan(caixa[t]):
        raise ValueError(
            f"a série de caixa NÃO cobre {datas[t]!r}, período em que o ativo "
            f"{ativo!r} está congelado. O motor não preenche buraco de série "
            f"de caixa: sem a taxa daquele período, a remuneração de uma "
            f"eventual liquidação seria inventada."
        )


def _evento_congelamento(ativo, datas, t_inicio, t_fim, peso, desfecho):
    """Monta um registro do log de congelamentos. Só formatação, sem lógica."""
    return {
        "ativo": ativo,
        "data_inicio": datas[t_inicio],
        "data_fim": datas[t_fim] if t_fim is not None else pd.NaT,
        "peso_no_inicio": float(peso),
        "periodos_congelado": int((t_fim if t_fim is not None else len(datas)) - t_inicio),
        "desfecho": desfecho,
    }


def _trilha(pesos, retornos, retornos_caixa=None):
    """Núcleo do motor: percorre os pregões aplicando pesos, deriva e turnover.

    É aqui que moram o deslocamento D->D+1, a deriva e a regra de ativo sem
    preço. As funções públicas `aplicar_pesos()`, `calcular_turnover()` e
    `rodar_backtest()` são camadas finas sobre esta, para que o cálculo exista
    em UM lugar só e não possa divergir entre elas.

    ---------------------------------------------------------------------
    CONVENÇÃO DE EXECUÇÃO — o motor implementa a (ii)
    ---------------------------------------------------------------------

    Duas convenções são possíveis e produzem números diferentes:

      (i)  a decisão usa informação até D e a EXECUÇÃO acontece ao preço de
           D+1. O primeiro retorno capturável seria r_{D+2}.
      (ii) a decisão usa informação até D e a execução acontece ao
           FECHAMENTO de D, aos preços de D; o retorno acrua a partir de
           D+1. O primeiro retorno capturado é r_{D+1}.

    **Este motor implementa a (ii).** Uma decisão indexada em D é aplicada
    na posição do primeiro pregão posterior a D e captura o retorno DESSA
    data — e `r_{D+1} = P_{D+1}/P_D − 1` é o movimento do fechamento de D ao
    fechamento de D+1. Capturar esse retorno inteiro só é possível se a
    posição foi assumida ao preço `P_D`, isto é, ao fechamento de D.

    Consequência prática, e é ela que decide a guarda de elegibilidade: o
    preço que precisa EXISTIR para uma decisão ser executável é o de **D**,
    não o de D+1. Condicionar a execução à existência de preço em D+1 seria
    exigir, no instante da ordem, uma informação que ainda é futuro — e
    filtrar o universo do otimizador por ela introduziria look-ahead na
    seleção de ativos, exatamente o vazamento que este módulo existe para
    impedir. Por isso `_aplicar_alvo()` testa a disponibilidade em D.

    Isso NÃO elimina o caso de ativo com preço em D e sem preço em D+1:
    suspensão overnight, ou o último pregão que ninguém sabia ser o último.
    Esse é evento exógeno, que nenhuma regra baseada em informação até D
    previne — e ele cai na REGRA DE ATIVO SEM PREÇO abaixo (congelamento),
    não em exceção.

    Quem fixa esta convenção nos testes é
    `test_deslocamento_d_mais_1_impede_look_ahead`, e especificamente a
    perna de CONTROLE (decisão em `d0` -> riqueza 2,0), não a armadilha
    (decisão em `d1` -> 1,0). A armadilha dá 1,0 sob as DUAS convenções e
    sozinha não distinguiria nada; é o controle que falharia sob a (i),
    porque a entrada ao preço de `d1` deixaria o +100% de `d1` para trás.

    ---------------------------------------------------------------------
    ATIVO SEM PREÇO — "MARCAR POSIÇÃO PARADA" NÃO É "IMPUTAR DADO"
    ---------------------------------------------------------------------

    Esta é a distinção que autoriza a regra abaixo, e ela precisa ficar
    explícita porque as duas coisas se parecem no código (as duas trocam um
    NaN por um número) e são opostas no significado:

      IMPUTAR DADO é inventar uma OBSERVAÇÃO que não houve e depois deixar
      essa observação inventada ENTRAR NA ESTIMAÇÃO — na covariância, na
      cauda do ICVaR, na média. É proibido aqui, sem exceção: uma matriz de
      covariância estimada sobre zeros imputados subestima risco, e a cauda
      de um ICVaR calculada sobre zeros imputados fica artificialmente
      curta. O motor NUNCA devolve retorno imputado para os modelos.

      MARCAR POSIÇÃO PARADA é reconhecer um FATO da carteira: um ativo que
      não negocia não muda de preço de mercado, então a posição detida nele
      não gera resultado naquele período. O zero aqui não é uma observação
      inventada de retorno do ativo — é a contabilidade correta de uma
      posição que ficou parada. É permitido, e é o único tratamento honesto:
      qualquer outro número seria pior.

    A diferença prática: o zero de congelamento existe SÓ dentro do motor,
    para marcar a carteira. Ele não é gravado em série de retorno, não
    alimenta estimador nenhum, e cada ocorrência é registrada no log de
    eventos justamente para que ninguém confunda uma coisa com a outra
    depois. Estimação continua vendo NaN, que é o que de fato houve.

    ---------------------------------------------------------------------
    A REGRA (decisoes/01_decisoes_estruturais.md, 11/08/2026)
    ---------------------------------------------------------------------

    1. Ativo DETIDO sem retorno -> marcação CONGELADA: contribui zero no
       período. O peso relativo continua derivando com o resto da carteira,
       porque o valor absoluto da posição é que ficou parado.
    2. Se o ativo volta a negociar, o retorno de volta bate INTEGRALMENTE em
       QUEM CARREGOU a posição durante a suspensão. Nada é suavizado,
       distribuído ou descartado.

       Isso vale INCLUSIVE quando a reabertura cai na mesma data em que um
       rebalanceamento passa a valer. Nesse caso o período é partido em duas
       etapas: primeiro o gap de reabertura é realizado sobre os pesos
       ANTIGOS — os que de fato carregaram o congelamento — e só então o
       alvo novo é aplicado, já sobre a riqueza corrigida. Rebalancear na
       mesma data não pode fazer o retorno evaporar, que era o defeito
       corrigido em 11/08/2026.
    3. Se chega uma data de rebalanceamento sem o ativo ter voltado, a
       posição é liquidada ao último preço observado (que é exatamente a
       marcação congelada) e o valor entra no rebalanceamento daquela data,
       redistribuído conforme os pesos-alvo — inclusive para a coluna de
       caixa, se for para lá que o alvo mandar. A liquidação só acontece
       para ativo que também não tinha preço em D: se havia preço em D, a
       posição foi simplesmente vendida ao fechamento de D, como qualquer
       outra, e não há o que liquidar ao último preço.
    4. Comportamento IDÊNTICO para qualquer trilha de pesos: o motor não
       sabe se veio de ICVaR, de mínima variância ou de 1/N.
    5. Todo congelamento e toda liquidação entram no log de eventos. Nada
       é silencioso.

    Por que esta regra e NÃO "liquidar ao último preço no momento do sumiço":
    liquidar no momento do sumiço embute look-ahead — vender no último
    pregão é vender sabendo que aquele pregão foi o último, informação que
    não existia naquele dia — e trata suspensão temporária como morte, de
    modo que papel que volta com gap negativo nunca atingiria a carteira.
    Isso é sistematicamente otimista.

    ---------------------------------------------------------------------
    O QUE CONTINUA SENDO ERRO (ausência ESTRUTURAL, não temporária)
    ---------------------------------------------------------------------

    A regra cobre ausência TEMPORÁRIA de preço. Estes casos não são
    cobertos e continuam levantando ValueError explícito:
      - ativo que recebe peso e nunca teve retorno na janela inteira (não
        há último preço observado para liquidar) — ver `_validar_entradas`;
      - pesos-alvo que atribuem peso não-zero a um ativo sem preço na DATA
        DA DECISÃO (D): comprar papel que não negociava quando a ordem foi
        colocada não é algo que a regra endereça;
      - série de caixa ausente ou sem cobrir o período de congelamento —
        ver `_exigir_caixa`.

    Ativo com peso EXATAMENTE zero e retorno NaN continua sendo não-evento:
    não é detido, não congela, não gera log.

    Args:
        pesos: DataFrame de pesos indexado pela data da DECISÃO.
        retornos: DataFrame de retornos simples dos ativos.
        retornos_caixa: Series (ou DataFrame de 1 coluna) com o retorno do
            caixa remunerado a CDI, no mesmo calendário. Obrigatória apenas
            se algum congelamento ocorrer; sem congelamento, o motor não a
            usa e ela pode ser None.

    Returns:
        dict com:
          pesos_efetivos: DataFrame dos pesos realmente detidos em cada pregão,
              já deslocados e já derivados (peso no INÍCIO do pregão);
          turnover: Series do giro cobrado em cada pregão;
          retornos_brutos: Series do retorno da carteira, antes de custo;
          rebal_descartados: nº de linhas de pesos sem pregão posterior;
          eventos_congelamento: DataFrame com um registro por congelamento
              (ativo, data_inicio, data_fim, peso_no_inicio,
              periodos_congelado, desfecho em {"retomada",
              "retomada_com_rebalanceamento", "liquidacao", "em_aberto"}).
              "retomada_com_rebalanceamento" marca a reabertura que caiu na
              mesma data em que um alvo passou a valer: o gap foi para quem
              carregou, mas a posição foi trocada logo em seguida, e chamar
              isso de "retomada" simples esconderia a diferença.
    """
    _validar_entradas(pesos, retornos)

    ativos = list(pesos.columns)
    ret = retornos[ativos]
    datas = ret.index

    # --- da data da DECISÃO para a data em que o alvo passa a VALER ----------
    # A execução é ao fechamento de D (convenção (ii), ver docstring). O alvo
    # portanto passa a valer — isto é, começa a acruar retorno — no primeiro
    # pregão POSTERIOR a D. searchsorted com side="right" devolve essa posição.
    # A posição da própria data da decisão é a anterior a essa.
    pos = datas.searchsorted(pesos.index, side="right")
    validas = pos < len(datas)
    rebal_descartados = int((~validas).sum())

    alvo_por_posicao = {}
    matriz_pesos = pesos.to_numpy(dtype=float)
    for i, (p, ok) in enumerate(zip(pos, validas)):
        if ok:
            # Se duas decisões passarem a valer no mesmo pregão, vale a última:
            # a decisão mais recente substitui a anterior antes de ser executada.
            alvo_por_posicao[int(p)] = matriz_pesos[i]

    if not alvo_por_posicao:
        raise ValueError(
            "nenhuma linha de pesos tem pregão posterior em que o alvo pudesse "
            "passar a valer. A execução é ao fechamento da data da decisão e o "
            "retorno só acrua a partir do pregão seguinte — uma decisão tomada "
            "no último pregão da série nunca chega a render nada."
        )

    matriz_ret = ret.to_numpy(dtype=float)
    n, k = matriz_ret.shape
    primeira = min(alvo_por_posicao)

    caixa = _alinhar_caixa(retornos_caixa, datas)

    pesos_efetivos = np.full((n, k), np.nan)
    turnover = np.zeros(n)
    retornos_brutos = np.full(n, np.nan)

    w = np.zeros(k)
    investido = False

    # Estado do congelamento, por ativo. -1 = não congelado.
    congelado_desde = np.full(k, -1, dtype=int)
    peso_no_inicio = np.zeros(k)
    eventos = []

    def _aplicar_alvo(t, w, indisponivel_t, indisponivel_D):
        """Executa o alvo que passa a valer em `t`. Devolve (pesos_novos, giro).

        A ordem é colocada ao FECHAMENTO de D (convenção (ii)), então é a
        disponibilidade de preço em **D** que decide o que pode ser comprado
        e o que precisa ser liquidado ao último preço observado.
        `indisponivel_t` entra só para identificar quem ainda não voltou.
        """
        alvo = alvo_por_posicao[t]

        # Caso NÃO coberto: comprar papel que não negociava quando a ordem foi
        # colocada. A checagem é em D, não em D+1 — checar D+1 exigiria, no
        # instante da ordem, informação que ainda é futuro.
        comprar_sem_preco = (alvo != 0.0) & indisponivel_D
        if comprar_sem_preco.any():
            i = int(np.flatnonzero(comprar_sem_preco)[0])
            raise ValueError(
                f"o alvo que passa a valer em {datas[t]!r} atribui peso "
                f"{alvo[i]:.6g} ao ativo {ativos[i]!r}, que NÃO tinha preço na "
                f"DATA DA DECISÃO. A execução acontece ao fechamento da data da "
                f"decisão, então é o preço DELA que precisa existir — assumir "
                f"posição a um preço que não existe seria inventar a marcação "
                f"de entrada. A regra de ativo sem preço cobre o papel que "
                f"estava detido e parou de negociar; ela não cobre COMPRAR um "
                f"papel que não negociava quando a ordem foi colocada. "
                f"Restrinja o universo do otimizador aos ativos com preço na "
                f"DATA DA DECISÃO (D) — informação observável em D, sem "
                f"look-ahead. NÃO filtre por preço em D+1: isso é futuro."
            )

        # Liquidação do que ainda está congelado. O valor é redistribuído
        # imediatamente pelos pesos-alvo desta data (decisoes/01, 11/08/2026):
        # a marcação congelada É o último preço observado, e o giro dessa
        # liquidação aparece no turnover como qualquer venda.
        # Só entra aqui quem também não tinha preço em D: com preço em D a
        # posição foi vendida normalmente ao fechamento de D, e não há o que
        # liquidar ao último preço.
        liquidar = (w != 0.0) & indisponivel_t & indisponivel_D
        for i in np.flatnonzero(liquidar):
            _exigir_caixa(caixa, t, datas, ativos[i])
            aberto = congelado_desde[i] >= 0
            eventos.append(_evento_congelamento(
                ativos[i], datas,
                int(congelado_desde[i]) if aberto else t, t,
                peso_no_inicio[i] if aberto else w[i],
                "liquidacao",
            ))
            congelado_desde[i] = -1

        # Giro = notional total negociado / patrimônio, convenção de duas
        # pontas: vender 10% de A e comprar 10% de B dá giro 0,20.
        # No primeiro rebalanceamento w é o vetor nulo, então o giro é a
        # montagem inicial da carteira (tipicamente 1,0). Isso é deliberado e
        # fica visível na série de turnover, para a equipe decidir se cobra
        # custo na montagem.
        return alvo.copy(), float(np.abs(alvo - w).sum())

    for t in range(primeira, n):
        r = matriz_ret[t]
        indisponivel = np.isnan(r)
        # Disponibilidade na DATA DA DECISÃO: a posição anterior a `t`. Só há
        # preço em D se o retorno de D existe. Antes do início da série não há
        # nada observável, e nada pode ser comprado.
        indisponivel_D = np.isnan(matriz_ret[t - 1]) if t > 0 else np.ones(k, dtype=bool)

        # O zero aqui é MARCAÇÃO DE POSIÇÃO PARADA, não imputação de dado: ele
        # não sai deste laço e não alimenta estimador nenhum. Ver a docstring.
        r_limpo = np.where(indisponivel, 0.0, r)

        rebalanceia = t in alvo_por_posicao
        voltaram = (congelado_desde >= 0) & (~indisponivel)
        # Quando a reabertura cai na data em que um alvo passa a valer, o
        # período é PARTIDO: o gap vai para quem carregou, e só depois o alvo
        # entra. Sem isso o retorno evapora (defeito corrigido em 11/08/2026).
        partir_periodo = rebalanceia and bool(voltaram.any())

        # --- (1) RETOMADA: o ativo voltou a ter preço ------------------------
        for i in np.flatnonzero(voltaram):
            eventos.append(_evento_congelamento(
                ativos[i], datas, int(congelado_desde[i]), t, peso_no_inicio[i],
                "retomada_com_rebalanceamento" if rebalanceia else "retomada",
            ))
            congelado_desde[i] = -1

        # --- (2) ETAPA A: o gap de reabertura bate em quem CARREGOU ----------
        g = 0.0
        if partir_periodo:
            r_gap = np.where(voltaram, r_limpo, 0.0)
            g = float(w @ r_gap)
            base_g = 1.0 + g
            if base_g <= 0.0:
                raise ValueError(
                    f"a carteira zerou ou ficou negativa em {datas[t]!r} no gap "
                    f"de reabertura (1 + retorno = {base_g:.6g}); a deriva de "
                    f"pesos não está definida a partir daí."
                )
            # Peso efetivo do período é o que de fato carregou o congelamento.
            pesos_efetivos[t] = w
            w = w * (1.0 + r_gap) / base_g

        # --- (3) ETAPA B: o alvo passa a valer -------------------------------
        if rebalanceia:
            w, turnover[t] = _aplicar_alvo(t, w, indisponivel, indisponivel_D)
            investido = True

        if not investido:
            continue

        # --- (4) CONGELAMENTO: abre ou continua ------------------------------
        # Ativo com peso EXATAMENTE zero e retorno ausente não é detido: não
        # congela, não gera log, não contamina a conta.
        congelando = (w != 0.0) & indisponivel
        for i in np.flatnonzero(congelando):
            if congelado_desde[i] < 0:
                congelado_desde[i] = t
                peso_no_inicio[i] = w[i]
            _exigir_caixa(caixa, t, datas, ativos[i])

        # --- (5) ETAPA C: o resto do período, sobre os pesos vigentes --------
        # Sem partição, isto é o período inteiro e a conta é idêntica à de
        # antes. Com partição, o gap já foi realizado na etapa A e não pode
        # ser contado de novo aqui.
        r_resto = np.where(voltaram, 0.0, r_limpo) if partir_periodo else r_limpo
        if not partir_periodo:
            pesos_efetivos[t] = w

        h = float(w @ r_resto)
        base_h = 1.0 + h
        if base_h <= 0.0:
            raise ValueError(
                f"a carteira zerou ou ficou negativa em {datas[t]!r} "
                f"(1 + retorno = {base_h:.6g}); a deriva de pesos não está "
                f"definida a partir daí. O motor para em vez de continuar com "
                f"peso sem significado."
            )
        retornos_brutos[t] = (1.0 + g) * base_h - 1.0

        # --- deriva: o peso acompanha o retorno do ativo, sem ninguém negociar
        w = w * (1.0 + r_resto) / base_h

    # Congelamento que nunca terminou dentro da janela é reportado como tal, e
    # não confundido com retomada nem com liquidação.
    for i in np.flatnonzero(congelado_desde >= 0):
        eventos.append(_evento_congelamento(
            ativos[i], datas, int(congelado_desde[i]), None, peso_no_inicio[i], "em_aberto"
        ))

    colunas_evento = [
        "ativo", "data_inicio", "data_fim", "peso_no_inicio",
        "periodos_congelado", "desfecho",
    ]
    log = pd.DataFrame(eventos, columns=colunas_evento)
    if len(log):
        log = log.sort_values(["data_inicio", "ativo"]).reset_index(drop=True)

    idx = datas
    return {
        "pesos_efetivos": pd.DataFrame(pesos_efetivos, index=idx, columns=ativos).dropna(how="all"),
        "turnover": pd.Series(turnover, index=idx).iloc[primeira:],
        "retornos_brutos": pd.Series(retornos_brutos, index=idx).dropna(),
        "rebal_descartados": rebal_descartados,
        "eventos_congelamento": log,
    }


def aplicar_pesos(pesos, retornos, retornos_caixa=None):
    """Combina a trilha de pesos com os retornos dos ativos.

    O QUE FAZ: devolve o retorno BRUTO da carteira em cada pregão, já com o
    deslocamento D->D+1, já com a deriva de pesos entre rebalanceamentos e já
    com a regra de ativo sem preço (congelamento / liquidação).

    O QUE ASSUME: pesos indexados na DATA DA DECISÃO (não na data de execução),
    somando 1 em cada linha, com todas as colunas presentes em `retornos`.

    O QUE NÃO FAZ: não desconta custo (isso é `aplicar_custos()`), não acumula
    riqueza (isso é `serie_riqueza()`), não desloca nada por fora — se quem
    chama já deslocou os pesos, o atraso vira dois pregões — e não devolve o
    log de congelamentos (para isso use `rodar_backtest()`).

    Args:
        pesos: DataFrame de pesos, indexado por data de decisão, uma coluna
            por ativo.
        retornos: DataFrame de retornos simples, indexado por data, contendo
            ao menos as colunas de `pesos`.
        retornos_caixa: Series de retorno do caixa a CDI. Obrigatória só se
            houver congelamento; ver `_trilha()`.

    Returns:
        Series de retornos brutos da carteira, começando no primeiro pregão em
        que há posição.

    Raises:
        ValueError: nas condições descritas em `_trilha()` e
            `_validar_entradas()`.
    """
    return _trilha(pesos, retornos, retornos_caixa)["retornos_brutos"]


def calcular_turnover(pesos, retornos, retornos_caixa=None):
    """Calcula o giro efetivamente negociado em cada rebalanceamento.

    O QUE FAZ: compara os pesos-alvo com os pesos DERIVADOS imediatamente
    antes do rebalanceamento, e devolve o notional total negociado como fração
    do patrimônio, na convenção de duas pontas (vender 10% de A e comprar 10%
    de B dá 0,20).

    POR QUE PRECISA DOS RETORNOS: sem eles não há como saber quanto os pesos
    derivaram desde o último rebalanceamento, e o giro sairia da diferença
    entre pesos-alvo consecutivos. Isso superestima o giro — parte do
    movimento em direção ao novo alvo já foi feita de graça pela deriva do
    mercado. É por isso que esta função recebe `retornos`, ao contrário do que
    a assinatura original do esqueleto sugeria.

    O QUE NÃO FAZ: não aplica custo nenhum, não decide se a montagem inicial
    da carteira deve ser cobrada — apenas a reporta como o giro do primeiro
    rebalanceamento, visível na série, para a equipe decidir.

    LIQUIDAÇÃO DE ATIVO CONGELADO CONTA COMO GIRO: fechar ao último preço
    observado uma posição que parou de negociar é uma venda, e aparece no
    turnover como qualquer outra. Não é giro fantasma nem giro de graça.

    Args:
        pesos: DataFrame de pesos indexado por data de decisão.
        retornos: DataFrame de retornos simples.
        retornos_caixa: Series de retorno do caixa a CDI. Obrigatória só se
            houver congelamento; ver `_trilha()`.

    Returns:
        Series de turnover por pregão: zero nos pregões sem rebalanceamento e
        positiva nos pregões em que a carteira foi redecidida.
    """
    return _trilha(pesos, retornos, retornos_caixa)["turnover"]


def aplicar_custos(retornos_brutos, turnover, custo_bps: float = 0.0):
    """Desconta o custo de transação proporcional ao giro.

    O QUE FAZ: subtrai `turnover * custo_bps / 10_000` do retorno bruto no
    pregão em que o giro aconteceu.

    O QUE ASSUME: custo linear no notional negociado. Não modela spread,
    impacto de mercado, corretagem fixa nem lote mínimo.

    O QUE NÃO FAZ: não escolhe o valor do custo. O default é ZERO porque o
    custo não é obrigatório no desafio e a decisão é da equipe — ela entra em
    `decisoes/01_decisoes_estruturais.md` antes de virar número aqui.

    Args:
        retornos_brutos: Series de retorno bruto da carteira, saída de
            `aplicar_pesos()`.
        turnover: Series de giro, saída de `calcular_turnover()`, alinhada ao
            mesmo índice.
        custo_bps: custo em basis points sobre o notional negociado. 1 bp =
            0,01%. Default 0.0 (sem custo).

    Returns:
        Series de retornos líquidos da carteira.

    Raises:
        ValueError: se `custo_bps` for negativo, ou se os índices das duas
            séries não coincidirem.
    """
    if custo_bps < 0:
        raise ValueError(f"custo_bps não pode ser negativo, recebido {custo_bps}.")
    giro = turnover.reindex(retornos_brutos.index)
    if giro.isna().any():
        raise ValueError(
            "turnover não cobre todo o índice de retornos_brutos; as duas "
            "séries têm de vir da mesma execução do motor."
        )
    return retornos_brutos - giro * (custo_bps / 10_000.0)


def serie_riqueza(retornos_liquidos, riqueza_inicial: float = 1.0):
    """Acumula os retornos líquidos em uma série de riqueza.

    O QUE FAZ: capitaliza multiplicativamente, `W_t = W_{t-1} * (1 + r_t)`.

    O QUE ASSUME: que `retornos_liquidos` são retornos SIMPLES por pregão, na
    ordem cronológica.

    O QUE NÃO FAZ: não anualiza, não converte para log, não mede nada. Métrica
    é `src/metricas.py`.

    Args:
        retornos_liquidos: Series de retornos líquidos, saída de
            `aplicar_custos()`.
        riqueza_inicial: valor inicial da carteira. Default 1.0, que faz a
            série ser lida diretamente como fator acumulado.

    Returns:
        Series de riqueza acumulada, no mesmo índice dos retornos. O primeiro
        valor já reflete o retorno do primeiro pregão investido.

    Raises:
        ValueError: se `riqueza_inicial` não for positiva.
    """
    if riqueza_inicial <= 0:
        raise ValueError(f"riqueza_inicial deve ser > 0, recebido {riqueza_inicial}.")
    return riqueza_inicial * (1.0 + retornos_liquidos).cumprod()


def rodar_backtest(pesos, retornos, custo_bps: float = 0.0, riqueza_inicial: float = 1.0,
                   retornos_caixa=None):
    """Orquestra o backtest completo: pesos -> deriva -> custos -> riqueza.

    O QUE FAZ: encadeia as etapas em uma única passagem pelo núcleo do motor,
    e devolve também os componentes intermediários, para que qualquer número
    do relatório possa ser auditado sem reexecutar nada.

    O QUE ASSUME: as mesmas convenções do módulo — pesos na data da decisão,
    execução em D+1, deriva entre rebalanceamentos, caixa como coluna, e a
    regra de ativo sem preço descrita em `_trilha()`.

    O QUE NÃO FAZ: não calcula peso, não escolhe frequência, não escolhe custo,
    não mede desempenho e não sabe qual estratégia gerou os pesos — a regra de
    ativo sem preço vale igual para modelo e para 1/N.

    Args:
        pesos: DataFrame de pesos produzido por uma estratégia qualquer,
            indexado pela data da decisão.
        retornos: DataFrame de retornos simples dos ativos, incluindo a coluna
            de caixa se houver.
        custo_bps: custo de transação em basis points sobre o giro. Default 0.
        riqueza_inicial: valor inicial da carteira. Default 1.0.
        retornos_caixa: Series de retorno do caixa a CDI, usada quando um
            ativo detido fica sem preço. Obrigatória só se houver
            congelamento; ver `_trilha()`.

    Returns:
        dict com `riqueza`, `retornos_liquidos`, `retornos_brutos`,
        `turnover`, `pesos_efetivos`, `rebal_descartados` (nº de linhas de
        pesos que não tinham pregão posterior e por isso nunca valeram) e
        `eventos_congelamento` (log de congelamentos e liquidações).
    """
    t = _trilha(pesos, retornos, retornos_caixa)
    liquidos = aplicar_custos(t["retornos_brutos"], t["turnover"], custo_bps)
    return {
        "riqueza": serie_riqueza(liquidos, riqueza_inicial),
        "retornos_liquidos": liquidos,
        "retornos_brutos": t["retornos_brutos"],
        "turnover": t["turnover"],
        "pesos_efetivos": t["pesos_efetivos"],
        "rebal_descartados": t["rebal_descartados"],
        "eventos_congelamento": t["eventos_congelamento"],
    }
