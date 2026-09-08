"""Aquisição e preparação de dados.

Duas camadas, com regras diferentes:

1. CAMADA BRUTA (`dados/bruto/`) — a série histórica COTAHIST da B3, ano a
   ano, como a B3 publica. Cobre 1986 até o ano corrente, portanto atravessa
   a data de corte. É uma cópia literal da fonte: nenhum filtro, nenhum
   ajuste de preço, nenhuma métrica derivada. As funções desta camada não
   estão sujeitas a `DATA_CORTE`, porque baixar o arquivo anual publicado
   não é o mesmo que olhar para o período reservado — a partição ainda não
   foi feita e nenhum preço posterior ao corte foi lido por ninguém.

2. CAMADA DE DESENHO (`dados/desenho/`) — o que efetivamente alimenta os
   modelos. Aqui vale a trava: `carregar_desenho()` e companhia NUNCA
   devolvem dado posterior a `DATA_CORTE`. É também nesta camada — e só
   nela — que os EVENTOS SOCIETÁRIOS marcados no próprio arquivo
   (ex-grupamento, ex-bonificação, troca de fator de cotação) são
   NEUTRALIZADOS no painel de preços, com log obrigatório. Ver a seção
   "EVENTOS SOCIETÁRIOS" e `decisoes/01`, 12/08/2026. A camada bruta
   continua intocada.

Este módulo não lê, não lista e não infere nada sobre `dados/holdout/`. O
acesso ao holdout é exclusividade de `src/holdout.py`, chamado uma única vez
pelo notebook 99.
"""

import ast
import json
import shutil
import urllib.parse
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_CORTE, DATA_INICIO, DIR_DESENHO


def _validar_janela(data_inicio: str, data_fim: str) -> None:
    """Trava de proteção do holdout.

    Levanta ValueError se qualquer ponta da janela pedida ultrapassar
    DATA_CORTE. Esta é a única parte de `dados.py` que já está implementada:
    ela precisa funcionar desde o primeiro dia, antes de existir qualquer
    outra coisa, porque é ela que impede um `carregar_desenho()` distraído
    de contaminar o período reservado.

    Args:
        data_inicio: data inicial pedida, formato ISO "YYYY-MM-DD".
        data_fim: data final pedida, formato ISO "YYYY-MM-DD".

    Raises:
        ValueError: se `data_inicio` ou `data_fim` for posterior a DATA_CORTE.
    """
    for rotulo, valor in (("data_inicio", data_inicio), ("data_fim", data_fim)):
        if valor > DATA_CORTE:
            raise ValueError(
                f"VIOLAÇÃO DO PROTOCOLO DE HOLDOUT: {rotulo}={valor} é posterior "
                f"a DATA_CORTE={DATA_CORTE}. O período após a data de corte é "
                f"holdout e só pode ser lido por src/holdout.py, uma única vez, "
                f"a partir de 13/08, com decisoes/02_pre_registro.md preenchido. "
                f"Se você acha que precisa desse dado agora, você não precisa."
            )


def carregar_desenho(tickers, data_inicio: str = DATA_INICIO, data_fim: str = DATA_CORTE,
                     dir_desenho=DIR_DESENHO):
    """Carrega os preços/retornos do período de desenho.

    CONTRATO: esta função NUNCA retorna dado posterior a DATA_CORTE.
    Não importa o que for pedido no argumento `data_fim`: se for além da data
    de corte, a função levanta ValueError em vez de truncar silenciosamente.
    Truncar em silêncio seria pior do que falhar, porque quem chamou acharia
    que recebeu o que pediu.

    Esta função também não tem visibilidade sobre `dados/holdout/`: ela lê
    exclusivamente de `dados/desenho/`.

    Args:
        tickers: iterável de códigos dos ativos do universo.
        data_inicio: início da janela, ISO "YYYY-MM-DD".
        data_fim: fim da janela, ISO "YYYY-MM-DD". Default e teto: DATA_CORTE.

    Returns:
        DataFrame indexado por data, uma coluna por ticker.

    Raises:
        ValueError: se a janela pedida ultrapassar DATA_CORTE.
    """
    _validar_janela(data_inicio, data_fim)
    painel = carregar_painel(dir_desenho, data_inicio, data_fim)
    alvo = list(dict.fromkeys(tickers))  # preserva ordem, remove repetido
    sub = painel[painel["CODISI"].isin(set(alvo))]
    precos = sub.pivot_table(
        index="data", columns="CODISI", values="PREULT", aggfunc="last"
    )
    return precos.reindex(columns=alvo).sort_index()


def baixar_precos(tickers, data_inicio: str, data_fim: str):
    """Baixa preços ajustados da fonte externa e grava em `dados/desenho/`.

    Args:
        tickers: iterável de códigos dos ativos.
        data_inicio: início da janela, ISO "YYYY-MM-DD".
        data_fim: fim da janela, ISO "YYYY-MM-DD".

    Returns:
        DataFrame de preços ajustados.
    """
    raise NotImplementedError


def calcular_retornos(precos):
    """Converte série de preços em série de retornos simples.

    BASE DO RETORNO: o ÚLTIMO PREÇO OBSERVADO, não o preço do pregão
    anterior. Isso NÃO é imputação e NÃO é forward-fill da série de preços:
    nenhuma observação inventada entra em lugar nenhum. O numerador é sempre
    o preço EFETIVAMENTE observado na data (e continua NaN quando o ativo não
    negociou); só o DENOMINADOR usa o último preço observado antes dela.

    É essa escolha que faz a regra de ativo sem preço funcionar como está
    registrada em `decisoes/01` (11/08/2026): durante a suspensão o retorno é
    NaN, o motor congela a marcação e a posição contribui zero; na reabertura
    o retorno é `P_reabertura / P_último_observado − 1`, ou seja, o gap
    INTEGRAL da suspensão, que é exatamente o que a regra 2 manda bater em
    quem carregou a posição. Usar o pregão anterior como base produziria NaN
    também na reabertura, e o gap nunca chegaria à carteira.

    "Último preço observado" é, aliás, o mesmo conceito que a regra 3 usa para
    liquidar posição congelada — é uma definição só, aplicada nos dois lugares.

    Args:
        precos: DataFrame de preços indexado por data, uma coluna por ativo.

    Returns:
        DataFrame de retornos simples, sem a primeira linha (que não tem base).
    """
    base = precos.ffill().shift(1)
    return (precos / base - 1.0).iloc[1:]


def carregar_benchmark(data_inicio: str = DATA_INICIO, data_fim: str = DATA_CORTE):
    """Carrega a série do benchmark no período de desenho.

    Sujeita ao mesmo contrato de `carregar_desenho()`.

    Raises:
        ValueError: se a janela pedida ultrapassar DATA_CORTE.
    """
    _validar_janela(data_inicio, data_fim)
    raise NotImplementedError


def carregar_caixa(data_inicio: str = DATA_INICIO, data_fim: str = DATA_CORTE,
                   dir_desenho=DIR_DESENHO):
    """Carrega a série DIÁRIA do ativo de caixa (CDI) no período de desenho.

    Sujeita ao mesmo contrato de `carregar_desenho()`: nunca devolve dado
    posterior a `DATA_CORTE`, e levanta erro em vez de truncar em silêncio.

    A ÚNICA transformação aplicada é `valor / 100`. O SGS publica a série 12
    em **percentual ao dia** (`0,0525` significa 0,0525% naquele dia); dividir
    por 100 converte para retorno decimal, que é a unidade em que o motor e as
    métricas trabalham. Isso é **leitura de unidade, não ajuste de taxa** — a
    mesma categoria da vírgula implícita do COTAHIST. Nada é anualizado, nada
    é composto e nenhuma data é preenchida aqui.

    QUAL É O PAPEL DESTA SÉRIE (`decisoes/01`, 11/08/2026): o CDI tem três
    papéis e esta função serve aos dois que existem. **(2) contabilidade do
    motor** — é a taxa do caixa residual e a remuneração de valor liquidado de
    posição congelada; **(3) métricas** — é a taxa livre de risco, aplicada
    identicamente às três séries. O papel **(1), variável de decisão, NÃO
    existe**: o CDI está FORA do universo de otimização, e esta função não o
    coloca em lugar nenhum de onde um otimizador possa alocá-lo.

    Args:
        data_inicio: início da janela, ISO "AAAA-MM-DD".
        data_fim: fim da janela, ISO "AAAA-MM-DD". Default e teto: DATA_CORTE.
        dir_desenho: pasta do período de desenho, onde `particionar_cdi()`
            gravou `cdi_sgs12.parquet`.

    Returns:
        Series de retorno decimal DIÁRIO do caixa, indexada por data
        (datetime64), ordenada e sem repetição.

    Raises:
        ValueError: se a janela pedida ultrapassar DATA_CORTE, ou se o parquet
            do CDI não existir na pasta de desenho.
    """
    _validar_janela(data_inicio, data_fim)

    caminho = Path(dir_desenho) / "cdi_sgs12.parquet"
    if not caminho.exists():
        raise ValueError(
            f"não encontrei {caminho}. A série do CDI é baixada por "
            f"`baixar_cdi_sgs12()` e particionada por `particionar_cdi()` — ver "
            f"notebooks/00_dados.ipynb. Sem ela o motor não tem taxa para o "
            f"caixa residual nem para a liquidação de posição congelada, e as "
            f"métricas não têm taxa livre de risco."
        )

    serie = pd.read_parquet(caminho, columns=["data", "valor"])
    serie = serie[
        (serie["data"] >= pd.Timestamp(data_inicio))
        & (serie["data"] <= pd.Timestamp(data_fim))
    ]
    serie = serie.drop_duplicates(subset="data").sort_values("data")
    return pd.Series(
        (serie["valor"].astype(float) / 100.0).to_numpy(),
        index=pd.DatetimeIndex(serie["data"]),
        name="caixa",
    )


# =============================================================================
# CAMADA BRUTA — COTAHIST (Série Histórica de Cotações da B3)
# =============================================================================
#
# Fonte: https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A<AAAA>.ZIP
# Layout: SeriesHistoricas_Layout.pdf, revisão 02, 05/10/2020.
#
# REGRAS DESTA CAMADA (não negociáveis):
#   - Nenhuma linha é filtrada. Todos os códigos BDI, tipos de mercado,
#     especificações e tickers entram.
#   - Nenhum preço é ajustado por provento, inflação, desdobramento ou fator
#     de cotação. FATCOT é preservado como coluna, não aplicado.
#   - Nenhuma linha duplicada, outlier ou preço zerado é removido.
#   - Nenhuma métrica derivada de preço é calculada.
#   - Os valores estão na MOEDA DA ÉPOCA (ver campo MODREF). Nada é convertido.
#
# A ÚNICA leitura de formato aplicada aos preços é a vírgula implícita: o
# layout declara os campos de preço como (11)V99, ou seja, inteiro com 2 casas
# decimais implícitas — 1234 no arquivo significa 12,34. Dividir por 100 é ler
# o formato corretamente, não ajustar o preço. O mesmo vale para VOLTOT
# ((16)V99, 2 casas) e PTOEXE ((07)V06, 6 casas).
# =============================================================================

URL_COTAHIST = "https://bvmf.bmfbovespa.com.br/InstDados/SerHist/COTAHIST_A{ano}.ZIP"
ANO_INICIAL_B3 = 1986
TAMANHO_REGISTRO = 245

# Cada campo: (nome, pos_inicial, pos_final, tipo, casas_decimais)
# Posições são 1-based e inclusivas, exatamente como no PDF do layout.
# tipo: "X" = alfanumérico, "N" = numérico.

LAYOUT_HEADER = [
    ("TIPREG", 1, 2, "N", 0),
    ("NOME_ARQUIVO", 3, 15, "X", 0),
    ("COD_ORIGEM", 16, 23, "X", 0),
    ("DATA_GERACAO", 24, 31, "N", 0),
    ("RESERVA", 32, 245, "X", 0),
]

LAYOUT_DETALHE = [
    ("TIPREG", 1, 2, "N", 0),
    ("DATA_PREGAO", 3, 10, "N", 0),
    ("CODBDI", 11, 12, "X", 0),
    ("CODNEG", 13, 24, "X", 0),
    ("TPMERC", 25, 27, "N", 0),
    ("NOMRES", 28, 39, "X", 0),
    ("ESPECI", 40, 49, "X", 0),
    ("PRAZOT", 50, 52, "X", 0),
    ("MODREF", 53, 56, "X", 0),
    ("PREABE", 57, 69, "N", 2),
    ("PREMAX", 70, 82, "N", 2),
    ("PREMIN", 83, 95, "N", 2),
    ("PREMED", 96, 108, "N", 2),
    ("PREULT", 109, 121, "N", 2),
    ("PREOFC", 122, 134, "N", 2),
    ("PREOFV", 135, 147, "N", 2),
    ("TOTNEG", 148, 152, "N", 0),
    ("QUATOT", 153, 170, "N", 0),
    ("VOLTOT", 171, 188, "N", 2),
    ("PREEXE", 189, 201, "N", 2),
    ("INDOPC", 202, 202, "N", 0),
    ("DATVEN", 203, 210, "N", 0),
    ("FATCOT", 211, 217, "N", 0),
    ("PTOEXE", 218, 230, "N", 6),
    ("CODISI", 231, 242, "X", 0),
    ("DISMES", 243, 245, "N", 0),
]

LAYOUT_TRAILER = [
    ("TIPREG", 1, 2, "N", 0),
    ("NOME_ARQUIVO", 3, 15, "X", 0),
    ("COD_ORIGEM", 16, 23, "X", 0),
    ("DATA_GERACAO", 24, 31, "N", 0),
    ("TOTAL_REGISTROS", 32, 42, "N", 0),
    ("RESERVA", 43, 245, "X", 0),
]

# Tabelas anexas do PDF, transcritas literalmente. Servem SÓ para rotular o
# inventário — nenhuma delas é usada para filtrar coisa alguma.

TABELA_CODBDI = {
    "02": "LOTE PADRAO",
    "05": "SANCIONADAS PELOS REGULAMENTOS BMFBOVESPA",
    "06": "CONCORDATARIAS",
    "07": "RECUPERACAO EXTRAJUDICIAL",
    "08": "RECUPERACAO JUDICIAL",
    "09": "RAET - REGIME DE ADMINISTRACAO ESPECIAL TEMPORARIA",
    "10": "DIREITOS E RECIBOS",
    "11": "INTERVENCAO",
    "12": "FUNDOS IMOBILIARIOS",
    "14": "CERT.INVEST/TIT.DIV.PUBLICA",
    "18": "OBRIGACOES",
    "22": "BONUS (PRIVADOS)",
    "26": "APOLICES/BONUS/TITULOS PUBLICOS",
    "32": "EXERCICIO DE OPCOES DE COMPRA DE INDICES",
    "33": "EXERCICIO DE OPCOES DE VENDA DE INDICES",
    "38": "EXERCICIO DE OPCOES DE COMPRA",
    "42": "EXERCICIO DE OPCOES DE VENDA",
    "46": "LEILAO DE NAO COTADOS",
    "48": "LEILAO DE PRIVATIZACAO",
    "49": "LEILAO DO FUNDO RECUPERACAO ECONOMICA ESPIRITO SANTO",
    "50": "LEILAO",
    "51": "LEILAO FINOR",
    "52": "LEILAO FINAM",
    "53": "LEILAO FISET",
    "54": "LEILAO DE ACOES EM MORA",
    "56": "VENDAS POR ALVARA JUDICIAL",
    "58": "OUTROS",
    "60": "PERMUTA POR ACOES",
    "61": "META",
    "62": "MERCADO A TERMO",
    "66": "DEBENTURES COM DATA DE VENCIMENTO ATE 3 ANOS",
    "68": "DEBENTURES COM DATA DE VENCIMENTO MAIOR QUE 3 ANOS",
    "70": "FUTURO COM RETENCAO DE GANHOS",
    "71": "MERCADO DE FUTURO",
    "74": "OPCOES DE COMPRA DE INDICES",
    "75": "OPCOES DE VENDA DE INDICES",
    "78": "OPCOES DE COMPRA",
    "82": "OPCOES DE VENDA",
    "83": "BOVESPAFIX",
    "84": "SOMA FIX",
    "90": "TERMO VISTA REGISTRADO",
    "96": "MERCADO FRACIONARIO",
    "99": "TOTAL GERAL",
}

TABELA_TPMERC = {
    10: "VISTA",
    12: "EXERCICIO DE OPCOES DE COMPRA",
    13: "EXERCICIO DE OPCOES DE VENDA",
    17: "LEILAO",
    20: "FRACIONARIO",
    30: "TERMO",
    50: "FUTURO COM RETENCAO DE GANHO",
    60: "FUTURO COM MOVIMENTACAO CONTINUA",
    70: "OPCOES DE COMPRA",
    80: "OPCOES DE VENDA",
}

TABELA_INDOPC = {
    1: "US$ - CORRECAO PELA TAXA DO DOLAR",
    2: "TJLP - CORRECAO PELA TJLP",
    8: "IGPM - CORRECAO PELO IGP-M (OPCOES PROTEGIDAS)",
    9: "URV - CORRECAO PELA URV",
}


# --- Passo 2: download bruto ------------------------------------------------


def anos_disponiveis(ano_final: int):
    """Lista os anos da série, do primeiro publicado pela B3 até `ano_final`.

    Args:
        ano_final: último ano a incluir (inclusive).

    Returns:
        Lista de inteiros.
    """
    return list(range(ANO_INICIAL_B3, ano_final + 1))


def baixar_cotahist_ano(ano: int, destino, forcar: bool = False) -> dict:
    """Baixa o .zip anual do COTAHIST, sem descompactar.

    O .zip é gravado como veio da B3 e não é tocado por nada depois — é a
    fonte de verdade do projeto. Se o arquivo já existir com o mesmo tamanho
    anunciado pelo servidor, o download é pulado.

    Args:
        ano: ano da série.
        destino: pasta onde gravar o .zip.
        forcar: se True, rebaixa mesmo que o arquivo já exista.

    Returns:
        Dicionário com ano, caminho, bytes, status e erro (se houver).
    """
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    url = URL_COTAHIST.format(ano=ano)
    caminho = destino / f"COTAHIST_A{ano}.ZIP"

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            tamanho_remoto = int(resp.headers.get("Content-Length") or 0)
            if caminho.exists() and not forcar and caminho.stat().st_size == tamanho_remoto:
                return {
                    "ano": ano,
                    "caminho": str(caminho),
                    "bytes": caminho.stat().st_size,
                    "status": "ja_existia",
                    "erro": "",
                }
            with open(caminho, "wb") as f:
                shutil.copyfileobj(resp, f, length=1024 * 1024)
    except Exception as exc:  # reportar, não contornar
        return {"ano": ano, "caminho": str(caminho), "bytes": 0, "status": "FALHA", "erro": repr(exc)}

    tamanho = caminho.stat().st_size
    if not zipfile.is_zipfile(caminho):
        return {
            "ano": ano,
            "caminho": str(caminho),
            "bytes": tamanho,
            "status": "FALHA",
            "erro": "arquivo baixado não é um zip válido",
        }
    return {"ano": ano, "caminho": str(caminho), "bytes": tamanho, "status": "ok", "erro": ""}


def baixar_cotahist(anos, destino, forcar: bool = False):
    """Baixa vários anos. Um ano que falha é reportado, não abortado.

    Args:
        anos: iterável de anos.
        destino: pasta onde gravar os .zip.
        forcar: repassa para `baixar_cotahist_ano`.

    Returns:
        DataFrame com uma linha por ano.
    """
    return pd.DataFrame([baixar_cotahist_ano(a, destino, forcar) for a in anos])


# --- Passo 3: parse sem filtro ----------------------------------------------


def _quebrar_registros(bruto: bytes):
    """Quebra o conteúdo do .TXT em registros de 245 bytes.

    Trata os dois casos possíveis: arquivo com terminadores de linha
    (CRLF ou LF) e arquivo em blocos fixos sem terminador.

    Returns:
        (registros_validos, registros_fora_do_tamanho) — o segundo item é uma
        lista de (indice, tamanho) para reportar, nunca para corrigir.
    """
    if b"\n" not in bruto:
        linhas = [bruto[i : i + TAMANHO_REGISTRO] for i in range(0, len(bruto), TAMANHO_REGISTRO)]
    else:
        linhas = bruto.split(b"\n")
        if linhas and linhas[-1] == b"":
            linhas.pop()
        linhas = [ln[:-1] if ln.endswith(b"\r") else ln for ln in linhas]

    validos, fora = [], []
    for i, ln in enumerate(linhas):
        if len(ln) == TAMANHO_REGISTRO:
            validos.append(ln)
        else:
            fora.append((i, len(ln)))
    return validos, fora


def _matriz(registros):
    """Empilha os registros numa matriz (n, 245) de bytes."""
    return np.frombuffer(b"".join(registros), dtype=np.uint8).reshape(-1, TAMANHO_REGISTRO)


def _fatiar(buf, ini: int, fim: int):
    """Extrai um campo de largura fixa como texto latin-1, sem espaços nas pontas.

    O encoding é latin-1 / Windows ANSI, conforme a B3 publica. Ler como UTF-8
    corrompe os acentos dos nomes de empresa.
    """
    largura = fim - ini + 1
    sub = np.ascontiguousarray(buf[:, ini - 1 : fim])
    coluna = sub.view(f"S{largura}").reshape(-1)
    return pd.Series(coluna).str.decode("latin-1").str.strip()


def _montar(buf, layout):
    """Converte a matriz de bytes num DataFrame segundo o layout informado.

    Campos "X" viram texto. Campos "N" viram número; quando o layout declara
    casas decimais implícitas, o inteiro é dividido pela potência de 10
    correspondente — leitura do formato, não ajuste de preço.

    Returns:
        (DataFrame, anomalias) — `anomalias` lista campos numéricos com
        conteúdo não numérico e não vazio, para reportar.
    """
    colunas, anomalias = {}, []
    for nome, ini, fim, tipo, dec in layout:
        texto = _fatiar(buf, ini, fim)
        if tipo == "X":
            colunas[nome] = texto.astype("string")
            continue
        numero = pd.to_numeric(texto, errors="coerce")
        ruim = int((numero.isna() & (texto != "")).sum())
        if ruim:
            anomalias.append({"campo": nome, "registros_nao_numericos": ruim})
        colunas[nome] = numero / (10**dec) if dec else numero.astype("Int64")
    return pd.DataFrame(colunas), anomalias


def parse_cotahist_ano(caminho_zip, destino_parquet, dir_temp) -> dict:
    """Descompacta um ano, converte para parquet e devolve o relatório do parse.

    Não filtra nada. Os três tipos de registro são separados em arquivos
    distintos e nenhum é descartado:
      - detalhe (01)  -> destino_parquet/cotahist_<ano>.parquet
      - header  (00)  -> destino_parquet/meta/header_<ano>.parquet
      - trailer (99)  -> destino_parquet/meta/trailer_<ano>.parquet

    O .zip de origem não é modificado. A pasta temporária é limpa ao final.

    Args:
        caminho_zip: caminho do .zip original.
        destino_parquet: pasta de saída dos parquets.
        dir_temp: pasta temporária para o .TXT descompactado.

    Returns:
        Dicionário com contagens por tipo de registro, total declarado no
        trailer, registros fora do tamanho e anomalias de conversão.
    """
    caminho_zip = Path(caminho_zip)
    destino_parquet = Path(destino_parquet)
    (destino_parquet / "meta").mkdir(parents=True, exist_ok=True)
    dir_temp = Path(dir_temp)
    dir_temp.mkdir(parents=True, exist_ok=True)

    ano = int(caminho_zip.stem.split("_A")[-1])

    with zipfile.ZipFile(caminho_zip) as z:
        nomes = z.namelist()
        if len(nomes) != 1:
            return {"ano": ano, "status": "FALHA", "erro": f"zip com {len(nomes)} arquivos: {nomes}"}
        alvo = dir_temp / nomes[0]
        with z.open(nomes[0]) as origem, open(alvo, "wb") as saida:
            shutil.copyfileobj(origem, saida, length=8 * 1024 * 1024)

    try:
        bruto = alvo.read_bytes()
        registros, fora = _quebrar_registros(bruto)
        buf = _matriz(registros)

        tipo = _fatiar(buf, 1, 2)
        mask = {"00": (tipo == "00").to_numpy(), "01": (tipo == "01").to_numpy(), "99": (tipo == "99").to_numpy()}
        outros = int(len(tipo) - sum(int(m.sum()) for m in mask.values()))

        det, anomalias = _montar(buf[mask["01"]], LAYOUT_DETALHE)
        hdr, _ = _montar(buf[mask["00"]], LAYOUT_HEADER)
        trl, _ = _montar(buf[mask["99"]], LAYOUT_TRAILER)

        det.to_parquet(destino_parquet / f"cotahist_{ano}.parquet", index=False)
        hdr.to_parquet(destino_parquet / "meta" / f"header_{ano}.parquet", index=False)
        trl.to_parquet(destino_parquet / "meta" / f"trailer_{ano}.parquet", index=False)

        declarado = int(trl["TOTAL_REGISTROS"].iloc[0]) if len(trl) else -1
        return {
            "ano": ano,
            "status": "ok",
            "erro": "",
            "header": int(mask["00"].sum()),
            "detalhe": int(mask["01"].sum()),
            "trailer": int(mask["99"].sum()),
            "outros_tipos": outros,
            "lidos_total": int(len(tipo)),
            "declarado_trailer": declarado,
            "fora_de_245_bytes": len(fora),
            "anomalias": anomalias,
        }
    finally:
        alvo.unlink(missing_ok=True)


# --- Passo 4: integridade ---------------------------------------------------


def tabela_integridade(relatorios):
    """Compara a contagem lida com a quantidade declarada no trailer.

    O layout diz que TOTAL_REGISTROS inclui header e trailer. Divergência
    significa parse errado — a função reporta, não corrige.

    Args:
        relatorios: lista de dicionários devolvidos por `parse_cotahist_ano`.

    Returns:
        DataFrame com a comparação por ano.
    """
    df = pd.DataFrame(relatorios)
    df["diferenca"] = df["lidos_total"] - df["declarado_trailer"]
    df["confere"] = df["diferenca"] == 0
    colunas = [
        "ano", "header", "detalhe", "trailer", "outros_tipos",
        "lidos_total", "declarado_trailer", "diferenca", "confere",
        "fora_de_245_bytes",
    ]
    return df[colunas].sort_values("ano").reset_index(drop=True)


# --- Passo 5: inventário (sem nenhuma estatística de preço) -----------------


def inventario_por_ano(dir_parquet):
    """Inventário ano a ano: volume de registros, cobertura temporal e tickers.

    Nenhuma estatística sobre valores de preço. Nem mínimo, nem máximo, nem
    média.

    Args:
        dir_parquet: pasta com os parquets anuais de detalhe.

    Returns:
        DataFrame com uma linha por ano.
    """
    linhas = []
    for caminho in sorted(Path(dir_parquet).glob("cotahist_*.parquet")):
        ano = int(caminho.stem.split("_")[-1])
        df = pd.read_parquet(caminho, columns=["DATA_PREGAO", "CODNEG"])
        linhas.append({
            "ano": ano,
            "registros_detalhe": len(df),
            "primeiro_pregao": _iso(df["DATA_PREGAO"].min()),
            "ultimo_pregao": _iso(df["DATA_PREGAO"].max()),
            "datas_distintas": int(df["DATA_PREGAO"].nunique()),
            "codneg_distintos": int(df["CODNEG"].nunique()),
        })
    return pd.DataFrame(linhas).sort_values("ano").reset_index(drop=True)


def _iso(aaaammdd):
    """Formata um inteiro AAAAMMDD como AAAA-MM-DD. Só apresentação."""
    if pd.isna(aaaammdd):
        return ""
    s = str(int(aaaammdd))
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"


def distribuicao(dir_parquet, coluna: str, rotulos=None, minimo: int = 0):
    """Contagem de registros por valor de uma coluna categórica, base inteira.

    Só contagem. Nenhum valor de preço entra aqui.

    Args:
        dir_parquet: pasta com os parquets anuais de detalhe.
        coluna: nome da coluna do layout (ex.: "CODBDI", "TPMERC", "ESPECI").
        rotulos: dicionário opcional código -> rótulo do layout.
        minimo: descarta da EXIBIÇÃO valores com contagem abaixo disso.
            Não altera os dados; é só corte de tabela para leitura.

    Returns:
        DataFrame com valor, rótulo, registros e participação percentual.
    """
    total = None
    for caminho in sorted(Path(dir_parquet).glob("cotahist_*.parquet")):
        s = pd.read_parquet(caminho, columns=[coluna])[coluna].value_counts()
        total = s if total is None else total.add(s, fill_value=0)

    df = total.sort_values(ascending=False).rename("registros").reset_index()
    df.columns = [coluna, "registros"]
    df["registros"] = df["registros"].astype("int64")
    if rotulos is not None:
        df["rotulo"] = df[coluna].map(rotulos).fillna("(não consta no layout)")
    df["pct"] = (100 * df["registros"] / df["registros"].sum()).round(4)
    return df[df["registros"] >= minimo].reset_index(drop=True)


def moedas_por_periodo(dir_parquet):
    """Faixa de datas em que cada moeda de referência (MODREF) aparece.

    A base atravessa vários regimes monetários e os valores estão na moeda da
    época. NADA é convertido — esta função apenas reporta o que o campo
    MODREF do layout permite identificar.

    Args:
        dir_parquet: pasta com os parquets anuais de detalhe.

    Returns:
        DataFrame com MODREF, primeira e última data, e nº de registros.
    """
    partes = []
    for caminho in sorted(Path(dir_parquet).glob("cotahist_*.parquet")):
        df = pd.read_parquet(caminho, columns=["MODREF", "DATA_PREGAO"])
        partes.append(df.groupby("MODREF")["DATA_PREGAO"].agg(["min", "max", "count"]))

    g = pd.concat(partes).groupby(level=0).agg({"min": "min", "max": "max", "count": "sum"})
    g = g.reset_index()
    g["primeira_data"] = g["min"].map(_iso)
    g["ultima_data"] = g["max"].map(_iso)
    g = g.rename(columns={"count": "registros"})
    return g[["MODREF", "primeira_data", "ultima_data", "registros"]].sort_values("primeira_data").reset_index(drop=True)


# --- Partição pelo corte temporal -------------------------------------------
#
# A partição é puramente temporal. Nenhum filtro por tipo de papel, código BDI,
# tipo de mercado, ticker ou liquidez. Todas as colunas e todas as linhas
# dentro da janela entram.
#
# `particionar()` ESCREVE em dados/holdout/ e não lê de volta. Devolve apenas a
# contagem de linhas e de arquivos do holdout — nada sobre o conteúdo. Depois
# de escrito, só src/holdout.py pode tocar naquela pasta.


def _para_int(data_iso: str) -> int:
    """Converte "AAAA-MM-DD" no inteiro AAAAMMDD usado pelo campo DATA_PREGAO."""
    return int(data_iso.replace("-", ""))


def ultimo_pregao(dir_parquet, ano: int) -> str:
    """Última data de pregão efetivamente presente no parquet de um ano.

    Lida da base, não suposta. Serve para fixar DATA_FIM sem inventar uma data
    que a fonte não tem.

    Args:
        dir_parquet: pasta com os parquets anuais de detalhe.
        ano: ano a inspecionar.

    Returns:
        Data em ISO "AAAA-MM-DD".
    """
    caminho = Path(dir_parquet) / f"cotahist_{ano}.parquet"
    return _iso(pd.read_parquet(caminho, columns=["DATA_PREGAO"])["DATA_PREGAO"].max())


def particionar(dir_parquet, dir_desenho, dir_holdout, data_inicio: str, data_corte: str, data_fim: str) -> dict:
    """Divide a base bruta em desenho e holdout pelo corte temporal.

    - desenho: DATA_INICIO <= data de pregão <= DATA_CORTE
    - holdout: DATA_CORTE  <  data de pregão <= DATA_FIM

    Anos inteiramente fora de [data_inicio, data_fim] não entram em nenhuma das
    duas pastas e permanecem apenas em `dados/bruto/`.

    Args:
        dir_parquet: pasta com os parquets anuais brutos.
        dir_desenho: pasta de saída do período de desenho.
        dir_holdout: pasta de saída do holdout.
        data_inicio: ISO "AAAA-MM-DD".
        data_corte: ISO "AAAA-MM-DD".
        data_fim: ISO "AAAA-MM-DD".

    Returns:
        Dicionário com o detalhamento por ano do DESENHO, e do holdout apenas
        o total de linhas e de arquivos escritos.
    """
    dir_desenho, dir_holdout = Path(dir_desenho), Path(dir_holdout)
    dir_desenho.mkdir(parents=True, exist_ok=True)
    dir_holdout.mkdir(parents=True, exist_ok=True)

    ini, corte, fim = _para_int(data_inicio), _para_int(data_corte), _para_int(data_fim)
    ano_ini, ano_fim = ini // 10000, fim // 10000

    por_ano, holdout_linhas, holdout_arquivos, na_janela = [], 0, 0, 0

    for caminho in sorted(Path(dir_parquet).glob("cotahist_*.parquet")):
        ano = int(caminho.stem.split("_")[-1])
        if ano < ano_ini or ano > ano_fim:
            por_ano.append({"ano": ano, "desenho": 0, "fora_do_recorte": True})
            continue

        df = pd.read_parquet(caminho)
        d = df["DATA_PREGAO"]
        na_janela += int(((d >= ini) & (d <= fim)).sum())

        desenho = df[(d >= ini) & (d <= corte)]
        if len(desenho):
            desenho.to_parquet(dir_desenho / f"cotahist_{ano}.parquet", index=False)
        por_ano.append({"ano": ano, "desenho": len(desenho), "fora_do_recorte": False})

        holdout = df[(d > corte) & (d <= fim)]
        if len(holdout):
            holdout.to_parquet(dir_holdout / f"cotahist_{ano}.parquet", index=False)
            holdout_linhas += len(holdout)
            holdout_arquivos += 1

        del df, desenho, holdout

    return {
        "desenho_por_ano": pd.DataFrame(por_ano),
        "holdout_linhas": holdout_linhas,
        "holdout_arquivos": holdout_arquivos,
        "linhas_na_janela": na_janela,
    }


def contar_bruto_na_janela(dir_parquet, data_inicio: str, data_fim: str) -> int:
    """Conta as linhas dos parquets BRUTOS dentro da janela de trabalho.

    Contagem independente, usada para conferir a partição: desenho + holdout
    tem de bater exatamente com este número.

    Args:
        dir_parquet: pasta com os parquets anuais brutos.
        data_inicio: ISO "AAAA-MM-DD".
        data_fim: ISO "AAAA-MM-DD".

    Returns:
        Total de linhas na janela.
    """
    ini, fim = _para_int(data_inicio), _para_int(data_fim)
    total = 0
    for caminho in sorted(Path(dir_parquet).glob("cotahist_*.parquet")):
        d = pd.read_parquet(caminho, columns=["DATA_PREGAO"])["DATA_PREGAO"]
        total += int(((d >= ini) & (d <= fim)).sum())
    return total


# --- Verificação mecânica da regra do holdout --------------------------------


def _nos_de_docstring(arvore: ast.AST) -> set:
    """Identifica, por identidade de objeto, os nós de docstring da árvore.

    Uma docstring é o primeiro statement do corpo de um módulo, função ou
    classe, quando esse statement é uma expressão contendo só uma constante
    string. Esses nós precisam ser excluídos da varredura porque não são
    código executável.
    """
    ids = set()
    for no in ast.walk(arvore):
        if isinstance(no, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            corpo = no.body
            if corpo and isinstance(corpo[0], ast.Expr):
                valor = corpo[0].value
                if isinstance(valor, ast.Constant) and isinstance(valor.value, str):
                    ids.add(id(valor))
    return ids


def verificar_acesso_holdout_em_codigo(caminho_arquivo) -> list:
    """Verifica se um arquivo .py referencia o caminho "dados/holdout" em código executável.

    GARANTE: percorre a árvore sintática (`ast`) do arquivo e reporta todo
    literal de string que contenha o trecho "dados/holdout" (case-insensitive)
    e que esteja em posição de código executável — argumento de chamada, lado
    direito de atribuição, valor default de parâmetro, ou qualquer outra
    expressão da árvore. Comentários e docstrings são excluídos porque não são
    nós executáveis (docstrings são identificados e ignorados explicitamente
    por `_nos_de_docstring`; comentários nem chegam a existir na árvore
    sintática do Python). Isso resolve pela raiz o problema da busca textual
    ingênua, que casava com qualquer menção literal ao caminho, executável ou
    não — inclusive dentro de docstring e comentário.

    O alvo é deliberadamente o caminho "dados/holdout", não a palavra
    "holdout" isolada: a palavra sozinha aparece em identificadores e
    mensagens legítimas que não são acesso a arquivo (ex.: as chaves
    `holdout_linhas`/`holdout_arquivos` que `particionar()` devolve, ou o
    texto da exceção de `_validar_janela`), e um casamento por palavra
    dispararia nelas sem que exista acesso algum.

    NÃO GARANTE: ausência de acesso indireto — string montada em tempo de
    execução por concatenação ou f-string vinda de fora do arquivo, caminho
    recebido como parâmetro de outra função (ex.: o parâmetro `dir_holdout`
    de `particionar()`, que é um nome, não um literal), nome reexportado por
    um módulo importado, ou qualquer referência que não apareça como literal
    de string na árvore sintática deste arquivo específico. Não é uma prova
    de que o arquivo nunca acessa o holdout em tempo de execução — é uma
    prova de que ele não o faz por meio de um literal de string visível
    estaticamente neste arquivo.

    Args:
        caminho_arquivo: caminho do arquivo .py a analisar.

    Returns:
        Lista de dicionários {"linha": int, "trecho": str}, um por
        referência executável encontrada. Lista vazia = nenhuma referência.
    """
    caminho = Path(caminho_arquivo)
    codigo = caminho.read_text(encoding="utf-8")
    arvore = ast.parse(codigo, filename=str(caminho))

    ignorar = _nos_de_docstring(arvore)

    # Montado por concatenação, e não como um único literal "dados/holdout",
    # para que o próprio código deste verificador não se autoacuse ao
    # escanear src/dados.py (que é o arquivo onde este verificador mora).
    alvo = "dados" + "/" + "holdout"

    achados = []
    for no in ast.walk(arvore):
        if isinstance(no, ast.Constant) and isinstance(no.value, str):
            if id(no) in ignorar:
                continue
            if alvo in no.value.lower():
                achados.append({"linha": no.lineno, "trecho": no.value})
    return achados


# =============================================================================
# CAMADA BRUTA — CDI (SGS/BCB, série 12)
# =============================================================================
#
# Fonte: API SGS do Banco Central, série 12 (CDI, taxa % ao dia). Registrada
# em decisoes/01_decisoes_estruturais.md, linha "Fonte do CDI", 10/08/2026.
#
# REGRAS DESTA CAMADA (mesmas da camada bruta do COTAHIST):
#   - A taxa vem em percentual ao dia, exatamente como o SGS publica. NADA é
#     convertido, NADA é anualizado, NADA é composto.
#   - Nenhuma linha é filtrada.
#   - A partição em desenho/holdout segue o MESMO DATA_CORTE do COTAHIST, e
#     obedece a mesma regra: escreve o holdout em disco e não o lê de volta.
# =============================================================================

URL_SGS_CDI = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados"
SGS_JANELA_ANOS = 10  # limite documentado da API do SGS por requisição


def _blocos_sgs(data_inicio: str, data_fim: str, janela_anos: int = SGS_JANELA_ANOS):
    """Divide [data_inicio, data_fim] em blocos consecutivos de até `janela_anos` anos.

    Cobre o intervalo inteiro sem sobreposição e sem buraco: o próximo bloco
    sempre começa no dia seguinte ao fim do bloco anterior.

    Args:
        data_inicio: ISO "AAAA-MM-DD".
        data_fim: ISO "AAAA-MM-DD".
        janela_anos: tamanho máximo de cada bloco, em anos.

    Returns:
        Lista de tuplas (inicio, fim), ambos `datetime.date`.
    """
    ini = date.fromisoformat(data_inicio)
    fim = date.fromisoformat(data_fim)
    if ini > fim:
        raise ValueError(f"data_inicio={ini} é posterior a data_fim={fim}.")

    blocos = []
    cursor = ini
    while cursor <= fim:
        try:
            teto = cursor.replace(year=cursor.year + janela_anos) - timedelta(days=1)
        except ValueError:  # cursor é 29/02 e o ano de destino não é bissexto
            teto = cursor.replace(month=2, day=28, year=cursor.year + janela_anos) - timedelta(days=1)
        bloco_fim = min(teto, fim)
        blocos.append((cursor, bloco_fim))
        cursor = bloco_fim + timedelta(days=1)
    return blocos


def _verificar_serie_cdi(serie: pd.DataFrame, limite_gap_dias: int = 10) -> dict:
    """Confere a série concatenada contra si mesma: sem duplicata, sem buraco anômalo.

    Não compara contra o calendário de pregões do COTAHIST — são fontes
    diferentes e não há garantia de calendário idêntico. Compara a série do
    CDI consigo mesma: nenhuma data repetida, e nenhum intervalo entre datas
    consecutivas maior que `limite_gap_dias` dias corridos. Um erro de
    paginação (bloco perdido, fronteira mal calculada) aparece aqui como
    duplicata (sobreposição) ou como buraco (lacuna) — os dois casos que
    esta função existe para pegar.

    Args:
        serie: DataFrame com coluna `data` (datetime64), ordenado.
        limite_gap_dias: maior intervalo corrido tolerado entre duas datas
            consecutivas antes de ser reportado como suspeito. CDI publica
            em todo dia útil; um feriado prolongado ainda fica bem abaixo
            do default de 10 dias.

    Returns:
        dict com `duplicatas` (nº de datas repetidas), `maior_gap_dias` e
        `gaps_suspeitos` (lista de datas ISO onde o gap excedeu o limite).
    """
    dif = serie["data"].diff().dt.days.dropna()
    gaps = dif[dif > limite_gap_dias]
    return {
        "duplicatas": int(serie["data"].duplicated().sum()),
        "maior_gap_dias": int(dif.max()) if len(dif) else 0,
        "gaps_suspeitos": serie.loc[gaps.index, "data"].dt.strftime("%Y-%m-%d").tolist(),
    }


def baixar_cdi_sgs12(data_inicio: str, data_fim: str) -> pd.DataFrame:
    """Baixa a série 12 (CDI) do SGS/BCB, paginando o limite de 10 anos por requisição.

    O QUE FAZ: baixa em blocos de até `SGS_JANELA_ANOS` anos, concatena e
    confere, por `_verificar_serie_cdi()`, que a série resultante não tem
    data duplicada nem buraco anômalo — a assinatura de um erro de
    paginação.

    O QUE NÃO FAZ: não filtra, não converte, não anualiza. `valor` sai
    exatamente como a API publica: taxa percentual ao dia.

    Args:
        data_inicio: início da janela pedida, ISO "AAAA-MM-DD".
        data_fim: fim da janela pedida, ISO "AAAA-MM-DD".

    Returns:
        DataFrame com colunas `data` (datetime64) e `valor` (float, % ao
        dia), ordenado por data, sem duplicata.

    Raises:
        ValueError: se algum bloco vier vazio, ou se a verificação encontrar
            data duplicada ou buraco suspeito entre blocos.
    """
    partes = []
    for ini, fim in _blocos_sgs(data_inicio, data_fim):
        params = {
            "formato": "json",
            "dataInicial": ini.strftime("%d/%m/%Y"),
            "dataFinal": fim.strftime("%d/%m/%Y"),
        }
        url = f"{URL_SGS_CDI}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            bruto = json.loads(resp.read().decode("utf-8"))
        if not bruto:
            raise ValueError(f"bloco {ini}–{fim} do SGS/CDI (série 12) veio vazio.")
        bloco = pd.DataFrame(bruto)
        bloco["data"] = pd.to_datetime(bloco["data"], format="%d/%m/%Y")
        bloco["valor"] = bloco["valor"].astype(float)
        partes.append(bloco[["data", "valor"]])

    serie = pd.concat(partes, ignore_index=True).sort_values("data").reset_index(drop=True)

    verificacao = _verificar_serie_cdi(serie)
    if verificacao["duplicatas"] or verificacao["gaps_suspeitos"]:
        raise ValueError(
            f"verificação de paginação do SGS/CDI falhou: {verificacao}"
        )

    return serie


def particionar_cdi(serie: pd.DataFrame, dir_desenho, dir_holdout, data_corte: str) -> dict:
    """Divide a série do CDI em desenho e holdout pelo mesmo corte temporal do COTAHIST.

    Mesma regra de `particionar()`: ESCREVE em `dir_holdout` e não lê de
    volta. Devolve só a contagem de linhas do holdout — nada sobre o
    conteúdo.

    Args:
        serie: DataFrame com colunas `data` e `valor`, saída de
            `baixar_cdi_sgs12`.
        dir_desenho: pasta de saída do período de desenho.
        dir_holdout: pasta de saída do holdout.
        data_corte: ISO "AAAA-MM-DD".

    Returns:
        dict com `desenho_linhas`, `desenho_primeira_data`,
        `desenho_ultima_data` e `holdout_linhas` (apenas contagem).
    """
    dir_desenho, dir_holdout = Path(dir_desenho), Path(dir_holdout)
    dir_desenho.mkdir(parents=True, exist_ok=True)
    dir_holdout.mkdir(parents=True, exist_ok=True)

    corte = pd.Timestamp(data_corte)
    desenho = serie[serie["data"] <= corte].reset_index(drop=True)
    holdout = serie[serie["data"] > corte].reset_index(drop=True)

    if len(desenho):
        desenho.to_parquet(dir_desenho / "cdi_sgs12.parquet", index=False)
    if len(holdout):
        holdout.to_parquet(dir_holdout / "cdi_sgs12.parquet", index=False)

    return {
        "desenho_linhas": len(desenho),
        "desenho_primeira_data": desenho["data"].min().strftime("%Y-%m-%d") if len(desenho) else None,
        "desenho_ultima_data": desenho["data"].max().strftime("%Y-%m-%d") if len(desenho) else None,
        "holdout_linhas": len(holdout),
    }


def datas_faltantes_cdi_vs_cotahist(desenho_cdi: pd.DataFrame, dir_parquet_desenho_cotahist) -> int:
    """Conta datas de pregão do COTAHIST (desenho) sem cotação de CDI correspondente.

    Calendário de referência: datas distintas de `DATA_PREGAO` nos parquets
    de `dir_parquet_desenho_cotahist` — já particionados pelo corte, portanto
    dentro do período de desenho, sem tocar holdout.

    Args:
        desenho_cdi: DataFrame de CDI do período de desenho (coluna `data`).
        dir_parquet_desenho_cotahist: pasta com os parquets de
            `dados/desenho/` do COTAHIST.

    Returns:
        Número de datas do calendário de pregões sem linha correspondente
        na série de CDI do desenho.
    """
    pregoes = set()
    for caminho in sorted(Path(dir_parquet_desenho_cotahist).glob("cotahist_*.parquet")):
        d = pd.read_parquet(caminho, columns=["DATA_PREGAO"])["DATA_PREGAO"]
        pregoes.update(pd.to_datetime(d.astype(str), format="%Y%m%d").tolist())

    presentes = set(desenho_cdi["data"].tolist())
    return len(pregoes - presentes)


# =============================================================================
# UNIVERSO INVESTÍVEL — filtro de tipo de papel e elegibilidade por data
# =============================================================================
#
# Tudo aqui implementa decisões já registradas em
# `decisoes/01_decisoes_estruturais.md`. Nenhum número desta seção foi
# escolhido aqui: cada um desce da âncora declarada lá.
#
#   - chave de ativo: CODISI (ISIN), nunca CODNEG        (10/08/2026)
#   - critério por INCLUSÃO, não por exclusão            (10/08/2026)
#   - raiz de ESPECI ∈ {ON, PN, PNA–PNG, UNT}
#     E CODBDI = 02 E TPMERC = 10                        (10/08/2026)
#   - qualificador REC e as 8 raízes ambíguas: EXCLUÍDOS (11/08/2026)
#   - N = 152 semanas, X = 100% da grade, cap K = 50     (11/08/2026)
#
# O CDI NÃO entra aqui: ele não é variável de decisão em nenhuma das três
# séries (`decisoes/01`, 11/08/2026). Este módulo devolve só ações.
# =============================================================================

# Raízes de ESPECI aceitas. Lista de INCLUSÃO: o que não está aqui não entra,
# e é isso que impede um tipo novo (FIAGRO, BDR de ETF) de escorregar para
# dentro do universo sem ninguém decidir.
RAIZES_ACAO = frozenset({
    "ON", "PN", "PNA", "PNB", "PNC", "PND", "PNE", "PNF", "PNG", "UNT",
})

# Recibo de subscrição: instrumento distinto, negocia com desconto e converte
# depois. Incluí-lo duplicaria o emissor na matriz.
QUALIFICADOR_EXCLUIDO = "REC"

CODBDI_LOTE_PADRAO = "02"
TPMERC_VISTA = 10

N_SEMANAS_JANELA = 152
CAP_UNIVERSO = 50

# ESCADA DE CONTINGÊNCIA — ciclo COMPLETO, 11/08/2026. Valor final: True.
#
# A escada foi pré-registrada em `decisoes/01` ANTES de qualquer diagnóstico,
# justamente para que o ajuste não pudesse ser escolhido por resultado:
#   - se K_t >= 30 em todas as eras -> congela a regra como está;
#   - se K_t < 30 em alguma era -> o ÚNICO relaxamento permitido é trocar
#     "presença em D" por "presença na última data de amostragem".
#     Nunca mexer em N nem no cap.
#
#   1. GATILHO DISPAROU: o diagnóstico mediu K_t < 30 em 194 das 1.341 datas
#      (eras 1997–2001), mínimo 4.
#   2. CONTINGÊNCIA APLICADA mecanicamente: False.
#   3. MEDIDA INEFICAZ: ganho médio de 0,22 ativo; K mínimo continuou 4 e as
#      MESMAS 194 datas seguiram abaixo de 30. Não restaurou o piso, que era
#      o propósito pré-registrado dela.
#   4. REVERTIDA: volta a True. Um remédio pré-registrado que falha no
#      propósito pré-registrado é removido pelo mesmo protocolo que o
#      introduziu. Reverter não é especificação nova — é retorno ao default.
#      O relaxamento cobrava dois custos sem entregar nada: a janela passava a
#      dar 151 retornos em vez dos 152 do paper-base, e criava estado
#      "elegível-mas-inegociável" em 5,9% das datas, contradizendo a
#      equivalência registrada entre elegibilidade e guarda do motor.
#   5. ESCADA EXAURIDA: nenhum outro relaxamento é permitido por ela. Qualquer
#      mudança daqui em diante é decisão NOVA e datada, tomada depois de ver
#      o resultado, e tem de ser declarada como tal.
#
# O gargalo medido não é tecnicismo da regra: como remover a exigência de
# presença em D quase não alterou K_t, a restrição ativa é a exigência de 152
# semanas contínuas. Entre 1997 e 2001 a B3 não continha 30 papéis com três
# anos de negociação semanal ininterrupta. É fato sobre o mercado, medido por
# régua fixada antes da medição.
EXIGIR_PRESENCA_EM_D = True

# FATCOT entra no painel porque a detecção de evento societário precisa dele
# (troca da unidade de cotação, ver a seção EVENTOS SOCIETÁRIOS). Antes de
# 12/08/2026 ele era descartado nesta fronteira — e a troca de unidade virava
# um "retorno" de até 1000x sem que nada a jusante pudesse sequer detectar.
COLUNAS_PAINEL = ["DATA_PREGAO", "CODISI", "ESPECI", "CODBDI", "TPMERC", "PREULT",
                  "VOLTOT", "FATCOT"]

# dtype de data usado no painel. DERIVADO da mesma conversão que popula a
# coluna, em vez de escrito à mão, para que o painel VAZIO tenha exatamente o
# mesmo tipo do painel cheio em qualquer versão do pandas (2.x devolve
# `datetime64[ns]`, 3.x devolve `datetime64[us]`). Um literal aqui ficaria
# desatualizado em silêncio, e o caso vazio voltaria a ter tipo divergente —
# que foi a metade do bug de 12/08/2026.
DTYPE_DATA = pd.to_datetime(pd.Series(["19950102"]), format="%Y%m%d").dtype


def _parquets_cotahist(dir_dados, quem: str):
    """Lista os parquets anuais de uma pasta. FALHA ALTO se não houver nenhum.

    POR QUE ISTO EXISTE (bug de 12/08/2026): `Path(dir).glob(...)` numa pasta
    inexistente não levanta nada — devolve um iterador vazio. Quem chamava
    seguia com uma lista vazia, montava um resultado vazio e devolvia esse
    resultado como se fosse a resposta. O erro só aparecia muito depois, longe
    da causa, e disfarçado de outra coisa: `painel["data"].min()` sobre um
    DataFrame vazio de dtype `object` devolve `float('nan')`, e o sintoma era
    um `AttributeError: 'float' object has no attribute 'date'`.

    "Pasta sem dado" e "pasta com dado que não passou no filtro" são situações
    diferentes e agora falham diferente: a primeira levanta aqui, a segunda
    devolve resultado vazio com os tipos certos.

    Args:
        dir_dados: pasta com os parquets `cotahist_*.parquet`.
        quem: nome da função chamadora, para a mensagem de erro.

    Returns:
        Lista ordenada de caminhos.

    Raises:
        ValueError: se a pasta não existir ou não contiver nenhum parquet.
    """
    caminho = Path(dir_dados)
    arquivos = sorted(caminho.glob("cotahist_*.parquet"))
    if arquivos:
        return arquivos

    absoluto = caminho if caminho.is_absolute() else Path.cwd() / caminho
    raise ValueError(
        f"{quem}: nenhum arquivo `cotahist_*.parquet` em {dir_dados!r}.\n"
        f"  caminho absoluto tentado : {absoluto}\n"
        f"  a pasta existe?          : {caminho.exists()}\n"
        f"  diretório de trabalho    : {Path.cwd()}\n"
        f"Se o caminho for RELATIVO e o processo tiver sido iniciado de outra "
        f"pasta (um notebook rodando de `notebooks/`, por exemplo), ele resolve "
        f"contra o diretório de trabalho e aponta para o lugar errado. As "
        f"constantes `DIR_DESENHO`/`DIR_HOLDOUT` de `config.py` são ancoradas "
        f"na raiz do repositório exatamente por isso — use-as em vez de digitar "
        f"o caminho. Se elas já estão em uso e este erro apareceu, então os "
        f"dados não foram baixados/particionados: rode "
        f"`notebooks/00_dados.ipynb`."
    )


def _painel_vazio():
    """Painel sem linha nenhuma, com os dtypes CORRETOS de um painel cheio.

    `pd.DataFrame(columns=[...])` cria colunas `object`, e `object` vazio tem
    duas armadilhas que o bug de 12/08/2026 usou as duas: `.min()` devolve
    `float('nan')` em vez de `NaT`, e comparar esse `nan` com um `Timestamp`
    levanta `TypeError` em vez de responder à pergunta que foi feita. Com o
    tipo certo, um painel vazio se comporta como painel — `.min()` é `NaT` e a
    comparação com a data de corte é uma comparação de datas de verdade.
    """
    return pd.DataFrame({
        "data": pd.Series([], dtype=DTYPE_DATA),
        "CODISI": pd.Series([], dtype="string"),
        "PREULT": pd.Series([], dtype="float64"),
        "VOLTOT": pd.Series([], dtype="float64"),
        "FATCOT": pd.Series([], dtype="Int64"),
        "ESPECI": pd.Series([], dtype="string"),
    })


def raiz_especi(especi):
    """Extrai a RAIZ de `ESPECI` — o primeiro token, sem o sufixo de segmento.

    O campo carrega o segmento de listagem junto ("ON NM", "PN N1", "UNT N2"),
    então comparar a string inteira com "ON" perderia todo o Novo Mercado.
    A raiz é o que identifica a espécie do papel; o resto é qualificador.

    Args:
        especi: Series de strings do campo ESPECI.

    Returns:
        Series com a raiz de cada linha.
    """
    # ESPECI vazio não tem raiz: devolve "" (que não está na lista de inclusão)
    # em vez de NaN, para que a comparação a jusante não vire um terceiro caso.
    return especi.fillna("").astype(str).str.strip().str.split(n=1).str[0].fillna("")


def filtrar_tipo_papel(df):
    """Aplica o filtro de tipo de papel por INCLUSÃO. Não filtra por data.

    Três componentes, todos necessários (`decisoes/01`, 10 e 11/08/2026):
      - raiz de `ESPECI` na lista de inclusão `RAIZES_ACAO`;
      - `CODBDI` = 02 (lote padrão) E `TPMERC` = 10 (à vista) — `ESPECI`
        sozinho não separa ação de derivativo, porque opção também carrega
        raiz UNT;
      - qualificador `REC` em qualquer posição: FORA.

    Exige `CODISI` não vazio: sem ISIN não há chave estável de ativo, e a
    linha não é identificável ao longo do tempo. Pela governança de inclusão,
    o que não é identificável não entra.

    Args:
        df: DataFrame no formato bruto do COTAHIST.

    Returns:
        DataFrame com o subconjunto de linhas que passa no filtro.
    """
    especi = df["ESPECI"].fillna("").astype(str).str.strip()
    resto = especi.str.split(n=1).str[1].fillna("")

    isin = df["CODISI"].fillna("").astype(str).str.strip()

    mascara = (
        raiz_especi(especi).isin(RAIZES_ACAO)
        & ~resto.str.contains(rf"\b{QUALIFICADOR_EXCLUIDO}\b", regex=True, na=False)
        & (df["CODBDI"].fillna("").astype(str).str.strip() == CODBDI_LOTE_PADRAO)
        & (pd.to_numeric(df["TPMERC"], errors="coerce") == TPMERC_VISTA)
        & (isin != "")
    )
    return df[mascara.to_numpy()]


# =============================================================================
# EVENTOS SOCIETÁRIOS — neutralização por marcador declarado no arquivo
# =============================================================================
#
# Registrado em `decisoes/01_decisoes_estruturais.md`, 12/08/2026, como
# **DECISÃO PÓS-RESULTADO declarada**: o defeito de dado é anterior ao
# resultado e independe dele, mas a descoberta veio da revisão adversarial de
# 12/08, depois do primeiro backtest.
#
# O PROBLEMA: o COTAHIST publica preço BRUTO. Grupamento, bonificação e troca
# de fator de cotação mudam a QUANTIDADE de ações (ou a unidade da cotação) e
# produzem saltos de preço que não são retorno — um grupamento 10:1 vira
# "+900%", uma bonificação 1:4 vira "-80%". Medido sobre dados/desenho em
# 12/08/2026: 675 retornos semanais acima de +100% no período avaliado, com
# máximo de +496.941%; 12 deles efetivamente detidos pela carteira 1/N.
#
# O CANAL DE DETECÇÃO É O PRÓPRIO ARQUIVO, nunca a magnitude do retorno:
#
#   - o qualificador do campo `ESPECI` (posições 5-8) carrega os marcadores
#     "ex-evento" documentados no layout rev. 02 (05/10/2020), p. 9:
#     `EG` = Ex-Grupamento, `EB` = Ex-Bonificação, e combinações (`EDB`,
#     `EJB`, `EBG`...). Na base, desdobramentos também aparecem sob `EB`;
#   - o campo `FATCOT` declara a unidade da cotação ('1' = unitária,
#     '1000' = por lote de mil, layout p. 6). A troca de unidade é da mesma
#     categoria da vírgula implícita: leitura de formato, não preço.
#
# Um limiar de magnitude NUNCA dispara correção. Magnitude aparece apenas em
# diagnóstico (contagem antes/depois no notebook 02). O que o marcador não
# cobre fica como RESÍDUO DECLARADO — reprecificações sem marcador (ex.:
# pós-cisão da Telebras em 1998) e quedas reais de mercado não são tocadas.
#
# O QUE É NEUTRALIZADO (só evento de QUANTIDADE):
#   (a) dia em que o qualificador GANHA a letra `G` (grupamento);
#   (b) dia em que GANHA a letra `B` (bonificação/desdobramento);
#   (c) dia em que `FATCOT` muda;
#   (d) extensão de defasagem-1: quando o ganho de G/B ocorre no pregão
#       observado imediatamente seguinte ao primeiro dia do mesmo episódio de
#       qualificadores, o primeiro dia também é neutralizado. Cobre o padrão
#       em que a B3 registra o "ex" um pregão depois do salto (Souza Cruz
#       21->22/03/2011; Mundial 30->31/05/2011). A defasagem 1 separa esses
#       casos, mecanicamente, dos dividendos reais seguidos de bônus dias
#       depois (CEBR 10/2021, defasagem 8, intocado).
#
# PROVENTOS NÃO SÃO NEUTRALIZADOS: `ED`, `EJ`, `ER`, `ES` e combinações sem
# G/B são dinheiro ou direito que SAI para o acionista — o degrau de preço é
# real para uma série price-only, e a linha "Tratamento de proventos"
# (decisoes/01, 10/08/2026) continua valendo, intocada. Grupamento e
# bonificação são o oposto: a quantidade muda e o valor do acionista não —
# o degrau é puramente de unidade.
#
# O FATOR NÃO É PARÂMETRO: é a razão de preço OBSERVADA no próprio dia
# marcado (`PREULT_dia / último PREULT observado`). Nada é estimado,
# arredondado ou buscado fora. O caso composto — grupamento E troca de
# FATCOT no mesmo dia (Bradesco 22/03/2004: 10.000:1 de ações com 1000->1 de
# cotação; AmBev 02/08/2007: 100:1 com 1000->1) — sai correto de graça,
# porque a razão observada já compõe os dois efeitos. Aplicar `PREULT/FATCOT`
# ingenuamente, ao contrário, DUPLICARIA a correção nesses dias e
# transformaria o +847% do Bradesco em ~+979.500% (armadilha verificada pela
# revisão de 12/08, afetando 388 ISINs que trocam de FATCOT na série).
#
# CUSTO DECLARADO: o movimento REAL de mercado do dia neutralizado
# (tipicamente ±2-5%) é perdido junto com o artefato. Erro limitado, sem
# direção preferencial, e cada ocorrência está no log.
# =============================================================================

# Letras de qualificador que identificam evento de QUANTIDADE. `G` e `B`
# apenas: `D`/`J`/`R`/`S` são proventos e ficam fora por decisão registrada.
LETRAS_EVENTO_QUANTIDADE = ("G", "B")

COLUNAS_LOG_EVENTOS = [
    "CODISI", "data", "canal", "qual_antes", "qual_depois",
    "fatcot_antes", "fatcot_depois", "preco_antes", "preco_depois",
    "fator", "dias_desde_ultimo_preco",
]


def qualificador_especi(especi):
    """Extrai o QUALIFICADOR de `ESPECI` — o sub-campo posicional 5-8.

    O campo `ESPECI` é `X(10)` com sub-campos posicionais (estrutura deduzida
    dos dados e validada em 10/08/2026, ver `decisoes/03_diario.md`):
    posições 1-3 = raiz, 4 = marcador `*`, 5-8 = qualificador de evento,
    9-10 = segmento de listagem. `raiz_especi()` lê o primeiro sub-campo;
    esta função lê o terceiro. Tokenizar por espaço não serve aqui: em
    "PN *REC" o qualificador é colado ao marcador, e em "ON  EG  NM" há dois
    espaços de cada lado.

    Args:
        especi: Series de strings do campo ESPECI (como vem do parquet, já
            sem espaços nas pontas).

    Returns:
        Series com o qualificador de cada linha ("" quando não há).
    """
    e = especi.fillna("").astype(str).str.ljust(10)
    return e.str.slice(4, 8).str.strip()


def detectar_eventos_societarios(painel):
    """Detecta os dias de evento societário e devolve o LOG, sem alterar nada.

    O QUE FAZ: compara cada pregão de cada ISIN com o pregão OBSERVADO
    anterior do mesmo ISIN, dentro da janela carregada, e marca os dias das
    regras (a)-(d) do cabeçalho da seção. Para cada dia marcado calcula o
    fator = `PREULT_dia / PREULT_anterior` — a razão observada, que é o que
    `aplicar_ajuste_societario()` vai neutralizar.

    O QUE NÃO FAZ: não olha magnitude de retorno, não altera o painel, não
    consulta nada fora da janela recebida. Um evento no PRIMEIRO dia
    observado do ISIN é indetectável por construção — não há pregão anterior
    para comparar — e também é inofensivo: nenhum retorno atravessa esse dia.
    Uma extensão de defasagem-1 cujo dia-base não tem preço anterior entra no
    log com fator NaN e é ignorada pela aplicação, pelo mesmo motivo.

    Args:
        painel: DataFrame longo com colunas `data`, `CODISI`, `PREULT`,
            `FATCOT` e `ESPECI`. Linhas repetidas do mesmo (ISIN, dia) são
            consolidadas com `last`, a mesma convenção do pivô de preços.

    Returns:
        DataFrame com as colunas de `COLUNAS_LOG_EVENTOS`, um dia neutralizado
        por linha, ordenado por (data, CODISI). Vazio se nada foi detectado.
    """
    obrigatorias = {"data", "CODISI", "PREULT", "FATCOT", "ESPECI"}
    faltam = obrigatorias - set(painel.columns)
    if faltam:
        raise ValueError(
            f"detectar_eventos_societarios: faltam as colunas {sorted(faltam)}. "
            f"A detecção precisa de ESPECI e FATCOT — se o painel veio de "
            f"`carregar_painel()`, essas colunas já estão lá."
        )
    if len(painel) == 0:
        return pd.DataFrame(columns=COLUNAS_LOG_EVENTOS)

    df = (painel[["data", "CODISI", "PREULT", "FATCOT", "ESPECI"]]
          .sort_values(["CODISI", "data"])
          .drop_duplicates(subset=["CODISI", "data"], keep="last")
          .reset_index(drop=True))
    df["qual"] = qualificador_especi(df["ESPECI"])

    g = df.groupby("CODISI", sort=False)
    df["qual_p1"] = g["qual"].shift(1)
    df["qual_p2"] = g["qual"].shift(2)
    df["fatcot_p1"] = g["FATCOT"].shift(1)
    df["preco_p1"] = g["PREULT"].shift(1)
    df["preco_p2"] = g["PREULT"].shift(2)
    df["data_p1"] = g["data"].shift(1)
    df["data_p2"] = g["data"].shift(2)

    tem_prev = df["qual_p1"].notna()
    qp1 = df["qual_p1"].fillna("")
    ganha = {}
    for letra in LETRAS_EVENTO_QUANTIDADE:
        ganha[letra] = (tem_prev & df["qual"].str.contains(letra, regex=False)
                        & ~qp1.str.contains(letra, regex=False))
    ganha_gb = ganha["G"] | ganha["B"]
    muda_fatcot = tem_prev & df["FATCOT"].ne(df["fatcot_p1"])

    # (d) extensão de defasagem-1: o ganho de G/B aconteceu em MEIO de episódio
    # (qual anterior não vazio) e o episódio começou exatamente no pregão
    # observado anterior (qual de dois pregões atrás vazio ou inexistente).
    extensao = ganha_gb & (qp1 != "") & (df["qual_p2"].fillna("") == "")

    def _fator(preco, base):
        preco = pd.to_numeric(preco, errors="coerce")
        base = pd.to_numeric(base, errors="coerce")
        f = preco / base
        return f.where((preco > 0) & (base > 0))

    linhas = []

    marcados = df[ganha_gb | muda_fatcot]
    for _, r in marcados.iterrows():
        canais = []
        if ganha["G"].loc[r.name]:
            canais.append("ganha_G")
        if ganha["B"].loc[r.name]:
            canais.append("ganha_B")
        if muda_fatcot.loc[r.name]:
            canais.append("fatcot")
        gap = (r["data"] - r["data_p1"]).days if pd.notna(r["data_p1"]) else np.nan
        linhas.append({
            "CODISI": r["CODISI"], "data": r["data"], "canal": "+".join(canais),
            "qual_antes": r["qual_p1"] if pd.notna(r["qual_p1"]) else "",
            "qual_depois": r["qual"],
            "fatcot_antes": r["fatcot_p1"], "fatcot_depois": r["FATCOT"],
            "preco_antes": r["preco_p1"], "preco_depois": r["PREULT"],
            "fator": _fator(pd.Series([r["PREULT"]]), pd.Series([r["preco_p1"]])).iloc[0],
            "dias_desde_ultimo_preco": gap,
        })

    for _, r in df[extensao].iterrows():
        gap = (r["data_p1"] - r["data_p2"]).days if pd.notna(r["data_p2"]) else np.nan
        linhas.append({
            "CODISI": r["CODISI"], "data": r["data_p1"],
            "canal": "extensao_defasagem_1",
            "qual_antes": "", "qual_depois": r["qual_p1"],
            "fatcot_antes": r["fatcot_p1"], "fatcot_depois": r["fatcot_p1"],
            "preco_antes": r["preco_p2"], "preco_depois": r["preco_p1"],
            "fator": _fator(pd.Series([r["preco_p1"]]), pd.Series([r["preco_p2"]])).iloc[0],
            "dias_desde_ultimo_preco": gap,
        })

    log = pd.DataFrame(linhas, columns=COLUNAS_LOG_EVENTOS)
    if len(log):
        # Um mesmo (ISIN, dia) pode ter sido marcado por regra direta E por
        # extensão de um vizinho; a neutralização é uma só, e o rótulo que
        # fica é o do canal direto (a extensão é o marcador mais fraco).
        eh_extensao = (log["canal"] == "extensao_defasagem_1").astype(int)
        log = (log.assign(_ext=eh_extensao)
               .sort_values(["data", "CODISI", "_ext"])
               .drop_duplicates(subset=["CODISI", "data"], keep="first")
               .drop(columns="_ext")
               .reset_index(drop=True))
    return log


def aplicar_ajuste_societario(painel, eventos):
    """Neutraliza os dias do log no painel de preços. Nunca silencioso.

    O QUE FAZ: para cada ISIN, acumula multiplicativamente os fatores dos
    eventos e divide `PREULT` pelo fator acumulado a partir de cada data de
    evento (ajuste para frente). O retorno diário do dia neutralizado vira
    exatamente zero; TODOS os outros retornos ficam idênticos, porque numerador
    e denominador de cada razão de preços carregam o mesmo fator acumulado.
    O preço original é preservado em `PREULT_SEM_AJUSTE`.

    O QUE NÃO FAZ: não altera `VOLTOT` — volume FINANCEIRO não muda de escala
    com a quantidade de ações (preço×quantidade se compensam), então a
    elegibilidade, o ranking do cap e `K_t` ficam idênticos por construção.
    Não toca eventos com fator NaN (sem preço-base observado).

    Args:
        painel: DataFrame longo com ao menos `data`, `CODISI`, `PREULT`.
        eventos: saída de `detectar_eventos_societarios()`.

    Returns:
        Cópia do painel com `PREULT` ajustado e a coluna nova
        `PREULT_SEM_AJUSTE` com o valor original.
    """
    out = painel.copy()
    out["PREULT_SEM_AJUSTE"] = out["PREULT"]

    ev = eventos.dropna(subset=["fator"])
    ev = ev[ev["fator"] > 0]
    if not len(ev) or not len(out):
        # mesma ordenação do caminho com eventos, para o contrato não depender
        # de haver ou não evento na janela.
        return out.sort_values(["data", "CODISI"]).reset_index(drop=True)

    ev = ev.sort_values(["CODISI", "data"]).copy()
    ev["fator_acum"] = ev.groupby("CODISI")["fator"].cumprod()

    out = out.sort_values("data", kind="mergesort")
    ev = ev.sort_values("data", kind="mergesort")
    out = pd.merge_asof(
        out, ev[["CODISI", "data", "fator_acum"]],
        on="data", by="CODISI", direction="backward",
    )
    out["PREULT"] = out["PREULT_SEM_AJUSTE"] / out["fator_acum"].fillna(1.0)
    out = out.drop(columns=["fator_acum"])
    return out.sort_values(["data", "CODISI"]).reset_index(drop=True)


def carregar_painel(dir_dados, data_inicio: str = DATA_INICIO, data_fim: str = DATA_CORTE,
                    ajustar_eventos_societarios: bool = True):
    """Lê os parquets de uma pasta e devolve o painel longo já filtrado por tipo.

    Sujeito ao contrato de holdout: a janela pedida não pode ultrapassar
    `DATA_CORTE`.

    EVENTOS SOCIETÁRIOS: por default o painel sai com `PREULT` já NEUTRALIZADO
    nos dias marcados no próprio arquivo (ver a seção EVENTOS SOCIETÁRIOS e
    `decisoes/01`, 12/08/2026). O preço original fica em `PREULT_SEM_AJUSTE` e
    o log completo das neutralizações fica em
    `painel.attrs["eventos_societarios"]` — nada é silencioso. O default é
    ligado de propósito: a correção tem de ser idêntica nas três séries, e um
    chamador distraído não pode obter, sem pedir, a série contaminada.
    `ajustar_eventos_societarios=False` existe para diagnóstico e para a
    comparação antes/depois, e nunca para produção de resultado.

    Args:
        dir_dados: pasta com os parquets `cotahist_*.parquet`.
        data_inicio: início da janela, ISO "AAAA-MM-DD".
        data_fim: fim da janela, ISO "AAAA-MM-DD". Teto: `DATA_CORTE`.
        ajustar_eventos_societarios: se True (default), neutraliza os dias
            marcados e anexa o log em `attrs`. Se False, devolve o preço cru
            e NÃO anexa log.

    CONTRATO DE TIPO: a coluna `data` é SEMPRE `datetime64`, inclusive quando
    o resultado é vazio. O campo `DATA_PREGAO` do COTAHIST é um inteiro
    `AAAAMMDD` e é convertido aqui; nada a jusante deve receber inteiro. Ver
    `_painel_vazio()` para por que o caso vazio também precisa do tipo certo.

    Returns:
        DataFrame com colunas `data` (datetime64), `CODISI`, `PREULT`,
        `VOLTOT`, `FATCOT`, `ESPECI` — uma linha por papel por pregão — mais
        `PREULT_SEM_AJUSTE` quando o ajuste está ligado.

    Raises:
        ValueError: se a janela pedida ultrapassar `DATA_CORTE`, ou se não
            houver nenhum parquet em `dir_dados` (ver `_parquets_cotahist`).
    """
    _validar_janela(data_inicio, data_fim)
    ini, fim = _para_int(data_inicio), _para_int(data_fim)

    partes = []
    for caminho in _parquets_cotahist(dir_dados, "carregar_painel"):
        df = pd.read_parquet(caminho, columns=COLUNAS_PAINEL)
        d = df["DATA_PREGAO"]
        df = df[((d >= ini) & (d <= fim)).to_numpy()]
        if not len(df):
            continue
        df = filtrar_tipo_papel(df)
        if not len(df):
            continue
        partes.append(pd.DataFrame({
            "data": pd.to_datetime(df["DATA_PREGAO"].astype(str), format="%Y%m%d"),
            "CODISI": df["CODISI"].astype(str).str.strip(),
            "PREULT": df["PREULT"].astype(float),
            "VOLTOT": df["VOLTOT"].astype(float),
            "FATCOT": df["FATCOT"],
            "ESPECI": df["ESPECI"].astype(str),
        }))

    if not partes:
        # Havia arquivo (senão `_parquets_cotahist` teria levantado), mas nada
        # sobreviveu à janela e ao filtro de tipo. É resultado legítimo, e sai
        # com os dtypes de um painel cheio.
        painel = _painel_vazio()
    else:
        painel = (pd.concat(partes, ignore_index=True)
                  .sort_values(["data", "CODISI"]).reset_index(drop=True))

    if ajustar_eventos_societarios:
        eventos = detectar_eventos_societarios(painel)
        painel = aplicar_ajuste_societario(painel, eventos)
        painel.attrs["eventos_societarios"] = eventos
    return painel


def carregar_calendario(dir_dados, data_inicio: str = DATA_INICIO, data_fim: str = DATA_CORTE):
    """Calendário de pregões da janela, do COTAHIST BRUTO, ANTES de qualquer filtro.

    POR QUE ISTO EXISTE, e por que não pode sair do painel filtrado: derivar o
    calendário das datas do painel já filtrado por tipo de papel cria uma
    dependência CIRCULAR — universo -> calendário -> universo. A grade semanal
    passaria a depender do próprio filtro, e mudar a lista de raízes de
    `ESPECI` mudaria em que dias a grade cai, o que não tem nada a ver com o
    calendário da bolsa.

    Um pregão é um pregão mesmo que naquele dia nenhuma ação do filtro tenha
    negociado. Esta função lê SÓ `DATA_PREGAO`, de todos os registros, sem
    filtro nenhum.

    Args:
        dir_dados: pasta com os parquets `cotahist_*.parquet`.
        data_inicio: início da janela, ISO "AAAA-MM-DD".
        data_fim: fim da janela, ISO "AAAA-MM-DD". Teto: `DATA_CORTE`.

    CONTRATO DE TIPO: devolve sempre um `DatetimeIndex`. `DATA_PREGAO` é um
    inteiro `AAAAMMDD` na fonte e é convertido aqui — tudo que deriva semana
    ISO (`todas_as_semanas`, `grade_semanal`) depende disso.

    Returns:
        DatetimeIndex crescente, sem repetição, com todos os pregões da janela.

    Raises:
        ValueError: se a janela pedida ultrapassar `DATA_CORTE`, se não houver
            parquet em `dir_dados`, ou se a janela não contiver pregão algum.
    """
    _validar_janela(data_inicio, data_fim)
    ini, fim = _para_int(data_inicio), _para_int(data_fim)

    datas = set()
    for caminho in _parquets_cotahist(dir_dados, "carregar_calendario"):
        d = pd.read_parquet(caminho, columns=["DATA_PREGAO"])["DATA_PREGAO"]
        datas.update(d[((d >= ini) & (d <= fim)).to_numpy()].unique().tolist())

    if not datas:
        # Calendário vazio não é resultado utilizável: `grade_semanal()` falha
        # nele, e uma checagem do tipo `calendario.max() <= DATA_CORTE` compara
        # `NaT` e devolve False — que se lê como "há dado depois do corte",
        # exatamente o OPOSTO do que aconteceu. Falha aqui, onde a causa está.
        raise ValueError(
            f"carregar_calendario: os parquets de {dir_dados!r} não têm nenhum "
            f"pregão entre {data_inicio} e {data_fim}. Um calendário vazio não "
            f"é utilizável a jusante e faria a trava do holdout comparar NaT, "
            f"acusando violação onde o que houve foi ausência de dado."
        )

    return pd.DatetimeIndex(
        sorted(pd.to_datetime(pd.Series(sorted(datas)).astype(str), format="%Y%m%d"))
    )


def todas_as_semanas(calendario):
    """Última sessão efetivamente negociada de CADA semana ISO do calendário.

    É a grade semanal do projeto inteiro, sem recorte por data de decisão:
    `grade_semanal()` é esta função mais o corte em `D` e a exigência de
    warm-up. Existem separadas porque a mesma grade tem dois usos — janela de
    estimação (recortada por `D`) e eixo do backtest (a série inteira) — e
    duas implementações da mesma grade que divergissem em silêncio fariam os
    modelos estimarem sobre datas diferentes das que o motor executa.

    A CONVENÇÃO ("última sessão da semana") está registrada em `decisoes/01`,
    12/08/2026, como **convenção pura, sem justificativa de mecanismo**: não
    há razão de mercado que prefira sexta a terça, e nenhuma foi inventada. O
    que ela precisa ser é fixa e declarada antes de qualquer resultado.

    O que NÃO é convenção: tomar a última sessão **realmente negociada** em vez
    da sexta-feira de calendário. Isso é imposto pela regra de nunca inventar
    pregão — semana com feriado na sexta cai na quinta.

    Args:
        calendario: datas de pregão (DatetimeIndex ou iterável de datas).

    Returns:
        DatetimeIndex crescente, uma data por semana ISO presente.
    """
    cal = pd.DatetimeIndex(pd.Index(calendario).unique()).sort_values()
    if len(cal) == 0:
        return pd.DatetimeIndex([])

    # Chave de semana ISO como inteiro único (ano*100 + semana). Indexar por
    # tupla criaria um MultiIndex e o agrupamento cairia no ANO, não na semana.
    iso = cal.isocalendar()
    semana = iso["year"].to_numpy() * 100 + iso["week"].to_numpy()
    ultimos = pd.DataFrame({"data": cal, "semana": semana}).groupby("semana")["data"].max()
    return pd.DatetimeIndex(sorted(ultimos.to_numpy()))


def grade_semanal(calendario, data_decisao, n_semanas: int = N_SEMANAS_JANELA):
    """Devolve as `n_semanas` datas semanais de amostragem anteriores a `D`.

    A grade sai do CALENDÁRIO DE PREGÕES REAL, nunca de datas de calendário:
    cada ponto é a **última sessão efetivamente negociada** de uma semana ISO.
    Semana com feriado na sexta cai na quinta; nenhuma data devolvida deixa de
    ser pregão.

    QUAL DIA DA SEMANA, E POR QUÊ: a última sessão da semana. **A justificativa
    é convenção, e nada além disso** — não há mecanismo que prefira sexta a
    terça. Está escrito assim de propósito: inventar uma razão de mercado para
    uma escolha que é arbitrária seria pior do que admitir que é arbitrária. O
    que a convenção precisa ser é FIXA e declarada antes do resultado, e é.

    Todos os pontos são ESTRITAMENTE ANTERIORES a `D`, e a semana do próprio
    `D` não contribui ponto nenhum. Junto com `D` — exigido à parte — isso dá
    `n_semanas + 1` observações de preço, que são exatamente as `n_semanas`
    retornos semanais que os dois modelos consomem.

    Args:
        calendario: datas de pregão (DatetimeIndex ou iterável de datas).
        data_decisao: `D`, ISO ou Timestamp.
        n_semanas: tamanho da janela. Default `N_SEMANAS_JANELA` (152).

    Returns:
        DatetimeIndex com exatamente `n_semanas` datas, em ordem crescente.

    Raises:
        ValueError: se o calendário não tiver `n_semanas` semanas completas
            antes de `D` — warm-up insuficiente, e o motor não deve rodar.
    """
    d = pd.Timestamp(data_decisao)
    cal = pd.DatetimeIndex(pd.Index(calendario).unique()).sort_values()
    cal = cal[cal <= d]
    if len(cal) == 0:
        raise ValueError(f"não há pregão algum até a data de decisão {d!r}.")

    grade = todas_as_semanas(cal)
    grade = grade[grade < d]

    if len(grade) < n_semanas:
        raise ValueError(
            f"warm-up insuficiente em {d.date()}: a grade semanal tem "
            f"{len(grade)} semanas completas antes da data de decisão, e a "
            f"regra de elegibilidade exige {n_semanas} "
            f"(decisoes/01, 11/08/2026). Com N=152 o primeiro rebalanceamento "
            f"só é possível ~3 anos depois do início da série."
        )
    return grade[-n_semanas:]


def elegiveis_na_data(painel, data_decisao, calendario, n_semanas: int = N_SEMANAS_JANELA,
                      exigir_presenca_em_D: bool = EXIGIR_PRESENCA_EM_D):
    """Todos os ativos que passam no critério de completude, ANTES do cap.

    É aqui que a regra de elegibilidade mora de verdade; `definir_universo()`
    é esta função mais o corte no cap. Separadas para que o diagnóstico de
    `K_t` possa medir quantos ativos passam antes de o cap morder, sem
    reimplementar a regra e sem risco de as duas divergirem.

    Args:
        painel: DataFrame longo já filtrado por tipo de papel.
        data_decisao: `D`, ISO ou Timestamp.
        calendario: calendário de pregões BRUTO — ver `carregar_calendario()`.
        n_semanas: janela. Default 152.
        exigir_presenca_em_D: se True, exige presença nas `n_semanas` datas da
            grade E em `D` (`n_semanas + 1` observações, `n_semanas` retornos).
            Se False — o valor vigente, ver `EXIGIR_PRESENCA_EM_D` — exige
            presença só nas `n_semanas` datas da grade, cuja última é a última
            data de amostragem. É o relaxamento da escada de contingência,
            aplicado em 11/08/2026. EFEITO COLATERAL a declarar: com `n_semanas`
            observações saem `n_semanas − 1` retornos, um a menos que no ramo
            original. `n_semanas` em si NÃO foi tocado.

    Returns:
        Lista de `CODISI` ordenada por volume mediano decrescente, sem cap.
        Desempate por `CODISI` crescente, para saída determinística.
    """
    return list(volume_mediano_elegiveis(
        painel, data_decisao, calendario, n_semanas, exigir_presenca_em_D
    ).index)


def volume_mediano_elegiveis(painel, data_decisao, calendario,
                             n_semanas: int = N_SEMANAS_JANELA,
                             exigir_presenca_em_D: bool = EXIGIR_PRESENCA_EM_D):
    """A regra de elegibilidade, devolvendo TAMBÉM o volume mediano de cada um.

    É aqui que a regra é implementada de fato; `elegiveis_na_data()` é esta
    função descartando os volumes e `definir_universo()` é esta função mais o
    cap. As três existem separadas para que a regra tenha **uma implementação
    só** — o volume mediano é o critério de ordenação do cap e também o proxy
    de liquidez que `metricas.capacity()` consome, e recalculá-lo por fora
    abriria a porta para as duas contas divergirem em silêncio.

    Args:
        painel: DataFrame longo já filtrado por tipo de papel.
        data_decisao: `D`, ISO ou Timestamp.
        calendario: calendário de pregões BRUTO.
        n_semanas: janela. Default 152.
        exigir_presenca_em_D: ver `elegiveis_na_data()`.

    Returns:
        Series indexada por `CODISI` com o volume financeiro MEDIANO nas datas
        de observação da janela, ordenada por volume decrescente e desempatada
        por `CODISI` crescente. Series vazia se ninguém for elegível.
    """
    d = pd.Timestamp(data_decisao)
    cal = pd.DatetimeIndex(pd.Index(calendario).unique()).sort_values()

    if d not in set(cal):
        raise ValueError(
            f"a data de decisão {d!r} não é pregão do calendário informado. "
            f"A data de decisão tem de ser pregão — o motor não inventa pregão."
        )

    vazia = pd.Series(dtype=float, name="volume_mediano")
    vazia.index.name = "CODISI"

    grade = grade_semanal(cal, d, n_semanas)
    exigidas = pd.DatetimeIndex(grade.tolist() + [d]) if exigir_presenca_em_D else grade
    n_exigidas = len(exigidas)

    sub = painel[painel["data"].isin(set(exigidas)).to_numpy()]
    if not len(sub):
        return vazia

    # Um mesmo ISIN pode aparecer em mais de uma linha no mesmo pregão (mais de
    # um CODNEG). Consolidar antes evita contar presença duas vezes e evita
    # medir volume sobre linha repetida.
    por_dia = sub.groupby(["CODISI", "data"], sort=False)["VOLTOT"].sum()

    presenca = por_dia.groupby(level="CODISI").size()
    completos = presenca[presenca == n_exigidas].index
    if not len(completos):
        return vazia

    mediana = por_dia[por_dia.index.get_level_values("CODISI").isin(completos)]
    mediana = mediana.groupby(level="CODISI").median()

    ordenado = (
        mediana.rename("volume_mediano")
        .reset_index()
        .sort_values(["volume_mediano", "CODISI"], ascending=[False, True])
    )
    return ordenado.set_index("CODISI")["volume_mediano"]


def definir_universo(painel, data_decisao, calendario, n_semanas: int = N_SEMANAS_JANELA,
                     cap: int = CAP_UNIVERSO,
                     exigir_presenca_em_D: bool = EXIGIR_PRESENCA_EM_D):
    """Devolve o universo elegível NA data de decisão, com informação até ela.

    RETROATIVO, SEM EXCEÇÃO: tudo que esta função olha é anterior ou igual a
    `data_decisao`. Nenhum passo consulta preço, volume ou existência de papel
    depois de `D` — nem para montar a grade, nem para medir presença, nem para
    ordenar por volume. É por isso que o `calendario` também é truncado em `D`
    antes de qualquer coisa.

    A REGRA (`decisoes/01_decisoes_estruturais.md`, 11/08/2026):
      1. grade de amostragem: as `n_semanas` últimas datas semanais antes de
         `D` (última sessão real de cada semana) — ver `grade_semanal()`;
      2. presença em **100%** dessas datas **E** em `D`. Não é presença em
         100% dos pregões: isso seria contrabandear filtro de liquidez para
         dentro de uma regra de completude, e deixaria um blue chip suspenso
         por um dia inelegível por três anos;
      3. entre os completos, os `cap` maiores por **volume financeiro mediano**
         nas datas da janela. Ranking, não piso em R$: percentil é invariante
         a 29 anos de inflação. Não há filtro de volume separado — a liquidez
         é controlada pelo próprio cap, e se menos de `cap` ativos passarem no
         critério de completude o cap simplesmente não morde.

    NADA É IMPUTADO. Ausência numa data exigida é ausência: o ativo sai do
    universo naquela data de decisão. Não há forward-fill, não há tolerância,
    não há "quase completo".

    Args:
        painel: DataFrame longo já filtrado por tipo de papel, com colunas
            `data`, `CODISI` e `VOLTOT` (saída de `carregar_painel()`).
            Recebido por argumento, e não lido de disco aqui, para que a regra
            possa ser testada sobre dado inventado sem tocar em `dados/`.
        data_decisao: `D`, ISO ou Timestamp.
        calendario: calendário de pregões BRUTO, de `carregar_calendario()`.
            OBRIGATÓRIO e sem default derivado do painel: derivá-lo do painel
            filtrado criaria a dependência circular descrita em
            `carregar_calendario()`.
        n_semanas: janela. Default 152.
        cap: número máximo de ativos. Default 50.

    Returns:
        Lista de `CODISI`, ordenada por volume mediano decrescente. Desempate
        por `CODISI` crescente, para que a saída seja determinística.

    Raises:
        ValueError: se `D` não for pregão do calendário, ou se não houver
            warm-up suficiente (ver `grade_semanal()`).
    """
    return elegiveis_na_data(
        painel, data_decisao, calendario, n_semanas, exigir_presenca_em_D
    )[:cap]


# --- Elegibilidade em toda a série (para diagnóstico) ------------------------
#
# `elegiveis_na_data()` responde por UMA data e é a implementação de
# referência da regra. Para varrer ~1.300 datas de rebalanceamento ela
# refiltraria o painel inteiro toda vez. As funções abaixo calculam a MESMA
# regra de uma vez só, com janela rolante sobre a grade semanal.
#
# As duas têm de concordar, e `conferir_elegibilidade()` existe para provar
# isso em datas amostradas — duas implementações da mesma regra que divergem
# em silêncio seriam pior do que uma só.


def matriz_semanal(painel, grade):
    """Volume financeiro por ativo em cada data da grade. NaN = não negociou.

    Args:
        painel: DataFrame longo filtrado por tipo de papel.
        grade: DatetimeIndex com as datas de amostragem.

    Returns:
        DataFrame (datas da grade × CODISI) com o volume somado do dia. A
        ausência fica como NaN, e é isso que faz a janela rolante saber
        distinguir "não negociou" de "negociou zero".
    """
    sub = painel[painel["data"].isin(set(grade)).to_numpy()]
    matriz = sub.pivot_table(
        index="data", columns="CODISI", values="VOLTOT", aggfunc="sum"
    )
    return matriz.reindex(index=pd.DatetimeIndex(grade).sort_values())


def matriz_precos_semanal(painel, grade):
    """Preço de fechamento por ativo em cada data da grade. NaN = não negociou.

    É o painel de preços sobre o qual o backtest inteiro roda. O passo é
    SEMANAL porque a frequência decidida é semanal (`decisoes/01`, 11/08/2026)
    e porque é a mesma grade que os modelos consomem: as 153 observações de
    preço que terminam em `D` dão exatamente os 152 retornos semanais da
    janela de estimação. Rodar o motor em passo diário e estimar em passo
    semanal faria motor e modelos verem objetos diferentes.

    A ausência fica como **NaN e não é preenchida**. É isso que aciona, no
    motor, a regra de ativo sem preço (congelamento / liquidação) e o que faz
    `calcular_retornos()` devolver o gap INTEGRAL na reabertura.

    Args:
        painel: DataFrame longo filtrado por tipo de papel.
        grade: DatetimeIndex com as datas de amostragem semanal.

    Returns:
        DataFrame (datas da grade × CODISI) com `PREULT`. Mesmo ISIN com mais
        de um `CODNEG` no mesmo pregão é consolidado com `last`, a mesma
        convenção já usada por `carregar_desenho()`.
    """
    sub = painel[painel["data"].isin(set(grade)).to_numpy()]
    matriz = sub.pivot_table(
        index="data", columns="CODISI", values="PREULT", aggfunc="last"
    )
    return matriz.reindex(index=pd.DatetimeIndex(grade).sort_values())


def retornos_semanais_caixa(caixa_diario, grade):
    """Capitaliza o CDI DIÁRIO entre datas consecutivas da grade semanal.

    POR QUE CAPITALIZAR E NÃO SOMAR OU MEDIAR: o CDI é uma taxa **efetiva ao
    dia** e o rendimento de um período é o PRODUTO dos fatores diários, não a
    soma das taxas. O retorno do caixa entre duas datas `g[t-1]` e `g[t]` é
    `prod(1 + cdi_d) − 1` sobre os dias `d` em `(g[t-1], g[t]]`. Somar as
    taxas subestimaria; usar a média multiplicada pelo número de dias erraria
    nas semanas curtas (feriado) e nas longas (emenda).

    Isso é **leitura de como o instrumento funciona**, não escolha de
    parâmetro: qualquer outra agregação estaria simplesmente errada sobre uma
    taxa efetiva ao dia.

    O resultado sai indexado nas datas da grade a partir da SEGUNDA — igual ao
    que `calcular_retornos()` faz com os preços —, para que as duas séries
    fiquem alinhadas sem ninguém deslocar nada por fora.

    Args:
        caixa_diario: Series de retorno decimal diário do caixa, saída de
            `carregar_caixa()`.
        grade: DatetimeIndex com as datas de amostragem semanal.

    Returns:
        Series de retorno semanal do caixa, indexada por `grade[1:]`.

    Raises:
        ValueError: se alguma data da grade não existir na série do CDI. Não
            preenchemos buraco de série de caixa — sem a taxa daquele período,
            a remuneração seria inventada.
    """
    caixa = pd.Series(caixa_diario).sort_index()
    g = pd.DatetimeIndex(pd.Index(grade).unique()).sort_values()
    if len(g) < 2:
        raise ValueError(
            f"a grade precisa de ao menos 2 datas para haver um retorno de "
            f"caixa entre elas, recebida com {len(g)}."
        )

    faltam = g.difference(pd.DatetimeIndex(caixa.index))
    if len(faltam):
        raise ValueError(
            f"a série do CDI não cobre {len(faltam)} data(s) da grade semanal, "
            f"a primeira em {faltam[0]!r}. O motor não preenche buraco de série "
            f"de caixa: sem a taxa daquele período, a remuneração do caixa "
            f"residual e de uma eventual liquidação seria inventada."
        )

    fator = (1.0 + caixa).cumprod()
    na_grade = fator.reindex(g)
    return (na_grade / na_grade.shift(1) - 1.0).iloc[1:]


def elegibilidade_rolante(matriz, n_semanas: int = N_SEMANAS_JANELA):
    """Volume mediano da janela, por data da grade, só onde a presença é 100%.

    A janela é de `n_semanas + 1` observações: as `n_semanas` datas anteriores
    mais a própria data — exatamente o que `elegiveis_na_data()` exige, e
    exatamente as `n_semanas` retornos semanais que os modelos consomem.

    O critério de completude sai de graça de `min_periods`: o `rolling` do
    pandas não conta NaN, então uma janela com qualquer ausência devolve NaN.
    Não há checagem separada de presença — é a mesma condição.

    Args:
        matriz: saída de `matriz_semanal()`.
        n_semanas: janela. Default 152.

    Returns:
        DataFrame do mesmo formato: volume mediano onde o ativo é elegível,
        NaN onde não é. `notna().sum(axis=1)` dá a série `K_t`.
    """
    janela = n_semanas + 1
    return matriz.rolling(janela, min_periods=janela).median()


def ranking_rolante(medianas, data, cap: int = CAP_UNIVERSO):
    """Universo de uma data a partir da janela rolante, COM o volume mediano.

    É o equivalente rápido de `definir_universo()` para varrer ~1.300 datas:
    a regra já foi aplicada uma vez em `elegibilidade_rolante()`, e aqui só se
    ordena e corta no cap. `conferir_elegibilidade()` existe para provar, em
    datas amostradas, que os dois caminhos dão a MESMA lista.

    Devolve o volume junto porque ele tem dois usos e precisa ser o mesmo nos
    dois: é o critério de ordenação do cap (`decisoes/01`, 11/08/2026) e é o
    proxy de liquidez que `metricas.capacity()` consome. Recalculá-lo por fora
    abriria a porta para o cap ordenar por um número e a capacidade medir outro.

    Args:
        medianas: saída de `elegibilidade_rolante()`.
        data: data de decisão, presente no índice de `medianas`.
        cap: número máximo de ativos. Default 50.

    Returns:
        Series indexada por `CODISI` com o volume mediano, ordenada por volume
        decrescente, desempatada por `CODISI` crescente, truncada no cap.
    """
    linha = medianas.loc[data].dropna()
    ordenado = (
        linha.rename("volume_mediano")
        .reset_index()
        .sort_values(["volume_mediano", "CODISI"], ascending=[False, True])
    )
    return ordenado.set_index("CODISI")["volume_mediano"].iloc[:cap]


def conferir_elegibilidade(painel, calendario, medianas, datas,
                           n_semanas: int = N_SEMANAS_JANELA, cap: int = CAP_UNIVERSO):
    """Confere a janela rolante contra `definir_universo()` nas datas dadas.

    Returns:
        DataFrame com uma linha por data conferida e a coluna `confere`.
    """
    linhas = []
    for d in datas:
        referencia = definir_universo(painel, d, calendario, n_semanas, cap)
        rolante = list(ranking_rolante(medianas, d, cap).index)
        linhas.append({
            "data": d,
            "n_referencia": len(referencia),
            "n_rolante": len(rolante),
            "confere": referencia == rolante,
        })
    return pd.DataFrame(linhas)


def cobertura_diaria(painel, calendario, isins, data_inicio, data_fim):
    """% de pregões da janela em que cada ativo efetivamente negociou.

    Mede COBERTURA DIÁRIA, não semanal: serve para responder se um ativo que
    passou na grade semanal negocia de fato todo dia, ou se está passando por
    negociar só no dia em que a grade cai.

    Args:
        painel: DataFrame longo filtrado por tipo de papel.
        calendario: calendário de pregões bruto.
        isins: iterável de CODISI a medir.
        data_inicio, data_fim: extremos da janela, inclusive.

    Returns:
        Series indexada por CODISI com a fração de pregões negociados.
    """
    cal = pd.DatetimeIndex(pd.Index(calendario).unique()).sort_values()
    cal = cal[(cal >= pd.Timestamp(data_inicio)) & (cal <= pd.Timestamp(data_fim))]
    total = len(cal)
    if total == 0:
        raise ValueError("janela sem pregão algum para medir cobertura.")

    alvo = set(isins)
    sub = painel[
        painel["CODISI"].isin(alvo).to_numpy()
        & painel["data"].isin(set(cal)).to_numpy()
    ]
    negociados = sub.groupby("CODISI")["data"].nunique()
    return (negociados.reindex(sorted(alvo)).fillna(0) / total).sort_values()
