# config.py — constantes mínimas para satisfazer o import de src/dados.py
# Este projeto NÃO tem holdout (decisão D03). DATA_CORTE existe aqui apenas
# porque dados.py a importa no topo do arquivo; ela NÃO particiona nada e
# NÃO é usada por detectar_eventos_societarios nem por aplicar_ajuste_societario.
# PROIBIDO usar estas constantes para dividir treino/teste. O split é D02
# (treino ate 2017-12, teste 2018-01 a 2026-07) e vive na celula A7.

from pathlib import Path

DATA_INICIO = "1995-01-01"
DATA_CORTE  = "2026-08-07"  # fim da base, NAO fronteira de holdout
DIR_DESENHO = Path(__file__).resolve().parent.parent / "dados"
