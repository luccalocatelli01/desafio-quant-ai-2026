# DECISÃO D03 — AUSÊNCIA DE HOLDOUT (declarada)
**Data:** 2026-08-15
**Decisão:** este projeto NÃO reserva partição de holdout. Todo o período
disponível (1995-01 a 2026-07) é usado em treino ou teste.
**Mecanismo:** o pré-registro de holdout do projeto anterior (novelo) foi
verificado e está EM BRANCO — nenhum modelo foi nomeado nele. Herdá-lo sem
reescrever seria afirmação falsa. Reservar 19 meses reduziria o teste de 103
para 84 meses; optou-se pelo poder estatístico, aceitando a perda da
reivindicação de pré-registro.
**Consequência assumida:** nenhum texto deste projeto — dossiê ou PDF — pode
afirmar ou insinuar que existe partição reservada, cofre, ou resultado fora da
amostra em sentido estrito. O período 2018-2026 é chamado de "teste".
**O que substitui a disciplina perdida:** o freeze datado do MODEL_SPEC na
célula B6, gravado em decisoes\ ANTES de qualquer backtest rodar (C1). Qualquer
alteração no MODEL_SPEC posterior a C1 é declarada como escolha pós-resultado.
**Reversível:** não.
