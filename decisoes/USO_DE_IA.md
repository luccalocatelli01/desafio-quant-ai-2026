# Bloco de IA Generativa — pronto para a página 5 (~150 palavras)

*(Fonte: `DOSSIE.md` §5 — log célula a célula, em tempo real, com a coluna "erro da IA?" preenchida e três erros de IA documentados. Forma de três marcadores por finalidade, recomendada pelo conselho. Nada abaixo foi escrito depois do fato: cada item tem data no log.)*

---

**Como usamos IA generativa — e contra quem ela apontou:**

- **Para desconfiar da fonte.** A IA mediu em vez de supor: detectou que a regra de classes herdada excluía preferenciais `PNA*`/`PNB*` em silêncio e mediu o delta antes de corrigir (**+57.914 linhas, +2,71%** — 15/08); refez a ponte de ISIN que falhava em 334 códigos (**540→607 casamentos, zero falso positivo** — 15/08).

- **Para desconfiar de nós.** O humano **proibiu** a IA de consertar o módulo de dados quando ele quebrou — a correção foi feita por fora, com cobertura de chave verificada (**714/714** — 15/08); e adotou a versão final que rende **menos** (V5, **−0,038 pp/ano** vs V4 — 16/08) por ser a correta.

- **Para atacar o próprio modelo.** Quatro rodadas de conselho adversarial com revisores cegos e árbitros (**10, 12 e 9 agentes** — 15–16/08) derrubaram o achado central; hipóteses da própria IA foram refutadas pela medição que ela mesma exigiu (delta de cobertura **0,00 pp**, 1996–2026 — 15/08); o teste de sorte foi endurecido por decisão própria até **p = 0,7350** (16/08). **Três erros de IA** estão documentados no log, com data e correção.

**Arquitetura de uso:** um orquestrador (escreve os prompts de célula e mantém o dossiê; não executa código) → um executor (Claude Code, uma célula por vez; o humano confere cada sanidade antes da próxima) → conselhos adversariais de revisores cegos (rodadas 2–4). **Nenhum LLM entra na estratégia** — a IA construiu, mediu e auditou; não decide alocação.
