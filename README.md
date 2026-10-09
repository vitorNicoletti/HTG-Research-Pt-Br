# HTG-Research-Pt-Br

Fidelidade de diacríticos na geração de escrita manuscrita (HTG) em português brasileiro.

Projeto de Creative Experience: Transformative Project II — Ciência da Computação, PUCPR, Turma B, Equipe 13.
Vitor Nicoletti · Vinícius Y. Borges · Bruno H. O. M. Dutra · Leonardo Saito

---

## Em uma frase

Modelos de geração de manuscrito treinados em inglês não desenham os acentos
do português, e as métricas usuais não enxergam isso. Este repositório mede
essa falha, tenta corrigi-la com fine-tune do **DiffusionPen** e desenvolve
formas de avaliar especificamente os diacríticos.

## Onde estamos

- **O que funciona:** com acentos sintéticos desenhados sobre palavras do IAM
  e um **peso maior no acento na loss**, o modelo passa a desenhar diacríticos
  a partir do texto. O modelo original não faz isso. O efeito é causal e
  medido: +7 pontos de diferença pareada, IC95 +3 a +11.
- **Problemas em aberto:**
  - o modelo põe acento também onde não deve;
  - erra a letra do agudo;
  - a letra fica menos legível sempre que dados com acento sintético entram
    no treino.

## Documentação

**Comece por [`docs/README.md`](docs/README.md)**: um resumo executivo e a
ordem de leitura de todos os documentos.

| | |
|---|---|
| [docs/01_problema_e_contexto.md](docs/01_problema_e_contexto.md) | problema, bases de dados, termos |
| [docs/02_diffusionpen_e_modificacoes.md](docs/02_diffusionpen_e_modificacoes.md) | o modelo e **todas as mudanças no treino oficial** |
| [docs/07_experimentos_com_acentos.md](docs/07_experimentos_com_acentos.md) | cada experimento: motivo, mudança, resultado |
| [docs/09_resultados_consolidados.md](docs/09_resultados_consolidados.md) | todas as tabelas |
| [docs/10_problemas_em_aberto.md](docs/10_problemas_em_aberto.md) | onde estamos travados e o que testar |
| [docs/11_como_rodar.md](docs/11_como_rodar.md) | guia prático |

Registros históricos: [`ACHADOS.md`](ACHADOS.md), com os achados na ordem em
que apareceram, e [`LOG.md`](LOG.md), o diário de execução.

## Início rápido

```bash
git clone https://github.com/koninik/DiffusionPen.git DiffusionPen   # o modelo (fora do git)
bash scripts/aplicar_mods.sh                                          # aplica as nossas modificações
python diagnostico/teste_direcao_gradiente.py                         # valide a GPU antes de treinar
python scripts/treinar.py experimentos/iam_pt_peso5.json --dry-run    # um experimento
```

Pesos, dados e todos os passos estão em [docs/11_como_rodar.md](docs/11_como_rodar.md).

> **Antes de treinar em qualquer GPU nova, rode os testes de `diagnostico/`.**
> A primeira placa usada (RX 6600 XT) calculava o gradiente na direção errada,
> e isso invalidou todos os treinos feitos nela
> ([docs/03](docs/03_ambiente_e_hardware.md)).
