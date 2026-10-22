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

Para uma visão do trabalho inteiro, fase por fase, com os números e uma figura
de cada etapa, abra [`resultados/README.md`](resultados/README.md).

O melhor modelo é o DiffusionPen com o acento separado na entrada de texto, um
professor (cópia congelada do modelo original) e peso no acento e nas vogais,
treinado por 40 épocas sobre uma base de acentos sintéticos. Na validação,
pelo olho de duas pessoas e pelos detectores:

| | modelo original | melhor modelo |
|---|---|---|
| acento pedido presente | não desenha | 89% |
| cedilha pedida desenhada | não desenha | 62 de 63 |
| acento em palavra que não pede | 5% | no nível do original |
| letra das palavras sem acento (CER) | 0,195 | 0,186 |

Em aberto ficam o acento a mais dentro das palavras acentuadas (15%), a letra
das palavras acentuadas, pior que a da base de treino, e uma medida automática
da posição do acento.

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
| [docs/13_investigacao_do_vazamento_e_da_letra.md](docs/13_investigacao_do_vazamento_e_da_letra.md) | testes pequenos que levaram às mudanças seguintes |
| [docs/14_acento_separado_e_professor.md](docs/14_acento_separado_e_professor.md) | acento separado e professor |
| [docs/15_acento_a_mais_e_variantes_do_professor.md](docs/15_acento_a_mais_e_variantes_do_professor.md) | zona vogais, treino longo, variantes e estado atual |
| [avaliacao_diacriticos/anotacao_40ep.md](avaliacao_diacriticos/anotacao_40ep.md) | anotação humana do melhor modelo |

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
