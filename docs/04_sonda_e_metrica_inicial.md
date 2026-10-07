# 04 — Fases 1 e 2: a sonda do modelo original e a métrica E1/E2

## Fase 1 — A sonda: como o modelo original falha

**Pergunta.** O DiffusionPen do IAM, sem nenhum ajuste, desenha os
diacríticos do português? Se não, como falha?

**Desenho do experimento.**
- **Lista fixa de palavras** (`comum/palavras.py`), em 4 grupos de 5:
  - **controle**, só ASCII: casa, pato, vento, nacao, coracao;
  - **til**: mão, pão, nação, irmã, põe;
  - **cedilha**: coração, ação, março, praça, força;
  - **agudo e circunflexo**: você, café, três, histórico, português.
- **3 sementes** (0, 1, 2), para separar falha sistemática de acaso.
- **Pares mínimos** ("nacao"/"nação", "coracao"/"coração"), gerados com o
  **mesmo estilo e a mesma semente**. É a comparação mais informativa: a única
  diferença entre as duas imagens é o acento no texto.
- **Patch aditivo** no DiffusionPen (`sonda/patches_diffusionpen.diff`), com um
  modo `sonda` que lê um manifesto e gera cada item com estilo e semente
  fixos. O modo original sorteava um estilo novo por palavra, o que inviabiliza
  a comparação por pares.

**Por que controle ASCII.** Se as palavras sem acento falhassem, o problema
seria o ambiente (pesos, versão), e não os diacríticos. Todo o resto da sonda
seria inválido.

**Resultados** (folhas em `saidas/diffusionpen/`):
- **O controle sai legível.** Em inglês, as amostras geradas são quase
  indistinguíveis das reais do IAM (`comparacao_real_vs_gerado.png`).
- **As palavras acentuadas falham de dois modos:**
  - **omissão limpa:** o sinal some e o resto sai certo. `café` → "cafe",
    `português` → "portugues", `pão` → "pao";
  - **degradação da palavra inteira:** `coração` → "ccrcemo", `praça` →
    "praxa", `põe` → "pae".
- **O caractere chega ao modelo.** `nacao` e `nação`, com o mesmo estilo e a
  mesma semente, **geram imagens diferentes**. Se o acento fosse filtrado na
  entrada, seriam idênticas. A falha é de desenho, não de entrada.

**Conclusão.** O modelo sabe que o texto pede algo diferente, mas nunca
aprendeu a desenhar o sinal. É preciso ensinar, e é isso que as fases
seguintes tentam.

## Fase 2 — A métrica E1/E2 (`avaliacao_diacriticos/`)

**Pergunta.** Como medir automaticamente se o acento foi desenhado e se a
palavra continua legível?

**Ideia.** Pontuar cada imagem em dois eixos independentes:
- **E1 — presença:** há tinta na faixa onde o sinal deveria estar? Acima da
  altura-x para til, agudo, grave e circunflexo; abaixo da linha de base para
  a cedilha.
- **E2 — integridade:** a palavra continua legível, descontando o acento?
  Medido pelo CER de um reconhecedor (TrOCR), com texto e leitura convertidos
  para ASCII.

O cruzamento dá quatro categorias:

|  | base legível | base degradada |
|---|---|---|
| **acento presente** | acerto | degradação com diacrítico |
| **acento ausente** | omissão | degradação sem diacrítico |

**O que foi feito e medido** (detalhes em `avaliacao_diacriticos/README.md` e
`ESTADO.md`):
- **Os pares saem alinhados:** a mesma palavra com e sem acento, com a mesma
  semente, sai na mesma posição (deslocamento 0, IoU mediano 0,93). Isso
  permite medir o E1 por **diferença de imagens** entre os gêmeos, mais
  robusto do que estimar a geometria da palavra.
- **31 testes sintéticos de resposta conhecida** para a geometria
  (`testes_metrica.py`, o único conjunto de testes automatizados do
  repositório).
- **A linha pautada do papel era medida como acento.** Foi corrigida
  removendo faixas finas (até 5 px). O controle positivo caiu de 0,861 para
  0,510, e foi reportado assim, sem mexer no limiar.

**Os três resultados que importam:**
1. **O E1 por faixa é um detector fraco em dado real** (AUC 0,68 contra um
   controle negativo do mesmo domínio). Contra o IAM o AUC parecia 0,85, mas
   boa parte era diferença de domínio, não de acento. **O E1 por diferença**
   separa bem til e grave, mas não agudo e cedilha, em que o ruído supera um
   acento nominal.
2. **O E2 não funciona com o TrOCR:** em escrita humana real, ele dá CER 0,83
   nas acentuadas e 1,04 nas ASCII, com 1 leitura perfeita em 120. A
   classificação em quatro categorias fica instável.
3. **A ferramenta de concordância humana (kappa)** está pronta, com uma
   planilha de 38 linhas estratificadas e a folha de contato, mas **a anotação
   humana não foi feita**.

**O que mudou depois.** Para avaliar os fine-tunes com acentos (fases 4 em
diante), construímos instrumentos mais simples e mais robustos:
- um **detector de marcas soltas** no lugar do E1;
- um **leitor CTC próprio**, treinado no IAM, no lugar do TrOCR;
- a **métrica de posição do acento**.

Todos estão descritos em [08_protocolo_de_avaliacao.md](08_protocolo_de_avaliacao.md).
O E1/E2 continua no repositório como trabalho da frente "métrica". A anotação
humana é uma pendência útil para validar qualquer uma das métricas.
