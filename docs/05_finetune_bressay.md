# 05 — Fase 3: fine-tune no BRESSAY

## Motivação

O caminho mais direto para ensinar diacríticos é treinar em escrita real em
português. O BRESSAY é o maior dataset público de manuscrito em português, com
416.826 recortes de palavras, e cerca de 10% delas têm diacrítico.

## Preparação

- **Split** (`scripts/preparar_split.py` → `bressay_split/`):
  - escritores disjuntos entre treino, validação e teste, seguindo a partição
    oficial por página;
  - filtros de qualidade, entre eles descartar palavras com menos de 28 px de
    altura;
  - 647 escritores e 74.882 palavras de treino.
- **Split reduzido** (`scripts/reduzir_split.py` → `bressay_split_25/`): 25%
  das palavras de cada escritor, com no mínimo 5 por escritor. São 18.724
  palavras de treino, 10,2% com diacrítico. Serve para iterar mais rápido.
- **Extrator de estilo:** chegamos a treinar um no BRESSAY, ainda na placa
  defeituosa (ver [03](03_ambiente_e_hardware.md)). Os treinos válidos usam o
  extrator do IAM.

## Os runs

| run | máquina | dados | resultado |
|---|---|---|---|
| vários, até 40 épocas | RX 6600 XT | split inteiro ou 20 mil amostras | **inválidos:** a placa calculava o gradiente na direção errada |
| `model_bressay_25` (`experimentos/bressay_25.json`) | RX 9060 XT | split de 25%, pré-processamento v1, 26 épocas | pior que o modelo original em tudo, **inclusive no controle ASCII** |
| `model_bressay_25_v2` (`experimentos/bressay_25_v2.json`) | RX 9060 XT | split de 25% filtrado (tinta ≥ 14 px), pré-processamento v2, 40 épocas | o formato da saída melhorou, mas **nenhum acento aprendido** |

## Por que não funcionou: a resolução do BRESSAY

Varredura de todas as imagens (`diagnostico/resolucao_bressay.py`):

| tipo | quantidade | altura mediana | p90 |
|---|---|---|---|
| palavras | 416.826 | **26 px** | 38 px |
| linhas | 30.090 | 21 px | 38 px |

As palavras do IAM têm ~50 px. **No BRESSAY, um til ou uma cedilha ocupa 1–2
pixels da imagem original.** Depois de redimensionado para os 64 px do modelo,
o que ele recebe é uma palavra pequena e pixelada, muitas vezes com a linha da
pauta embaixo. O fine-tune aprende fielmente essa aparência, e quase não há
acento visível para aprender.

Filtrar só as páginas de melhor resolução não é viável: com corte em 35 px por
página sobram 12 páginas de treino. A distribuição é estreita demais.

## O pré-processamento v2

Detalhado em [05b_pre_processamento_bressay.md](05b_pre_processamento_bressay.md).
- **O que faz:** amplia para 64 px, remove a linha pautada (sem apagar traços
  que a cruzam), recorta justo na tinta e escala como no IAM.
- **O que corrigiu:** o defeito de formato. O v1 gerava palavras minúsculas
  com a pauta embaixo.
- **O que não corrigiu:** os modelos continuaram piores que o original, e no
  par mínimo "nacao"/"nação" o til era ignorado.

## Dois resultados laterais importantes

**1. O fine-tune apagou o que o modelo sabia (esquecimento catastrófico).**
Na sonda com o escritor 12 do IAM e as mesmas referências da fase 1:
- o controle ASCII, perfeito antes, ficou ilegível ou quase;
- o modelo deixou de seguir o estilo do escritor.

**2. Reduzir a resolução não ajuda: o VAE perde o acento em 32×128**
(`diagnostico/teste_vae_resolucao.py`, figura `saidas/diffusionpen/vae_resolucao.png`).
Passando palavras do BRESSAY e do RIMES (francês) pelo VAE, ida e volta:
- **em 64×256** os acentos voltam intactos;
- **em 32×128** o agudo some e as palavras borram.

Ou seja, em 64×256 o VAE **não** é o gargalo. O problema está no que o UNet
aprende a partir de dados de baixa resolução.

## O critério de "deriva" não serve

O `scripts/medir_deriva.py` mede quanto os pesos se afastaram dos do IAM
(‖W − W_IAM‖ / ‖W_IAM‖) e tinha uma "faixa-alvo" de 2,5–4,0%, calibrada na
placa defeituosa.
- No `bressay_25`, com deriva de 2,2%, o modelo já estava pior que o original.
- No v2, de 30 a 40 épocas (deriva de 2,5% a 2,9%), dentro da "faixa-alvo",
  nenhum modelo gera bem.

**A deriva continua sendo registrada, mas não é critério de parada.** O
critério são as amostras e a avaliação.

## Conclusão da fase

Com os pixels do BRESSAY não dá para ensinar o acento. A resolução é um
resultado em si: o maior dataset público de manuscrito em português tem
palavras de ~26 px, em que o diacrítico ocupa 1–2 pixels.

O próximo passo foi **ensinar o acento sem depender desses pixels**, com
acentos sintéticos desenhados em palavras do IAM, que têm resolução boa
([06](06_acentos_sinteticos.md) e [07](07_experimentos_com_acentos.md)).
