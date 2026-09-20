# O alvo do treino, medido

Gerado por `diagnostico/inspecionar_dados.py` sobre `bressay_split/splits/train.tsv`
(74.882 amostras). Todas as alturas sao em pixels da imagem crua, antes de
qualquer pre-processamento.

## Resolucao da fonte

| | BRESSAY (words) | IAM (words) |
|---|---|---|
| altura nativa mediana | **31 px** | **70 px** |
| fator para chegar na altura 64 do modelo | **2,06x ampliacao** | 0,91x reducao |

O modelo trabalha em 64x256. O BRESSAY chega em 31 px e precisa ser **ampliado
2x** para caber. Ampliar nao cria informacao: o borrao do alvo e produzido pelo
proprio redimensionamento, antes de o treino comecar.

Nao existe subconjunto nitido para o qual filtrar:

| corte | amostras que sobram |
|---|---|
| altura >= 40 px | 3.299 (4,4%) |
| altura >= 48 px | 667 (0,9%) |
| altura >= 56 px | 97 (0,1%) |
| altura >= 64 px | **1** |

Os recortes vindos das 110 paginas de alta resolucao (>= 1000 px de largura)
tem altura mediana 32 px, igual aos das paginas pequenas -- os recortes de
palavra foram reduzidos para um tamanho uniforme quando o dataset foi cortado,
e o dataset nao publica as coordenadas das caixas, so as imagens ja recortadas.
`lines/` tem altura mediana 20 px, pior ainda.

## Linha pautada

73% dos alvos de treino tem a linha pautada do papel atravessando a imagem de
ponta a ponta. O modelo esta sendo treinado para desenhar um sublinhado embaixo
de cada palavra.

Cobertura de tinta mediana: 24,8% (normal, comparavel ao IAM).

## O pre-processamento nao e o culpado

Hipotese testada e **refutada**: em `diffusionpen_mods/utils/bressay_dataset.py`
a linha que normaliza a altura para 64 esta comentada (linha 110), e o ramo
`else` faz `resize((256,64))` sem preservar aspecto. Medido: **100%** das
imagens caem no ramo `ImageOps.pad`, que preserva o aspecto. Distorcao mediana
1,00, nenhuma imagem fora da faixa 0,9-1,1. O ramo com defeito nunca e
acionado porque nenhum recorte chega a 256 px de largura.

As duas folhas de contato (`bressay_como_o_modelo_ve.png` e
`mesmo_recorte_no_padrao_iam.png`) sao visualmente equivalentes: passar os
mesmos recortes pelo pre-processamento do IAM nao melhora nada. A normalizacao
de contraste por percentis escurece um pouco, mas os recortes ilegiveis sao
ilegiveis nos dois caminhos.

## Diacriticos disponiveis no split de treino

10,1% das amostras (7.560) tem pelo menos um diacritico.

| | ã | á | ç | í | ó | é | ú | ê | õ | â |
|---|---|---|---|---|---|---|---|---|---|---|
| amostras | 2.538 | 1.194 | 1.576 | 702 | 560 | 541 | 354 | 249 | 221 | 113 |

Com `--max_samples 20000` isso vira ~2.000 amostras com diacritico, das quais
~30 com `â` e ~60 com `õ`.

Os marcadores de ilegibilidade do BRESSAY (`@@???@@` e afins) **nao** aparecem
no split: esse filtro ja esta correto.

## Consequencia

O fine-tune nao esta destruindo o modelo -- esta aprendendo o alvo com
fidelidade. O alvo e borrado, sublinhado e de baixa resolucao. Por isso o
resultado piora com o numero de passos e e praticamente insensivel ao `lr`, e
por isso a deriva parou de prever a qualidade da imagem.
