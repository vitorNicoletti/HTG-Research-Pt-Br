# Pré-processamento: do recorte do BRESSAY ao alvo de treino

Como uma imagem do BRESSAY vira o alvo que o DiffusionPen tenta reproduzir.
Código em `diffusionpen_mods/utils/bressay_dataset.py` (`load_image`).

![etapas do pipeline](figuras/pipeline_etapas.png)

## As quatro etapas

| # | o que faz | onde |
|---|---|---|
| 1 | Lê o PNG recortado do dataset | `Image.open(...).convert("RGB")` |
| 2 | Estica o contraste por percentis | percentil 3 → preto, percentil 40 → branco |
| 3 | Enquadra em 256×64 | `ImageOps.pad`, preservando o aspecto |
| 4 | Converte em tensor | `ToTensor` + `Normalize(0.5, 0.5)` → faixa [−1, 1] |

Depois disso o VAE do Stable Diffusion comprime a imagem 8× e o modelo trabalha
num latente de **4×8×32**. As 5 imagens de estilo de cada amostra passam pelo
mesmo caminho.

### Etapa 2 — por que percentis, e não Otsu

Toma o percentil 3 dos cinzas (a tinta mais escura) e manda para preto; toma o
percentil 40 (o papel) e manda para branco; estica linearmente entre os dois. O
40 funciona como "papel" porque a tinta ocupa ~25% dos pixels de um recorte.

**Os cinzas intermediários sobrevivem** — é isso que diferencia de binarizar. A
escolha herda do IAM, cujas imagens o modelo pré-treinado viu em tons de cinza.
Comparação com Otsu nos mesmos recortes:
`../diagnostico/resultados/dados/normalizacao_vs_otsu.png`.

### Etapa 3 — onde o borrão nasce

O modelo trabalha em 256×64. Os recortes do BRESSAY chegam com **31 px de
altura** (mediana) e precisam ser **ampliados 2,06×**. Ampliar não cria
informação: o borrão do alvo é produzido aqui, antes de o treino começar.

| | BRESSAY | IAM |
|---|---|---|
| altura nativa mediana | **31 px** | **70 px** |
| fator até a altura 64 | **amplia 2,06×** | reduz 0,91× |
| largura por caractere | 14 px | — |

Não há subconjunto nítido para o qual filtrar. Das 74.882 amostras de treino:

| corte de altura | sobram |
|---|---|
| ≥ 40 px | 3.299 (4,4%) |
| ≥ 56 px | 97 |
| ≥ 64 px | **1** |

Os recortes vindos das 110 páginas de alta resolução também têm 32 px: as
palavras foram reduzidas a um tamanho uniforme quando o dataset foi cortado, e
o BRESSAY não publica as coordenadas das caixas, só as imagens já recortadas.
`lines/` tem 20 px de altura, é pior.

## A linha pautada

**73% dos alvos de treino** têm a linha do caderno atravessando a imagem de
ponta a ponta. Não é tinta do escritor — e o modelo aprende a desenhá-la: ela
aparece embaixo de quase toda palavra gerada.

Ela também quebra a métrica de diacríticos, que estima o corpo da palavra pelas
fileiras com mais tinta: a pauta é a fileira mais cheia, e o corpo estimado cai
de 25 px para 12 px. Figura em
`../diagnostico/resultados/dados/pauta_explicada.png`; correção já aplicada em
`avaliacao_diacriticos/metrica.py`.

## O que foi descartado antes do split

O `scripts/preparar_split.py` cortou **288.152 das 416.826** palavras:

| motivo | descartadas |
|---|---|
| altura < 28 px | **234.007** |
| menos de 2 letras | 47.556 |
| marcação de ilegibilidade (`@@???@@`) | 6.169 |
| tinta fora da faixa | 420 |

O corte de altura está quase em cima da mediana da distribuição (31 px), então
é um limiar sobre um contínuo, não uma separação entre bom e ruim.

Sobraram 74.882 treino / 23.539 validação / 30.253 teste, **disjuntos por
escritor**.

## Diacríticos disponíveis no treino

10,1% das amostras (7.560) têm pelo menos um:

| ã | á | ç | í | ó | é | ú | ê | õ | â |
|---|---|---|---|---|---|---|---|---|---|
| 2.538 | 1.194 | 1.576 | 702 | 560 | 541 | 354 | 249 | 221 | **113** |

## Pontos para discutir

1. **O teto de nitidez é do dado, não do treino.** Nenhuma taxa de aprendizado
   ou número de épocas produz nitidez a partir de um alvo ampliado 2×. O
   critério de sucesso deveria ser o diacrítico aparecer, não a imagem ficar
   bonita.
2. **Binarizar o alvo** é a única das mudanças propostas que altera o alvo — e o
   modelo já provou que aprende o alvo com fidelidade. Não foi testado.
3. **Remover a pauta do alvo de treino**, já que hoje o modelo a aprende.
4. **Baixar o corte de altura** de 28 para ~22 triplicaria o conjunto. Não deve
   melhorar a legibilidade, mas melhoraria a cobertura de diacríticos raros —
   hoje são 113 amostras com `â` no split inteiro.
5. **Declarar a limitação** no texto: til e cedilha quase não têm gêmeo ASCII
   que seja palavra real no corpus (só `sã`, `fã`, `dã`), justamente as marcas
   que interessam.

Medições completas em `../diagnostico/resultados/dados/RESUMO.md`, reprodutíveis
com `python diagnostico/inspecionar_dados.py`.
