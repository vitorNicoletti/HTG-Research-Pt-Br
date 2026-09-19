# Métrica de avaliação de diacríticos

Instrumento para medir se um gerador de manuscrito desenha os diacríticos do
português. Não depende de o fine-tune ter dado certo: ele mede, e o resultado
pode perfeitamente ser "não deu".

Nada aqui toca o treino nem `diffusionpen/DiffusionPen/train.py`.

## A ideia

A sonda gera **pares mínimos** ("nacao" / "nação") com estilo e semente fixos,
de modo que a única variável entre os gêmeos seja o diacrítico. Cada imagem é
pontuada em dois eixos **independentes**:

- **E1 — presença**: há tinta na faixa esperada do diacrítico? Acima da
  altura-x para til, agudo, grave e circunflexo; abaixo da linha de base para
  cedilha.
- **E2 — integridade da base**: a palavra continua legível descontando-se o
  diacrítico? Medido pelo CER de um reconhecedor, com alvo e predição dobrados
  para ASCII.

O cruzamento dos dois dá as quatro categorias:

|                     | base íntegra        | base degradada               |
|---------------------|---------------------|------------------------------|
| **acento presente** | acerto              | degradação com diacrítico    |
| **acento ausente**  | omissão do acento   | degradação sem diacrítico    |

## Arquivos

| arquivo | papel |
|---|---|
| `gerar_pares.py` | sonda: gera os pares mínimos com estilo/semente fixos |
| `metrica.py` | núcleo: E1 (duas variantes), CER, dobra ASCII |
| `reconhecedor.py` | E2: TrOCR + CER dobrado para ASCII |
| `preparar_reais.py` | controle: recortes REAIS do BRESSAY no mesmo formato |
| `avaliar.py` | aplica E1/E2 a um conjunto, classifica, IC por bootstrap |
| `calibrar.py` | Passo 4: valida nos dois casos de resposta conhecida |
| `anotacao.py` | Passo 6: planilha de anotação humana e Cohen's kappa |
| `testes_metrica.py` | testes sintéticos de resposta conhecida da geometria |
| `checar_alinhamento.py` | Passo 1: mede se os gêmeos saem alinhados |

## Passo 1 — o portão (resultado: PASSOU)

Antes de decidir como fazer o E1, foi preciso medir se os gêmeos saem
espacialmente alinhados. Se saem, o E1 pode ser feito por **diferença de
imagens**, que é mais robusto do que estimar linha de base e altura-x.

6 pares mínimos, mesmo escritor (174), mesma semente, `ema_ep11.pt`:

```
par                      dx   dy   IoU@0  corr_col
ação/acao                 0    0   0.879     0.992
àqueles/aqueles           0    0   0.967     1.000
doença/doenca             0    0   0.949     0.999
importância/importancia   0    0   0.649     0.853
não/nao                   0    0   0.915     0.999
país/pais                 0    0   0.957     0.999
MEDIANAS                  0    0   0.932     0.999
```

Deslocamento ótimo zero em todos os pares. **Os gêmeos saem alinhados**, então
o E1 por diferença é válido. As imagens estão em `passo1_contato.png` (ordem
por par: acentuada / ascii / |diff|).

As duas variantes do E1 continuam existindo porque servem a coisas diferentes:

- `e1_por_diff` — usa o par. Mais limpo, mas exige o gêmeo.
- `e1_por_faixa` — usa só a imagem acentuada. É a **única** que funciona em
  recortes reais do BRESSAY, onde não existe gêmeo ASCII do mesmo punho. Toda
  comparação entre gerado e real usa esta variante nos dois lados.

## Achado de infraestrutura: lote > 1 corrompe a amostragem nesta GPU

Descoberto ao montar a sonda, e vale para qualquer geração neste repositório.
Mesma entrada, mesmo escritor, mesmo ruído, RX 6600 XT / ROCm:

```
batch 4, rep 0 : [0.0793, 0.0515, 0.0423, 0.0643]   std da imagem
batch 4, rep 1 : [nan, nan, nan, nan]
batch 4, rep 2 : [0.1229, 0.0596, 0.0756, 0.0691]
batch 2        : [0.1863, 0.0597, 0.0398, 0.0637]
batch 1, rep 0 : [0.1858, 0.1945, 0.2165, 0.2193]
batch 1, rep 1 : [0.1809, 0.1945, 0.2165, 0.2193]
```

Com lote > 1 a saída colapsa para cinza quase uniforme (ou vira NaN) e não é
reproduzível. Com lote 1 é estável e **igual à CPU**: nas 12 imagens do portão,
max|dif| GPU-vs-CPU = 0.0039, que é 1/255, a quantização do PNG.

As operações básicas da GPU estão sãs (matmul, conv e reduções bit-exatos, sem
NaN em stress) — o problema aparece só no caminho completo de amostragem.

Consequência para a métrica: uma imagem colapsada seria lida como "acento
omitido", ou seja, o artefato viraria resultado. Por isso:

- `gerar_pares.py` usa `--batch 1` por padrão;
- `--autoteste-lote` verifica a invariante e **falha** se o lote divergir do
  individual (medido: passa bit-exato em batch 1, falha com mean|dif| = 0.43 em
  batch 4);
- o manifest marca `colapsada` para imagens com NaN ou std < 0.02, e o
  `avaliar.py` as descarta em vez de pontuá-las.

Varrendo as pastas `amostras_*` já existentes: **12 de 332 sub-painéis (3,6%)
estão degenerados**, todos em `ação.png`, nas pastas `amostras_bressay_ep15`,
`amostras_bressay_ep15_sem_flag` e `amostras_v2_5ep` — os 4 estilos com
std = 0,000, imagem em branco. Não é o modelo falhando em "ação": é a
amostragem em lote.

## Limitação do corpus: til e cedilha quase não têm gêmeo real

Dos 10.691 tipos do BRESSAY, 2.296 têm diacrítico e 188 têm o gêmeo ASCII
também como palavra real. Mas a distribuição por marca é muito desigual:

| marca | pares com gêmeo ASCII real |
|---|---|
| agudo | 148 |
| circunflexo | 23 |
| grave | 8 |
| cedilha | 5 |
| til | **3** (`sã`, `fã`, `dã`) |

Til e cedilha são justamente as marcas que o IAM puro apaga. Por isso
`pares_sonda30.tsv` é estratificado por marca (6 por marca) e traz uma coluna
`gemeo_e_palavra_real`: vale 1 para agudo/grave/circunflexo e 0 para
til/cedilha. Nesses últimos o gêmeo ASCII é uma não-palavra, e qualquer
diferença observada pode ser efeito de não-palavra em vez de efeito de
diacrítico. **Isso tem que ser declarado ao reportar.**

## Como rodar

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
PY=/nix/store/98rpw6g3y3j5vc2xiyhwdqgy7xl1qyix-python3-3.14.7-env/bin/python

# 0. testes do instrumento (não usam o gerador)
$PY avaliacao_diacriticos/testes_metrica.py

# 1. invariante de amostragem do checkpoint que for usar
$PY avaliacao_diacriticos/gerar_pares.py --save-path ./sanity \
    --ckpt ema_ckpt.pt --out-dir /tmp/x \
    --pares-tsv avaliacao_diacriticos/pares_gate.tsv \
    --n-styles 1 --seeds 0 --batch 1 --autoteste-lote

# 2. gerar a sonda (N declarado: 29 pares x 4 estilos x 3 sementes)
$PY avaliacao_diacriticos/gerar_pares.py --save-path <modelo> \
    --ckpt <ckpt.pt> --out-dir <saida> \
    --pares-tsv avaliacao_diacriticos/pares_sonda30.tsv \
    --n-styles 4 --seeds 0 1 2 --batch 1

# 3. avaliar
$PY avaliacao_diacriticos/avaliar.py --dir <saida> \
    --csv-out <saida>.csv --com-e2 --device cpu \
    --eixo1 faixa --limiar-e1 <da calibração> --limiar-e2 0.5

# 4. planilha de anotação humana e kappa
$PY avaliacao_diacriticos/anotacao.py preparar --csv <saida>.csv \
    --dir <saida> --out anotacao.csv --contato anotacao.png
$PY avaliacao_diacriticos/anotacao.py kappa --csv anotacao_preenchida.csv
```
