# Métrica de diacríticos

Mede se um gerador de manuscrito desenha os diacríticos do português. Tem dois
eixos. O eixo 1 mede a presença do acento na imagem e está pronto. O eixo 2
mede se a palavra continua legível e depende de um reconhecedor de texto que
ainda não temos.

## Como o eixo 1 funciona

1. Gera a mesma palavra duas vezes, com e sem acento ("ação" e "acao"), com o
   mesmo escritor e a mesma semente. A única diferença entre as duas imagens é
   o texto pedido.
2. Binariza as duas e remove a linha pautada do papel, quando houver.
3. Alinha uma imagem sobre a outra por correlação cruzada.
4. Conta a tinta que a versão acentuada tem a mais na região do acento: acima
   da altura-x (abaixo da linha de base, para a cedilha), na coluna do
   caractere acentuado. Divide pela tinta do corpo da letra na mesma coluna.

O escore é uma fração do tamanho da letra. Zero significa que nada foi
desenhado. Um acento de tamanho normal vale entre 0,13 e 0,18, conforme a
marca.

## Arquivos

| arquivo | papel |
|---|---|
| `e1.py` | o eixo 1 |
| `montar_pares.py` | escolhe as palavras da sonda (`pares_sonda.tsv`) |
| `gerar_pares.py` | gera os pares a partir de um checkpoint |
| `avaliar.py` | aplica a métrica a uma pasta de amostras e resume |
| `metrica.py` | funções de apoio e a variante para imagem sem par |
| `testes_metrica.py` | testes com figuras sintéticas de resposta conhecida |
| `experimentos/` | scripts de validação e comparação, e os do eixo 2 |
| `figuras/` | figuras explicativas, geradas por `experimentos/figuras.py` |
| `resultados/` | medições em CSV e JSON |
| `amostras/` | conjuntos de imagem, fora do git |

## Como usar

Todos os comandos rodam da raiz do repositório, dentro do ambiente:

```bash
cd ~/repos/htg-tcc
nix develop
```

Testes:

```bash
python avaliacao_diacriticos/testes_metrica.py
```

Medir um checkpoint:

```bash
# gera os pares: 77 palavras, 4 escritores, 2 sementes
python avaliacao_diacriticos/gerar_pares.py --ckpt <modelo.pt> \
    --out-dir avaliacao_diacriticos/amostras/<nome> \
    --pares-tsv avaliacao_diacriticos/pares_sonda.tsv \
    --n-styles 4 --seeds 0 1

# aplica a métrica
python avaliacao_diacriticos/avaliar.py \
    --dir avaliacao_diacriticos/amostras/<nome> \
    --csv-out avaliacao_diacriticos/resultados/<nome>.csv --eixo1 diff
```

Depois de gerar, confira no `manifest.jsonl` da pasta se alguma imagem veio
com `"colapsada": true`. Se veio, gere de novo com `--device cpu`.

Refazer as figuras:

```bash
python avaliacao_diacriticos/experimentos/figuras.py
```

## Como ler o resultado

O `avaliar.py` imprime, por marca, o escore médio e um intervalo de confiança
de 95%. O intervalo é calculado reamostrando palavras, porque várias imagens
da mesma palavra não são medidas independentes.

Para decidir se um escore conta como acento, compare com
`resultados/limiares_eixo_diff.json`. Os campos por marca:

- `p95`: o limiar. Um gerador que não desenha acento fica acima dele em menos
  de 5% das imagens.
- `acento_nominal`: o escore de um acento de tamanho normal.
- `decide_por_amostra`: se é falso, o ruído supera um acento normal e só a
  média de muitas imagens é confiável naquela marca.
- `palavras`: quantas palavras distintas sustentam os números.

## Resultado no modelo do IAM

O DiffusionPen treinado só no IAM não desenha acentos e serve de controle
negativo. O controle positivo é um acento de tamanho normal pintado sobre a
imagem sem acento. Medido em `amostras/ger_iam_sonda77x4/`, 1.232 imagens.

| marca | palavras | limiar (p95) | acento normal | AUC | decide por imagem |
|---|---|---|---|---|---|
| grave | 5 | 0,043 | 0,151 | 1,000 | sim |
| til | 18 | 0,090 | 0,148 | 0,968 | sim |
| circunflexo | 18 | 0,047 | 0,132 | 0,989 | sim |
| agudo | 18 | 0,177 | 0,167 | 0,937 | não |
| cedilha | 18 | 0,206 | 0,179 | 0,931 | não |

AUC é a probabilidade de uma imagem com acento pontuar mais que uma sem
acento. 0,5 é acaso e 1,0 é separação perfeita.

Na cedilha o modelo do IAM desenha uma cauda abaixo da letra em parte das
amostras (ver `figuras/e1_falhas.png`). O escore na coluna do "ç" é maior que
nas outras colunas da mesma imagem em 0,026, com intervalo de 0,009 a 0,044.
Por isso o limiar da cedilha medido neste controle está inflado.

## Comparação com SSIM e PSNR

SSIM e PSNR comparam uma imagem com uma referência pixel a pixel. Em geração
não existe referência, exceto dentro do par: a imagem sem acento serve de
referência para a acentuada. Medido em `experimentos/ssim_psnr.py`, com três
controles negativos: o que o IAM gerou, duas saídas sem acento com sementes
diferentes, e um acento pintado na coluna errada.

| escore | contra o gerador | contra a variação entre sementes | contra acento na coluna errada |
|---|---|---|---|
| E1 | 0,95 | 0,92 | 0,97 |
| SSIM na região do acento | 0,69 | 0,56 | 0,97 |
| PSNR na região do acento | 0,98 | 0,97 | 0,93 |
| SSIM na imagem inteira | 0,00 | 0,00 | 0,52 |
| PSNR na imagem inteira | 0,21 | 0,01 | 0,52 |

Na imagem inteira as duas ficam abaixo do acaso, porque a variação do gerador
entre duas imagens é muito maior que um diacrítico. Restrito à região do
acento, o PSNR é um pouco melhor que o E1 em dois controles e pior no
terceiro, que verifica se o acento está no lugar certo. O E1 foi mantido. A
figura é `figuras/e1_vs_ssim_psnr.png`.

## Eixo 2

O eixo 2 precisa de um reconhecedor que leia manuscrito em português. Três
modelos prontos foram testados em recortes reais do BRESSAY
(`experimentos/htr_comparar.py`):

| | TrOCR | EasyOCR | PaddleOCR |
|---|---|---|---|
| erro por caractere (CER) | 1,13 | 0,75 | 0,43 |
| palavras acentuadas em que emite algum acento | 0% | 27,5% | 20,8% |
| acento inventado em palavra sem acento | 0% | 0,8% | 0% |

O PaddleOCR é o melhor. Em 1.000 palavras acentuadas ele acerta as letras em
15% e, dessas, acerta também o acento em 28%. Não é suficiente para sustentar
o eixo. O PaddleOCR roda no shell `nix develop .#htr`, com Python 3.13.

## Limitações

- O controle positivo é um acento pintado. Ainda não há um gerador que
  desenhe acentos para servir de positivo real.
- Os limiares valem para o pré-processamento `v1` do BRESSAY. Se o `v2` virar
  o padrão de `load_image`, eles precisam ser medidos de novo.
- Til e cedilha quase não têm par em que a forma sem acento também seja uma
  palavra do corpus. A coluna `gemeo_e_palavra_real` de `pares_sonda.tsv`
  marca os casos.
- Grave só tem 5 palavras distintas no corpus com quatro letras ou mais.
- Em imagem sem par, como um recorte real, só dá para usar a variante de
  `metrica.py` que olha uma imagem. Ela separa mal palavra com acento de
  palavra sem acento (AUC 0,68).

## Figuras

| arquivo | conteúdo |
|---|---|
| `e1_passo_a_passo.png` | as etapas do eixo 1 em um par |
| `e1_exemplos.png` | um par de cada marca |
| `e1_falhas.png` | os maiores escores do controle negativo |
| `e1_vs_ssim_psnr.png` | comparação com SSIM e PSNR |
| `anotacao_folha.png` | folha para anotação humana |
