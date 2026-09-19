# Estado — onde parei e como retomar

Sessão interrompida a pedido (desligar a máquina). **Nada foi perdido**: tudo
abaixo são arquivos normais em `avaliacao_diacriticos/`, não há estado em
memória. Nenhum processo ficou rodando.

## Pronto e verificado

| Passo | Estado |
|---|---|
| 1 — portão de alinhamento | **PASSOU**. dx=dy=0 nos 6 pares, IoU mediano 0.932, corr de colunas 0.999. Números em `passo1_alinhamento.json`, imagens em `passo1_contato.png` e `gate_cpu/`. |
| 2 — E1 | Implementado nas duas variantes (`e1_por_diff`, `e1_por_faixa`). 14 testes sintéticos de resposta conhecida passam (`testes_metrica.py`), incluindo "ascendente não conta como acento" e o caso do pingo do "i". |
| 3 — E2 | TrOCR funcionando, com CER dobrado para ASCII dos dois lados. Precisou reconstruir o `tokenizer.json` (o repo do Hub só tem vocab+merges e o transformers 5.x exige tokenizer fast); está versionado em `trocr_tokenizer/`, com round-trip conferido. |
| 6 — kappa | Implementado e validado em dados sintéticos (1.0 / −1.0 / ~0). Falta rodar sobre dados reais, que dependem do Passo 4/5. |

## Passo 4 — controles: metade medido

O eixo E1 já separa os dois extremos, **usando a mesma variante (faixa) nos
dois lados**, que é a única comparação honesta:

| conjunto | n | E1 escore médio | IC95 |
|---|---|---|---|
| IAM puro (`ger_iam_puro/`, 29 pares × 2 estilos × 1 semente) | 58 | **0.141** | [0.090, 0.209] |
| recortes REAIS do BRESSAY (`reais_test/`, 120 acentuados) | 148 | **0.861** | [0.713, 1.035] |

Pela variante de diferença (só disponível no gerado), o IAM puro dá
**0.003** [−0.013, 0.022] — ou seja, praticamente nenhuma tinta a mais onde o
diacrítico deveria estar. É o controle negativo se comportando como esperado.

**O que falta:** o E2 nos recortes reais não terminou (TrOCR em CPU, ~240
imagens; foi interrompido). Sem ele não dá para calcular o piso de leitura nem
fechar a calibração do limiar. Retomar com:

```bash
PY=/nix/store/98rpw6g3y3j5vc2xiyhwdqgy7xl1qyix-python3-3.14.7-env/bin/python
cd ~/repos/htg-tcc
$PY avaliacao_diacriticos/avaliar.py --dir avaliacao_diacriticos/reais_test \
    --csv-out avaliacao_diacriticos/res_reais.csv --com-e2 --device cpu
$PY avaliacao_diacriticos/calibrar.py \
    --negativo avaliacao_diacriticos/res_iam_faixa.csv \
    --positivo avaliacao_diacriticos/res_reais.csv \
    --json-out avaliacao_diacriticos/calibracao_e1.json
```

Aviso que já dá para dar: numa amostra de 10 recortes **reais**, o TrOCR leu
"econômico"→"economic", "perpetuação"→"perpetuation", "não"→'" Chicago"', com
CER entre 0.11 e 1.67. É provável que o E2 sature perto do teto já no
manuscrito humano. Se isso se confirmar, o achado é que **E2 não discrimina
integridade de base com este reconhecedor** — o que é uma medida válida e
precisa ser reportada, não um defeito a esconder.

## Passo 5 — não feito, por decisão sua

Você pediu para não gerar amostras com os modelos do BRESSAY por ora, já que
ainda não estão bons. O conjunto parcial que tinha começado foi apagado. O
comando está pronto no README quando quiser; com `--batch 1` na GPU dá ~2,5 s
por imagem, ou seja ~30 min para 29 pares × 4 estilos × 3 sementes.

## Dois achados que valem independentemente do fine-tune

1. **Lote > 1 corrompe a amostragem nesta GPU** (NaN, colapso para cinza, não
   reproduzível). Com lote 1 a GPU bate com a CPU em 1/255. Consequência
   concreta: 12 de 332 sub-painéis das pastas `amostras_*` estão em branco
   (std = 0.000), todos em `ação.png`, em `amostras_bressay_ep15`,
   `amostras_bressay_ep15_sem_flag` e `amostras_v2_5ep`. Isso é artefato de
   amostragem, não incapacidade do modelo. Detalhes e medições no README.
   O `gerar_amostras.py` da raiz gera 4 estilos num lote só — **não mexi
   nele**, é da outra sessão.

2. **Til e cedilha quase não têm gêmeo ASCII real no corpus**: dos 188 pares,
   só 5 são de cedilha e 3 de til (`sã`, `fã`, `dã`). Como são justamente as
   marcas que o IAM apaga, `pares_sonda30.tsv` marca numa coluna quais pares
   têm gêmeo real e quais não — o confundimento com "não-palavra" precisa ser
   declarado no TCC.

## Não commitado

Não commitei nada: o checkout está na `main` e tem alterações não commitadas de
outra sessão (`flake.nix`, `requirements.txt`). Se quiser que eu commite só a
pasta `avaliacao_diacriticos/` numa branch separada, é só pedir.
