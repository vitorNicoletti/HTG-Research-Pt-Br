# Estado — o que está medido e o que falta

Atualizado ao fim da sessão que corrigiu a linha pautada. Tudo abaixo são
arquivos normais em `avaliacao_diacriticos/`, commitados na branch
`diagnostico-gpu-e-reorganizacao`. Nenhum processo ficou rodando.

Os números detalhados e o raciocínio estão no `README.md`; aqui fica só o
estado e o que fazer em seguida.

## Pronto e verificado

| Passo | Estado |
|---|---|
| 1 — portão de alinhamento | **PASSOU**. dx=dy=0 nos 6 pares, IoU mediano 0,932. |
| 2 — E1 | Duas variantes, com remoção da linha pautada. **31 testes sintéticos** de resposta conhecida passam. |
| 3 — E2 | Implementado e **medido: satura**. Ver abaixo. |
| 4 — controles | Fechado, com um controle a mais do que o planejado. |
| 5 — N declarado | Fechado **para o IAM puro**: 696 imagens, 0 colapsadas, 0 NaN. |
| 6 — kappa | Ferramenta pronta e validada; planilha e folha de contato geradas. Falta a anotação humana. |

## Os três resultados que importam

**1. A pauta estava sendo medida como acento, e corrigir derrubou o controle
positivo de 0,861 para 0,510.** Reportado como queda, sem mexer no limiar. O
discriminador certo é espessura (bandas de até 5 px), não cobertura: a correção
ingênua por cobertura destruía palavras curtas. 100% das amostras de um modelo
ajustado no BRESSAY (`am_v2/`) têm a pauta, então a correção vale para o gerado.

**2. O E1 por faixa é um detector fraco em material real (AUC 0,680).** Isso só
apareceu com um controle negativo **no mesmo domínio** (`controle_ascii.py`:
palavra real do BRESSAY sem diacrítico, com acento fabricado). Contra o IAM
puro o AUC parecia 0,849 — boa parte era diferença de domínio, não presença de
acento. A barra do Passo 4 ("perto de 100% nos dois lados") **não é atingida**
com limiar único.

**3. O E2 não está validado: o TrOCR não lê este corpus.** Em recorte humano
real, CER 0,826 nas acentuadas e 1,035 nas ASCII, com **1 leitura perfeita em
120**. A classificação em quatro categorias vira de ponta-cabeça conforme o
limiar (89,9% de "degradação" com corte 0,3; 83,8% de "acerto" com corte
relativo). Enquanto não houver reconhecedor treinado em manuscrito português, a
classificação honesta tem **duas** categorias, não quatro.

## Para quando o fine-tune do BRESSAY estiver pronto

O instrumento está calibrado e caracterizado. A sequência é:

```bash
cd ~/repos/htg-tcc
# O ambiente do projeto ja exporta HSA_OVERRIDE_GFX_VERSION e
# PYTORCH_HIP_ALLOC_CONF, e poe o `python` certo no PATH.
nix develop

# 1. invariante de amostragem do checkpoint novo (obrigatório: lote > 1
#    corrompe nesta GPU, e imagem colapsada vira "acento omitido")
python avaliacao_diacriticos/gerar_pares.py --ckpt <novo.pt> --out-dir /tmp/x \
    --pares-tsv avaliacao_diacriticos/pares_gate.tsv \
    --n-styles 1 --seeds 0 --batch 1 --autoteste-lote

# 2. gerar com o mesmo N do controle negativo (696 imagens, ~30 min na GPU)
python avaliacao_diacriticos/gerar_pares.py --ckpt <novo.pt> \
    --out-dir avaliacao_diacriticos/amostras/ger_ft --batch 1 \
    --pares-tsv avaliacao_diacriticos/pares_sonda30.tsv \
    --n-styles 4 --seeds 0 1 2

# 3. ler o eixo por DIFERENÇA (não o por faixa) contra limiares_eixo_diff.json
python avaliacao_diacriticos/avaliar.py --dir avaliacao_diacriticos/amostras/ger_ft \
    --csv-out avaliacao_diacriticos/resultados/res_ft.csv --eixo1 diff --limiar-e1 0.0455
```

O limiar 0,0455 é o p95 do til no controle negativo. **Use o limiar da marca**
(`limiares_eixo_diff.json`), não um global: para agudo (p95 0,264) e cedilha
(p95 0,206) o ruído supera um acento nominal (0,166), então nessas duas marcas
só a média agregada com IC é confiável — classificação amostra a amostra, não.
Para til (0,046) e grave (0,037) o eixo separa por amostra com folga de 3 a 4×.

A referência de escala vem de `teste_sensibilidade_diff.py`: 0,0000 = gêmeos
idênticos, 0,0389 = meio acento nominal, 0,1657 = acento nominal. O IAM puro,
em 348 diacríticos, dá 0,0116 — cerca de 7% de um acento.

## O que falta, em ordem

1. **Anotação humana** (Passo 6). `anotacao_reais.csv` tem 38 linhas
   estratificadas pelas quatro categorias e `figuras/anotacao_folha.png` é a folha de
   contato. Quem anota preenche só `acento_presente` e `base_legivel` (0/1) e
   **não deve ver as colunas `_metrica_*`**. Depois:
   `anotacao.py kappa --csv <preenchida>`. O kappa do eixo E2 vai quantificar o
   quanto o TrOCR discorda de um leitor humano — que é a evidência mais direta
   do problema 3.
2. **Trocar o reconhecedor do E2**, ou declarar o eixo como não medido. Não
   afrouxar o limiar.
3. **Aumentar o n de cedilha e grave** nos controles se eles forem virar
   conclusão: hoje são n=8 e n=7 no positivo real.

## Avisos que continuam valendo

- **Lote > 1 corrompe a amostragem nesta GPU** (NaN, colapso para cinza, não
  reproduzível). `--batch 1` é obrigatório e o `--autoteste-lote` verifica.
  Na corrida de 696 imagens deu 0 colapsadas e 0 NaN.
- **Til e cedilha quase não têm gêmeo ASCII real no corpus** (3 e 5 pares dos
  188). Em `pares_sonda30.tsv` a coluna `gemeo_e_palavra_real` diz quais pares
  têm o gêmeo como palavra real; para til e cedilha ele é não-palavra, e o
  confundimento precisa ser declarado.
- **O escore E1 depende da `folga`** (varia 45% entre folga 0 e 2), embora a
  separação não dependa (AUC 0,79–0,88). Sempre declarar a folga usada e
  calibrar o limiar na mesma folga.
