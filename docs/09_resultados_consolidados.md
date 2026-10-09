# 09 — Resultados consolidados

Todas as tabelas num lugar só. A explicação de cada experimento está em
[07](07_experimentos_com_acentos.md) e a de cada medida em
[08](08_protocolo_de_avaliacao.md).

## Tabela principal: validação, protocolo `avaliar_pt.py`

30 acentuadas + 30 esqueletos + 20 sem acento, 20 escritores do `iam_test`
× 2 sementes, 3.200 imagens por modelo.

| modelo | fase | marca: acentuada | marca: esqueleto | marca: sem acento | **diferença pareada** | CER sem acento | CER sem acento pedido e sem marca | agudo na letra certa | til na letra certa | tinta borrada |
|---|---|---|---|---|---|---|---|---|---|---|
| IAM original (sem fine-tune) | — | 6% | 5% | 5% | +1% | 0,22 | 0,19 | 1% | 0% | 0,59 |
| só originais do IAM, 4 ép. | 7c | 4% | 4% | 4% | +1% | 0,21 | 0,18 | 0% | 0% | 0,60 |
| pt, 4 ép. | 5 | 38% | 27% | 26% | +11% | 0,34 | — | — | — | — |
| pt, 8 ép. | 5 | 38% | 27% | 24% | +12% | 0,32 | — | — | — | — |
| pt, 12 ép. | 5 | 41% | 28% | 24% | +14% | 0,33 | — | — | — | — |
| pt, 16 ép. | 5 | 42% | 26% | 26% | +16% | 0,34 | — | 7% | 26% | — |
| controle, pares alinhados, 16 ép. | 6b | 46% | 30% | 25% | +16% | 0,32 | 0,28 | 9% | 30% | 0,68 |
| **peso 5, 16 ép.** | 6b | 57% | 34% | 32% | **+23%** | 0,32 | 0,27 | 14% | **52%** | 0,68 |
| peso 5 + 100% IAM, 16 ép. | 7b | 62% | 37% | 30% | +25% | 0,35 | — | 15% | 53% | — |
| IAM real + peso 5 + zona vazia 2, 12 ép. | 8 | 74% | 54% | 56% | +20% | 0,46 | 0,35 | 19% | 33% | 0,59 |
| IAM real + peso 5 + zona vogais 5, 12 ép. | 9 | 69% | 48% | 48% | +21% | 0,47 | 0,36 | 15% | 28% | 0,59 |

Pastas de cada avaliação, no WSL:

| modelo | pasta |
|---|---|
| IAM original e pt | `avaliacao_pt_val` |
| controle e peso 5 | `avaliacao_peso_val` |
| peso 5 + 100% IAM | `avaliacao_orig100_val` |
| só originais | `avaliacao_so_originais_val` |
| zona vazia | `avaliacao_pares_val` |
| zona vogais | `avaliacao_vogais_val` |

Notas:
- **"CER sem acento"** usa só as 20 palavras sem acento, como no `resumo.md`.
  **"sem acento pedido e sem marca"** junta esqueletos e palavras sem acento,
  só nas imagens sem marca (`comparar_avaliacoes.py`).
- **Posição:** palavras com um único sinal ou todas, conforme o relatório em
  `diagnostico/resultados/posicao_acento*/`.
- **Cedilha:** não medida. O detector não acha cedilha encostada no c.

## Comparações estatísticas (bootstrap por palavra, IC95)

| comparação | medida | diferença | IC95 | leitura |
|---|---|---|---|---|
| peso 5 − controle | diferença pareada | **+7,1 pp** | +3,1 a +11,1 | o peso aumenta o acento condicionado ao texto |
| peso 5 − controle | marca em palavra sem acento | +6,7 pp | +3,0 a +10,5 | e aumenta o vazamento |
| peso 5 − controle | CER sem marca | −0,007 | −0,020 a +0,006 | sem efeito na letra |
| controle − pt | diferença pareada | −0,5 pp | −4,5 a +3,3 | alinhar os pares sozinho não muda nada |
| 100% IAM − 30% IAM (peso 5) | CER sem acento | **+0,027** | +0,010 a +0,045 | mais escrita real não recupera a letra |
| só originais − IAM original | CER sem acento | −0,009 | −0,023 a +0,007 | o fine-tune em si não estraga a letra |
| peso 5 + 100% IAM − só originais | CER sem acento | **+0,137** | +0,107 a +0,167 | a base sintética estraga a letra |
| zona vogais − zona vazia | marca falsa | **−6,7 pp** | −9,8 a −3,6 | as vogais reduzem o vazamento |
| zona vogais − zona vazia | CER sem marca | +0,016 | +0,001 a +0,032 | e a letra não melhora |

## Medição de marcas da fase 4 (80 painéis por palavra)

40 escritores do IAM × 2 sementes; palavras da sonda e de controle em inglês.

| palavra | IAM | sem teto | teto 25% |
|---|---|---|---|
| nação | 2% | 85% | 64% |
| coração | 6% | 86% | 81% |
| mãe | 6% | 71% | 40% |
| café | 8% | 62% | 41% |
| pão | 0% | 56% | 35% |
| você | 9% | 55% | 44% |
| avó | 1% | 19% | 20% |
| nacao (sem acento) | 4% | 52% | 49% |
| coracao (sem acento) | 6% | 42% | 65% |
| the (sem acento) | 2% | 30% | 19% |
| and (sem acento) | 0% | 28% | 24% |

Os modelos da fase 4 **não passaram pelo protocolo `avaliar_pt.py`**. O
`model_iam_acentuado_teto25` é o próximo candidato a avaliar (ver
[10](10_problemas_em_aberto.md)).

## Legibilidade das bases (`diagnostico/cer_base.py`)

| o que foi lido | CER | leitura exata |
|---|---|---|
| escrita real do `iam_test` | 0,12 | 62% |
| base pt: sem acento | 0,19 | 30% |
| base pt: pares | 0,19 | 26% |
| base pt: acentuadas (contra o esqueleto) | 0,27 | 11% |

## Treinos no BRESSAY

Sem avaliação pelo protocolo. Inspeção visual e deriva em
[05](05_finetune_bressay.md):
- **`bressay_25`, 26 épocas:** pior que o original em tudo;
- **`bressay_25_v2`, 40 épocas:** formato corrigido, nenhum acento.

## Figuras principais

Imagens versionadas, que abrem direto do repositório:

| figura | o que mostra |
|---|---|
| `diagnostico/resultados/peso_acento/comparacao/28_prestacao.png` | IAM × pt × controle × peso 5: o peso 5 escreve "ção" com til e cedilha |
| `diagnostico/resultados/peso_acento/comparacao/07_havera.png` | o agudo na letra errada ("hávera") e o vazamento para "havera" |
| `diagnostico/resultados/peso_acento/comparacao_orig100/` | peso 5 com 30% × 100% de escrita real |
| `diagnostico/resultados/peso_acento/comparacao_pares/` | IAM real + zona vazia: hastes somem; folha de amostras de 4 e 12 épocas |
| `diagnostico/resultados/peso_acento/comparacao_vogais/` | zona vazia × zona vogais |
| `diagnostico/resultados/peso_acento/mascaras.png` | máscara do acento (base pt) |
| `diagnostico/resultados/peso_zona_iam/zona.png` | zona vazia sobre a base IAM |
| `diagnostico/resultados/zona_vogais/zona_vogais.png` | zona vogais: faixa só sobre as vogais |
| `saidas/diffusionpen/fine_tune_iam_acentuado/` | fase 4: comparações e evolução por época |
| `saidas/diffusionpen/fine_tune_25*/` | BRESSAY v1 e v2 |
| `saidas/diffusionpen/vae_resolucao.png` | o VAE preserva o acento em 64×256 e perde em 32×128 |

As 30 figuras comparativas completas de cada avaliação e todas as 3.200
imagens por modelo ficam no WSL, em `saidas/comparacao_*/` e
`avaliacao_*/paineis/`.
