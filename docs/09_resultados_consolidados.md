# 09 — Resultados consolidados

Todas as tabelas num lugar só. A explicação de cada experimento está em
[07](07_experimentos_com_acentos.md), [14](14_acento_separado_e_professor.md)
e [15](15_acento_a_mais_e_variantes_do_professor.md), e a de cada medida em
[08](08_protocolo_de_avaliacao.md) e em
[avaliacao_diacriticos/README.md](../avaliacao_diacriticos/README.md).

A primeira seção é o estado atual: o modelo final e a métrica final. As seções
seguintes, a partir de "Fases 4 a 9", são das fases anteriores ao acento
separado e ficam como histórico.

## Modelo final

Zona vogais com 40 épocas (`modelos/zona_vogais_40ep.pt`). Parte dos pesos do
IAM e treina só o UNet, com três ingredientes:
- acento separado: o CANINE lê a palavra sem acento e o sinal entra por um
  vetor próprio, somado na posição da letra ([14](14_acento_separado_e_professor.md));
- professor: nas amostras sem diacrítico, o alvo é a previsão de uma cópia
  congelada do modelo original ([14](14_acento_separado_e_professor.md));
- peso 5 no acento e na zona das vogais, `--peso_zona 5 --zona vogais`
  ([15](15_acento_a_mais_e_variantes_do_professor.md)).

Dados: a base `iam_pt_alinhado` (palavras portuguesas geradas pelo DiffusionPen
e acentuadas, cada uma com o par sem acento) mais 30% das palavras originais do
`iam_train_val`. lr 2e-5, lote 32, extrator de estilo do IAM congelado.

### Onde foi treinado

| épocas | máquina | como |
|---|---|---|
| 1 a 16 | Google Colab, NVIDIA Tesla T4, torch 2.11.0+cu130 | três sessões, cada uma retomando da anterior |
| 17 a 40 | RX 9060 XT, torch 2.11.0+rocm7.13.0 | continuação do estado salvo no Colab, em dois blocos (até 30 e até 40) |

- Os parâmetros são os mesmos em todos os blocos (registrados em
  `diagnostico/resultados/acento_separado/config_zona_vogais*.jsonl`).
- Entre o commit do Colab (`cc290e7`) e o da RX 9060 XT (`37e0e5c`), o
  `diffusionpen_mods/train.py` só ganhou a opção `--professor_fora_t_max`,
  desligada neste treino. O caminho de treino é o mesmo.
- A T4 passou nos testes de `diagnostico/` antes de treinar, e a primeira
  época repetiu os números do mesmo treino na máquina local.
- A retomada carrega pesos, otimizador e EMA (`models/estado.pt`,
  [02](02_diffusionpen_e_modificacoes.md)). Para o modelo é um treino só de 40
  épocas, pausado e continuado.
- Retreinar tudo numa máquina só não muda a conclusão e trocaria o modelo
  anotado pelas pessoas. A validação humana da métrica (212 imagens) é deste
  modelo de 40 épocas.

Os demais treinos com acento separado rodaram na RX 9060 XT, menos o "fora em
ruído baixo", que rodou no Colab.

### Protocolo `avaliar_pt.py`, modelos com acento separado

3.200 imagens por modelo, como na tabela da fase 9 abaixo. A marca conta
qualquer mancha solta em cima ou embaixo, e por isso inclui a cedilha solta
nas palavras com ç. O CER é do leitor `modelos/leitor_iam_transformer.pt`.

| modelo | marca: acentuada | marca: esqueleto | marca: sem acento | diferença pareada | CER acentuada | CER esqueleto | CER sem acento |
|---|---|---|---|---|---|---|---|
| IAM original | 6% | 5% | 5% | +1 | | | 0,22 |
| acento separado, 16 ép. | 61% | 11% | 11% | +51 | 0,39 | 0,28 | 0,29 |
| só professor, sem acento separado, 11 ép. | 64% | 20% | 21% | +44 | 0,48 | 0,22 | 0,26 |
| acento separado e professor, 16 ép. | 71% | 5% | 8% | +66 | 0,43 | 0,17 | 0,22 |
| pares sem professor, 16 ép. | 63% | 9% | 6% | +53 | 0,37 | 0,25 | 0,24 |
| professor fora da máscara, 16 ép. | 55% | 6% | 6% | +50 | 0,28 | 0,17 | 0,21 |
| zona vogais e fora em ruído baixo, 16 ép. | 58% | 5% | 5% | +53 | 0,31 | 0,17 | 0,21 |
| zona vogais, 16 ép. | 62% | 6% | 6% | +56 | 0,41 | 0,17 | 0,23 |
| zona vogais, 30 ép. | 64% | 5% | 5% | +59 | 0,39 | 0,17 | 0,21 |
| **zona vogais, 40 ép.** | 63% | 4% | 6% | +59 | 0,39 | 0,17 | 0,21 |

Resumos em `diagnostico/resultados/acento_separado/avaliacao_*_val/resumo.md`.
Para o acento a mais, use só a marca em cima
(`diagnostico/marcas_so_cedilha.py`, [15](15_acento_a_mais_e_variantes_do_professor.md)).

### Métrica final (E1 e E2)

A métrica final está em `avaliacao_diacriticos/` e foi validada contra a
anotação de duas pessoas no modelo de 40 épocas. Ela é diferente da métrica
E1/E2 da fase 2 ([04](04_sonda_e_metrica_inicial.md)), que foi abandonada.

- E1 diz se cada sinal pedido está na imagem e em que letra. Por palavra
  acentuada: correto, a mais, letra errada ou ausente.
- E2 diz se a palavra acentuada ficou mais difícil de ler que o seu esqueleto,
  com o mesmo escritor e a mesma semente, as duas lidas com os sinais apagados.
  Degradada quando a diferença de CER passa de 0,2.

| modelo | correto | a mais | letra errada | ausente | marca indevida, sem acento pedido | palavra degradada (E2) |
|---|---|---|---|---|---|---|
| só professor, 11 ép. | 14% | 23% | 16% | 46% | 20% | 50% |
| professor, 16 ép. | 44% | 26% | 16% | 13% | 5% | 46% |
| pares sem professor, 16 ép. | 47% | 18% | 15% | 20% | 10% | 19% |
| fora da máscara, 16 ép. | 39% | 11% | 14% | 36% | 5% | 15% |
| fora em ruído baixo, 16 ép. | 43% | 8% | 19% | 31% | 4% | 25% |
| zona vogais, 16 ép. | 55% | 14% | 14% | 17% | 5% | 46% |
| zona vogais, 30 ép. | 68% | 9% | 11% | 12% | 4% | 40% |
| **zona vogais, 40 ép.** | **72%** | 8% | 8% | 12% | 4% | 41% |

Tabelas completas e por classe de sinal em
`avaliacao_diacriticos/resultados/sinais/` e
`avaliacao_diacriticos/resultados/integridade/`.

Validação contra as pessoas, modelo de 40 épocas:

| medida | concordância |
|---|---|
| marca em cima | kappa 0,82 |
| cedilha (tinta abaixo do c) | kappa 0,91 |
| sinal presente | precisão 0,97, recall 0,90 |
| categoria da palavra | kappa 0,33; correto 87% pelas pessoas e 75% pela métrica |
| E2 | sem validação humana direta |

A métrica é mais severa que as pessoas, sobretudo no til, que encosta na letra
ou sai partido. O tipo do sinal (agudo, til...) não é julgado.

Figuras prontas para o artigo em
[resultados/figuras/artigo/](../resultados/figuras/artigo/README.md).

## Fases 4 a 9: validação, protocolo `avaliar_pt.py`

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
