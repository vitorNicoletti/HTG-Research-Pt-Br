# 07 — Fases 4 a 9: os experimentos com acentos sintéticos

Este documento conta, em ordem, cada experimento feito depois do BRESSAY. Cada
fase tem quatro partes:
- **por quê:** o que o resultado anterior sugeria;
- **o que mudou:** dados, loss e parâmetros;
- **resultado:** números da validação e figuras;
- **conclusão:** o que aprendemos e para onde isso levou.

Todos os fine-tunes partem dos pesos do IAM e usam lr 2e-5, lote 32 e o
extrator de estilo do IAM. O resto está em
[02](02_diffusionpen_e_modificacoes.md). As métricas (marca, diferença
pareada, CER, posição) estão explicadas em
[08_protocolo_de_avaliacao.md](08_protocolo_de_avaliacao.md). Todas as tabelas
juntas estão em [09_resultados_consolidados.md](09_resultados_consolidados.md).

---

## Fase 4 — Acentos sintéticos sobre palavras do IAM

### 4a. `model_iam_acentuado` (base sem teto)

**Por quê.** O BRESSAY não tem resolução para ensinar o acento. O IAM tem boa
resolução, mas está em inglês e não tem acento. A ideia foi **desenhar** o
acento em palavras reais do IAM e trocar a letra no rótulo: a imagem de "can"
com um til sobre o "a" vira uma amostra rotulada "cãn". O gerador está em
[06](06_acentos_sinteticos.md).

**O que mudou.**
- **Base `iam_acentuado`** (`scripts/gerar_base_acentos.py`): 29.732
  palavras do IAM (todas as elegíveis), cada uma com **um** sinal sorteado
  numa letra sorteada.
- **Dados do treino:** a base + as 55.535 palavras originais, sem acento.
- **Treino:** 10 épocas (`experimentos/iam_acentuado.json`).

**Resultado.**
- **Pela primeira vez, o acento aparece por causa do texto.** "nação",
  "coração" e "pão" saem com traços de til e cedilha nos lugares aproximados.
  O modelo original desenha "nação" igual a "nacao".
- **Vazamento:** "the" (sem acento) sai como "thé" já com 2 épocas. O treino
  viu 2.907 "the" sem acento e 2.858 com acento ("thé" e "thê" eram 9,6% da
  base).
- **"ó" e "ê" não apareceram;** "nação" ficou com letras deformadas.
- Figuras: `saidas/diffusionpen/fine_tune_iam_acentuado/`.

### 4b. `model_iam_acentuado_teto25` (teto de 25% por palavra)

**Por quê.** Frear o vazamento em palavras frequentes como "the".

**O que mudou.** No máximo 25% das ocorrências de cada palavra recebem
acento: 9.892 acentuadas, com "the" em 718 acentuadas contra 2.907 sem.
10 épocas.

**Resultado:** medição com 80 painéis por palavra (40 escritores × 2
sementes), `scripts/medir_marcas.py`:

| palavra | IAM original | sem teto | teto 25% |
|---|---|---|---|
| nação | 2% | 85% | 64% |
| café | 8% | 62% | 41% |
| avó | 1% | 19% | 20% |
| **nacao (sem acento)** | 4% | **52%** | **49%** |
| **the (sem acento)** | 2% | **30%** | **19%** |
| **coracao (sem acento)** | 6% | **42%** | **65%** |

**Conclusão da fase 4.**
- O dado sintético **ensina** a relação "diacrítico no texto → marca na
  imagem", algo que o BRESSAY não conseguiu.
- **O condicionamento é fraco:** "nação" 85% contra "nacao" 52%. O modelo
  aprendeu "às vezes ponha uma marca", mais do que "ponha a marca quando o
  texto pede".
- O teto não resolve.
- **Hipótese:** acentuar palavras **inglesas** ("thé", "ãnd") cria pares com
  palavras reais frequentes e não mostra nenhuma sequência do português.

---

## Fase 5 — Base de palavras portuguesas (`model_iam_pt`)

**Por quê.** Ensinar acentos em palavras portuguesas de verdade, cada uma com
**todos** os seus acentos no lugar certo ("coração" com til e cedilha).

**O que mudou.**
1. **Vocabulário particionado** (`vocabulario_pt/`, ver
   [01](01_problema_e_contexto.md)): treino, validação e teste por grupo de
   palavra, congelado antes de gerar qualquer imagem.
2. **Base `iam_pt`** (`scripts/gerar_base_pt.py`):
   - o **DiffusionPen original** escreve o **esqueleto** ("coracao") no estilo
     de um escritor do treino do IAM;
   - a imagem é recortada na tinta e ampliada 2× para o desenho;
   - o alinhador CTC lê o esqueleto: se a confiança for baixa, a geração
     ilegível sai;
   - `gerador.acentuar_palavra` desenha **todos** os sinais. Se algum sair
     invisível, a amostra sai.
3. **Três tipos de amostra:**
   - **acentuada** (5.188), rótulo "coração";
   - **par** (5.188): a mesma imagem antes dos sinais, rótulo "coracao";
   - **sem acento** (6.516): palavra portuguesa sem acento, como "casa".

   São 16.892 amostras ao todo, só com palavras de treino do vocabulário e
   só com escritores do treino do IAM.
4. **Treino:** a base + 30% das originais do IAM, 16 épocas
   (`experimentos/iam_pt.json`).
5. **Avaliação nova** (`scripts/avaliar_pt.py`, ver [08](08_protocolo_de_avaliacao.md)):
   - palavras **de validação** e escritores do `iam_test`;
   - um **leitor independente** (Transformer CTC, CER 0,084 no IAM val),
     diferente do alinhador que filtrou a base, para não haver
     circularidade.

**Resultado (validação).**

| modelo | marca: acentuada | marca: esqueleto | marca: sem acento | diferença pareada | CER sem acento |
|---|---|---|---|---|---|
| IAM original | 6% | 5% | 5% | +1% | 0,22 |
| pt, 4 épocas | 38% | 27% | 26% | +11% | 0,34 |
| pt, 16 épocas | 42% | 26% | 26% | +16% | 0,34 |

**Conclusão.**
- **Agora há condicionamento ao texto:** com o mesmo escritor e a mesma
  semente, a versão acentuada tem marca 16 pontos mais vezes que o esqueleto.
- Ainda há **vazamento** (26% contra 5%).
- **A letra piorou:** CER de 0,22 para 0,34, já com 4 épocas.

---

## Fase 6 — O acento pesa pouco na loss? Diagnóstico e peso no acento

### 6a. Diagnóstico (`diagnostico/diag_peso_acento.py`)

**Por quê.** Hipótese levantada pelo Vitor: o erro de errar um acento quase
não pesa na loss, porque o acento é uma fração mínima da imagem.

**O que foi medido**, em 96 pares da base, reproduzindo exatamente a conta da
loss do treino:
- **O sinal chega:** o CANINE distingue "coração" de "coracao".
- **O modelo pt reage ao acento:** dando o texto errado, a loss na região do
  acento sobe +10–14% em ruído médio, contra +1,5% no IAM original.
- **Mas a pressão é pequena:** um acento faltando custa 12–26% da loss
  **daquela amostra** em ruído médio e alto, mas só ~15% das amostras têm
  acento, e no lote isso vira poucos por cento.
- **Defeito encontrado nos dados:** **64% dos pares estavam desalinhados.**
  Quando o sinal passava da borda, a tela da acentuada era ampliada e a do par
  não; o pré-processamento escalava as duas de forma diferente.
  - **Correção:** `gerador.par_na_tela` e `scripts/alinhar_pares.py`, criando
    a base `iam_pt_alinhado`.
  - **Efeito:** a diferença entre acentuada e par caiu de 12,8% para 0,40%
    dos pixels, o tamanho de um acento.

### 6b. Controle × peso 5

**O que mudou.** Dois treinos iguais (16 épocas, base `iam_pt_alinhado`, 30%
de originais):
- **controle** (`iam_pt_alinhado.json`): loss original;
- **peso 5** (`iam_pt_peso5.json`): o erro na máscara do acento vale 5×, na
  acentuada **e no par** (ver [02](02_diffusionpen_e_modificacoes.md)).

Conferência antes de treinar (`diagnostico/conferir_mascara_acento.py`):
- todos os 5.188 pares têm máscara;
- a máscara cobre 7% do latente de uma acentuada e 2,3% do lote;
- com peso 5, o acento passa de 2% para 11% da loss do lote.

No começo do treino, o erro dentro da máscara era **3,6×** o de fora: o acento
é a parte que o modelo mais erra.

**Resultado (validação).**

| modelo | marca: acentuada | marca: esqueleto | marca: sem acento | diferença pareada | CER sem acento |
|---|---|---|---|---|---|
| pt (pares desalinhados) | 42% | 26% | 26% | +16% | 0,34 |
| controle (pares alinhados) | 46% | 30% | 25% | +16% | 0,32 |
| **peso 5** | **57%** | 34% | 32% | **+23%** | 0,32 |

Bootstrap por palavra, peso 5 − controle:
- diferença pareada **+7,1 pp** (IC95 +3,2 a +11,1);
- marca falsa +6,7 pp (IC95 +2,9 a +10,6).

**Conclusão.**
- **O peso funciona e o efeito é causal:** o controle mostra que alinhar os
  pares sozinho não mudou nada.
- **O peso também aumenta o vazamento,** e o CER não mudou.
- **Explicação do vazamento:** errar para menos (não desenhar o acento pedido)
  passou a custar 5×, enquanto errar para mais (acento numa palavra sem
  acento) continuou custando 1× na maior parte da imagem. Na dúvida, desenhar
  ficou mais barato.

### 6c. Posição do acento (`scripts/medir_posicao_acento.py`)

**Por quê.** O detector de marca não sabe **onde** a marca caiu: "hávera"
conta como acerto para "haverá".

**Resultado** (peso 5, palavras com um único sinal):

| sinal | letra certa | letra vizinha | outra letra | sem marca |
|---|---|---|---|---|
| agudo | 14% | 7% | **21%** | 57% |
| til | 50% | 16% | 2% | 32% |
| circunflexo | 4% | 1% | 10% | 85% |

**Conclusão.**
- **O agudo cai 2× mais em outra letra do que na certa.** O modelo aprende
  "esta palavra leva um agudo", não "o agudo vai nesta letra".
- **O til acerta metade:** está quase sempre no fim da palavra (-ão), uma
  posição fácil de aprender.
- **A cedilha não é medida:** ver [08](08_protocolo_de_avaliacao.md).

---

## Fase 7 — Por que a letra piora? Três controles

A fase 5 mostrou o CER subindo de 0,22 para 0,32–0,34. Esta fase testou as
hipóteses uma a uma.

### 7a. A base é menos legível? (`diagnostico/cer_base.py`)

Leitor independente em 1.500 imagens por grupo:

| o que foi lido | CER |
|---|---|
| escrita real do IAM (`iam_test`) | 0,12 |
| base: sem acento e pares (como o treino vê) | 0,19 |
| base: acentuadas | 0,27 |
| base lida sem a redução para 64 px | a mesma |

- **A reamostragem** (ampliar 2× e reduzir) **não é a causa.**
- **A base é tão legível quanto o gerador que a fez** (0,19 contra 0,18–0,22),
  e menos que a escrita real.
- **O acento desenhado** atrapalha a leitura em +0,08.
- **Os modelos ajustados (0,32–0,34) são piores que a própria base.**

### 7b. Mais escrita real resolve? (`model_iam_pt_peso5_orig100`)

O peso 5 com **100%** das originais do IAM (55.535 em vez de 16.660):
- **CER 0,35**, um pouco **pior** (+0,027, IC95 +0,010 a +0,045);
- diferença pareada **+25%**, igual dentro do ruído.

Mais escrita real não recuperou a letra.

### 7c. O fine-tune em si estraga? (`model_iam_so_originais`)

Mesmo `train.py`, mesmo pré-processamento e mesmos hiperparâmetros, **só com
as palavras originais do IAM** (base vazia, `base_vazia/`), 4 épocas:
- **CER 0,21**, igual ao IAM original (0,22). A diferença está dentro do ruído.

**Conclusão da fase 7.**
- **O regime de treino está correto:** o fine-tune sem dados sintéticos não
  piora nada.
- **O que piora a letra são os dados com acento sintético,** mesmo diluídos
  (23% ou 77% do treino).
- **Medição de nitidez do traço** (fração de tinta cinza, borrada):

| modelo | tinta borrada | contraste na borda |
|---|---|---|
| IAM original | 0,59 | 299 |
| só originais | 0,60 | 296 |
| controle e peso 5 (base pt) | 0,68 | 263–265 |

  As bases **geradas** borram o traço. O modelo copia a distribuição das
  gerações do IAM e soma os próprios erros: é uma cópia da cópia.

---

## Fase 8 — Volta à escrita real do IAM, com os pesos (`model_iam_acentuado_pares`)

**Por quê.** Se as imagens geradas borram, voltar para palavras **reais** do
IAM, mantendo os pesos que funcionaram.

**O que mudou.**
- **Base `iam_acentuado_teto25_pares`** (`scripts/criar_pares_iam.py`): a base
  com teto de 25% (9.892 acentuadas) + **o par** de cada uma. O par é a
  palavra original do IAM na mesma tela, e nada é gerado pelo modelo.
- **Dados:** 100% das originais, 75.319 amostras ao todo.
- **Pesos:** 5 no acento; **peso 2 na zona vazia** (a faixa sem tinta acima e
  abaixo do corpo da palavra), **em todas as amostras**, para cobrar acento
  onde o texto não pede.
- **Treino:** 12 épocas.

**Resultado.**

| | marca: esqueleto | marca: sem acento | diferença pareada | CER sem acento | CER sem acento pedido e sem marca |
|---|---|---|---|---|---|
| peso 5 (base pt) | 34% | 32% | +23% | 0,32 | 0,27 |
| **IAM real + pesos** | **54%** | **56%** | +20% | **0,46** | **0,35** |

- **O traço voltou a ser nítido:** tinta borrada 0,589, igual ao IAM original.
- **Mas a forma das letras piorou:** as hastes somem ("havera" vira "cavera"
  e "navera"), e o vazamento aumentou.
- Nas amostras, o modelo troca o tipo de sinal (circunflexo em vez de agudo)
  e põe sinais em várias letras ("provâvêl").

**Interpretação.**
- **A zona vazia cobrava 2× a tinta na faixa onde ficam as hastes,** e foi
  aplicada a quase todo o treino. Ficou mais barato encurtá-las.
- **A base em inglês tem acento em letra sorteada,** sem regra ortográfica. O
  rótulo diz qual sinal e onde, mas o padrão aprendido é "qualquer sinal, em
  qualquer lugar".

---

## Fase 9 — Penalidade só onde um acento caberia (`model_iam_acentuado_vogais`)

**Por quê.**
- Equilibrar o custo: acento a mais deve custar tanto quanto acento a menos.
- Fazer isso **sem** tocar nas hastes.

**O que mudou.** Só a zona: **"vogais"**, a faixa acima de cada vogal e abaixo
de cada c, nas colunas da letra (alinhador CTC), com **peso 5**, igual ao do
acento. O resto é igual à fase 8, 12 épocas.

Conferência antes de treinar (`diagnostico/conferir_zona_vogais.py`):
- 0,5% das células da zona sobre letras com haste, contra 21,6% da zona
  vazia;
- 62% das marcas falsas do modelo da fase 8 caem nessa faixa;
- 11% das palavras originais ficam sem máscara (alinhamento fraco).

**Resultado.**

| | marca: esqueleto | marca: sem acento | diferença pareada | CER sem acento pedido e sem marca |
|---|---|---|---|---|
| zona vazia (fase 8) | 54% | 56% | +20% | 0,35 |
| **zona vogais** | **48%** | **48%** | +21% | 0,36 |

Bootstrap por palavra, vogais − vazia:
- marca falsa **−6,7 pp** (IC95 −9,8 a −3,6);
- CER sem marca +0,016 (+0,001 a +0,032).

Posição: agudo 15% na letra certa, til 28%.

**Conclusão.**
- **A zona nas vogais reduz o acento falso,** mas pouco: ainda é ~10× o IAM
  original.
- **A letra não voltou.** As hastes reaparecem nas figuras, mas as letras
  continuam deformadas. **Então a zona vazia não era a causa principal da
  piora da letra.**
- **O que os treinos com letra ruim têm em comum,** e o fine-tune só com
  originais não tem: os dados com acento sintético (base pt ou base IAM com
  pares) e o peso no acento.

**Próximo teste sugerido.** Avaliar o `model_iam_acentuado_teto25` (fase 4b),
que já existe e nunca passou pelo protocolo de CER. Ele usa a mesma base, sem
pares e sem pesos. Isso separa "os dados sintéticos estragam a letra" de "o
peso estraga a letra". Ver [10](10_problemas_em_aberto.md).
