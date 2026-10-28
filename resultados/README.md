# Resultados

O caminho do trabalho em uma página, da primeira tentativa ao modelo atual,
com os números principais e uma figura de cada fase. Os detalhes de cada passo
estão nos documentos de `docs/`, citados em cada seção.

Todos os números de modelo são da validação, pelo protocolo de
`scripts/avaliar_pt.py` ([docs/08](../docs/08_protocolo_de_avaliacao.md)),
salvo onde dito. O split de teste nunca foi aberto.

Um resumo da métrica para a equipe está em
[metrica/](metrica/README.md). As figuras para o artigo estão em
[figuras/artigo/](figuras/artigo/README.md).

| fase | pergunta | resposta |
|---|---|---|
| 1. Sonda | o DiffusionPen original escreve acento? | não, omite ou deforma |
| 2. BRESSAY | treinar em escrita real em português resolve? | não, piorou tudo |
| 3. Acentos sintéticos | e desenhar o acento nas imagens de treino? | o acento aparece, mas também onde não deve, e a letra piora |
| 4. Peso no acento | cobrar mais o erro na região do acento? | mais acento, mais acento indevido |
| 5. Acento separado e professor | separar letra e sinal na entrada e ancorar no modelo original? | acento indevido e letra voltam ao nível do original |
| 6. Zona vogais e treino longo | equilibrar o custo de errar para mais e para menos? | o melhor modelo, acento pedido em 88% e acento a mais em 5% |
| 7. Validação humana e métrica | o que medimos bate com o que uma pessoa vê? | sim para presença, cedilha e letra do acento; o tipo do sinal ainda não é medido |
| 8. Métrica final nos treinos | quantas palavras cada modelo acerta por inteiro? | 72% no melhor modelo, contra 44% do professor |

## 1. Sonda: o modelo original não sabe desenhar acento

O DiffusionPen público, treinado no IAM (inglês), recebe o texto com acento,
mas não desenha o sinal. Em "mão", "ação", "você" ou "força" o acento some ou
a palavra deforma. Detalhes em [docs/04](../docs/04_sonda_e_metrica_inicial.md).

![Sonda do modelo original](figuras/01_sonda_modelo_original.png)

## 2. BRESSAY: escrita real em português não resolveu

O primeiro fine-tune usou o BRESSAY, base de escrita manuscrita em português.
A letra piorou em tudo e nenhum acento foi aprendido. As palavras do BRESSAY
têm cerca de 26 px de altura, e nesse tamanho o acento ocupa 1 ou 2 px. Ver
[docs/05](../docs/05_finetune_bressay.md).

Na figura, o modelo original à esquerda e os dois fine-tunes no BRESSAY.

![Fine-tune no BRESSAY](figuras/02_bressay_40ep.png)

## 3. Acentos sintéticos: o acento aparece

A saída foi criar dados. Palavras em português são geradas pelo próprio
DiffusionPen e um script desenha o acento sobre a letra certa, localizada por
um alinhador. Cada palavra acentuada tem um par, a mesma imagem sem o sinal.
Ver [docs/06](../docs/06_acentos_sinteticos.md).

![Base sintética](figuras/03_base_sintetica.png)

Com essa base o modelo passa a desenhar acento a partir do texto, o que o
original não faz. Mas também põe acento onde não se pede, e a letra fica pior.
Os controles mostraram que a piora vem dos dados sintéticos e não do fine-tune
em si. Ver [docs/07](../docs/07_experimentos_com_acentos.md).

## 4. Peso no acento

O acento ocupa 2% da imagem no espaço latente. Cobrar 5 vezes mais o erro
nessa região aumentou o acento pedido e também o indevido.

| modelo, 16 épocas | marca na acentuada | marca no esqueleto | marca sem acento | diferença pareada | CER sem acento |
|---|---|---|---|---|---|
| IAM original | 6% | 5% | 5% | +1 | 0,22 |
| controle, sem peso | 46% | 30% | 25% | +16 | 0,32 |
| peso 5 | 57% | 34% | 32% | +23 | 0,32 |

O controle e o peso 5 são o mesmo treino na base sintética, e diferem só no
peso. Por isso a diferença entre eles, +7 pontos na diferença pareada (IC95 de
+3 a +11), é atribuída ao peso. O CER desta tabela é o das 20 palavras sem
acento.

O esqueleto é a mesma palavra pedida sem o acento. A diferença pareada mede
quanto do acento vem do texto. Cada palavra acentuada é gerada junto com o seu
esqueleto, com o mesmo escritor e a mesma semente, então a única diferença
entre as duas imagens é o texto pedido.

| "haverá" | "havera" | conta |
|---|---|---|
| marca | sem marca | +1, o acento veio do texto |
| marca | marca | 0, o modelo põe marca de qualquer jeito |
| sem marca | sem marca | 0, não desenhou |
| sem marca | marca | −1, o contrário do pedido |

A diferença pareada é a média disso, ou seja, a marca na acentuada menos a
marca no esqueleto. Um modelo que pusesse acento em tudo teria 100% de marca
na acentuada e diferença pareada zero.

Na figura, "haverá" e "havera" no modelo original, no controle sem peso e no
peso 5. Barra verde é marca detectada na acentuada e vermelha no esqueleto.

![Peso 5](figuras/03_peso5_havera.png)

Tentativas seguintes, como penalizar o acento nas regiões vazias ou voltar a
palavras reais do IAM, reduziram pouco o acento indevido e não trouxeram a
letra de volta.

## 5. Acento separado e professor

Uma investigação com testes pequenos na CPU ([docs/13](../docs/13_investigacao_do_vazamento_e_da_letra.md))
levou a duas mudanças no treino ([docs/14](../docs/14_acento_separado_e_professor.md)).

- Acento separado. O codificador de texto lê a palavra sem acento, e o sinal
  entra por um vetor próprio, somado só na posição da letra. A parte das letras
  fica igual à que o modelo original conhece.
- Professor. Uma cópia congelada do modelo original. Nas palavras sem acento,
  o modelo novo aprende a responder como ele, o que protege a letra.

| modelo, 16 épocas | marca na acentuada | marca no esqueleto | marca sem acento | diferença pareada | CER sem acento pedido |
|---|---|---|---|---|---|
| IAM original | 6% | 5% | 5% | +1 | 0,195 |
| peso 5 | 57% | 34% | 32% | +23 | 0,312 |
| acento separado | 61% | 11% | 11% | +51 | 0,281 |
| acento separado e professor | 71% | 5% | 8% | +66 | 0,190 |

As duas peças são necessárias, cada uma para uma coisa.

| | sem professor | com professor |
|---|---|---|
| texto inteiro pelo codificador | acento indevido em 32%, letra ruim | acento indevido em 20%, letra boa |
| acento separado | acento indevido em 11%, letra ruim | acento indevido em 5%, letra boa |

![Acento separado e professor](figuras/05_separado_professor_havera.png)

## 6. Zona vogais e treino longo: o modelo atual

Com o professor, o problema restante era o acento a mais dentro da palavra
acentuada, como "sófrerá" no lugar de "sofrerá". O peso 5 só valia onde o
acento estava, então errar para menos custava 5 e errar para mais custava 1. A
zona vogais põe o mesmo peso em cima de toda vogal e embaixo de todo c. Ver
[docs/15](../docs/15_acento_a_mais_e_variantes_do_professor.md).

| | professor, 16 épocas | zona vogais, 40 épocas |
|---|---|---|
| acento pedido, palavras com agudo, til ou circunflexo | 86% | 88% |
| o mesmo, palavras sem haste alta | 76% | 88% |
| acento a mais, palavras só com cedilha | 27% | 5% |
| marca em palavra que não pede acento | 5% e 8% | 4% e 6% |
| CER das palavras sem acento pedido | 0,190 | 0,186 |
| CER das palavras acentuadas | 0,43 | 0,39 |

![Medidas por época](figuras/06_evolucao_por_epocas.png)

O treino melhora até 30 épocas e estaciona em 40. A letra das palavras sem
acento fica no nível do modelo original o tempo todo.

Nas figuras, o professor de 16 épocas em cima e a zona vogais de 40 embaixo.
Em "caçadora" os acentos sobrando somem e a cedilha fica. Em "pôr" o sinal
passa a cair sobre o "o". Em "relâmpagos" ele sai do segundo "a" para o
primeiro.

![caçadora](figuras/06_professor_zona40_cacadora.png)
![pôr](figuras/06_professor_zona40_por.png)
![relâmpagos](figuras/06_professor_zona40_relampagos.png)

Três variantes para limpar a letra das palavras acentuadas foram testadas e
descartadas. Todas limparam a letra e tiraram o acento das palavras sem haste
alta, como "pôr" e "pás". Na base, a palavra acentuada é sempre menor que a
normal, para o sinal caber em cima, e as variantes cobravam do modelo a
palavra no tamanho normal.

## 7. Validação humana e métrica

Duas pessoas anotaram 212 imagens do modelo de 40 épocas sem saber o que tinha
sido pedido, e uma terceira desempatou. Ver
[avaliacao_diacriticos/anotacao_40ep.md](../avaliacao_diacriticos/anotacao_40ep.md).

![Folha de anotação](figuras/07_folha_de_anotacao.png)

As duas pessoas concordam bem sobre acento e cedilha, e pouco sobre
legibilidade.

| pergunta | concordam | kappa |
|---|---|---|
| acento em cima, palavras com acento | 69 de 80 (86%) | 0,64 |
| acento em cima, palavras com i | 28 de 32 (88%) | 0,78 |
| acento em cima, palavras só com cedilha | 52 de 55 (95%) | |
| acento em cima, palavras sem acento pedido | 45 de 45 (100%) | |
| cedilha | 176 de 177 (99%) | |
| legível | 161 de 212 (76%) | de 0,15 a 0,44 conforme o grupo |

Nas linhas sem kappa quase todas as respostas são iguais, e o kappa vai a zero
ou não pode ser calculado mesmo com as duas pessoas concordando, pelo mesmo
motivo explicado mais abaixo.

O que elas viram no modelo de 40 épocas.

| | gabarito humano |
|---|---|
| acento pedido presente, palavras com sinal em cima | 71 de 80 |
| acento a mais nessas palavras | 12 de 80 |
| cedilha pedida desenhada | 62 de 63 |
| cedilha onde não se pediu | 0 de 117 |
| qualquer marca em palavra que não pede acento | 0 de 45 |
| legível | 142 de 180 |

Como as medidas automáticas se saem contra esse gabarito.

| pergunta | medida | concordam | kappa |
|---|---|---|---|
| há marca em cima | detector de marcas soltas | 162 de 180 (90%) | 0,79 |
| há cedilha | tinta abaixo do c, criada nesta etapa | 173 de 180 (96%) | 0,91 |
| há cedilha | detector de marcas soltas | 125 de 180 (69%) | 0,15 |
| o acento pedido está na letra certa | detector com alinhador | 61 de 80 (76%) | 0,35 |
| há acento em letra que não o pede | detector com alinhador | 63 de 80 (79%) | 0,36 |

Nas duas últimas linhas o detector erra para lados diferentes. Quando diz que
o acento pedido está na letra certa, acerta 98% das vezes, mas perde um em
cada quatro que as pessoas veem. Quando acusa acento a mais, acerta só 38%
das vezes.

O kappa e a proporção de concordância às vezes contam histórias diferentes.
O detector de marcas soltas concorda com as pessoas em 69% das imagens sobre a
cedilha, e mesmo assim tem kappa 0,15. Das 180 imagens, 118 não têm cedilha.
Um detector que respondesse "não" para todas acertaria 118, ou 66%, sem
detectar nada, e esse detector faz quase isso: acha 8 das 62 cedilhas. O kappa
desconta esse acerto que viria por acaso, e por isso é a medida principal. O
mesmo vale para o acento a mais, com 79% de concordância e só 12 casos em 80.

A cedilha, 40% dos sinais avaliados, passou a ser medida. Na figura, cedilhas
pedidas que a medida nova acerta, com a coluna do c em azul e o corte em
vermelho.

![Cedilha](figuras/07_cedilha_acertos.png)

A anotação também corrigiu uma medida de treino. O acento a mais vinha sendo
contado com qualquer marca solta, e nas palavras com ç a marca embaixo é a
própria cedilha solta. Corrigido, o acento a mais caiu de 22% para 5% no
modelo de 40 épocas. A correção está no [docs/15](../docs/15_acento_a_mais_e_variantes_do_professor.md).

## 8. Métrica final nos treinos

Com a anotação como gabarito, as medidas de presença e de posição viraram uma
métrica só ([avaliacao_diacriticos/README.md](../avaliacao_diacriticos/README.md),
seção "Métrica final"). Cada palavra pedida com acento recebe uma categoria.

| categoria | quando |
|---|---|
| correto | todos os sinais pedidos, na letra certa, e nada a mais |
| a mais | os sinais pedidos estão certos, mas há marca em outra letra |
| letra errada | o acento pedido falta na letra dele, e há marca em outra |
| ausente | falta um sinal pedido |

Duas mudanças no detector de marcas vieram da anotação. Uma marca que cai numa
consoante vai para a vogal mais próxima, porque acento só cai em vogal. E uma
marca alta, inteira acima da palavra, deixa de ser descartada como haste. Por
vogal, a letra marcada concorda com as pessoas com kappa 0,75, medido em
palavras que não foram usadas para decidir as regras.

Aplicada aos modelos de treino, nas 1.200 palavras acentuadas de cada um:

| modelo | correto | a mais | letra errada | ausente | marca em palavra sem acento |
|---|---|---|---|---|---|
| só professor, 11 épocas | 15% | 23% | 16% | 46% | 20% |
| professor, 16 épocas | 44% | 26% | 16% | 13% | 5% |
| pares sem professor, 16 épocas | 47% | 18% | 15% | 20% | 10% |
| fora da máscara, 16 épocas | 39% | 11% | 15% | 36% | 5% |
| fora da máscara em ruído baixo, 16 épocas | 43% | 8% | 19% | 31% | 4% |
| zona vogais, 16 épocas | 55% | 14% | 14% | 17% | 5% |
| zona vogais, 30 épocas | 68% | 9% | 11% | 12% | 4% |
| zona vogais, 40 épocas | 72% | 8% | 8% | 12% | 4% |

A métrica conta a história do treino num número só. A zona vogais de 40 épocas
acerta a palavra inteira em 72% dos casos, contra 44% do professor. As
variantes com o professor fora da máscara aparecem com "ausente" alto, porque
perdem o acento nas palavras sem haste.

Uma ressalva sobre os valores absolutos. No modelo de 40 épocas, as pessoas
viram 87% das palavras corretas, e a métrica dá 75% nas mesmas imagens. A
diferença vem quase toda do acento que encosta na letra: a pessoa o vê, e o
detector, que procura mancha solta, não. O "a mais" bate, 6% nas duas. Para
comparar modelos a métrica serve; para dizer quantas palavras um modelo
acerta, o número da métrica fica uns 12 pontos abaixo do real.

### Por classe de sinal

O planejamento do TCC pedia a métrica separada por classe de caractere.
Separando pelo que a palavra pede, no modelo de 40 épocas, pelo gabarito
humano das imagens anotadas:

| classe | imagens anotadas | correto pelas pessoas | correto pela métrica |
|---|---|---|---|
| cedilha | 55 | 96% | 89% |
| cedilha e til | 8 | 100% | 88% |
| til | 20 | 90% | 65% |
| agudo | 36 | 78% | 64% |
| circunflexo | 16 | 62% | 56% |

A cedilha, sinal embaixo e preso ao c, é o que o modelo acerta melhor. Entre os
sinais em cima, o til vai melhor que o agudo, e o circunflexo é o mais fraco.
O agudo erra mais de letra: pela métrica, 18% das palavras com agudo têm o
acento só em outra vogal, contra 6% do til.

A métrica erra de forma diferente em cada classe. Ela subestima o til em 25
pontos, porque o til costuma encostar na letra, cobrir duas vogais ou sair
partido, e com isso o põe no nível do agudo. A ordem entre as classes vale
pelo gabarito humano. A tabela da métrica por classe, para todos os modelos,
está em `avaliacao_diacriticos/resultados/sinais/por_classe.md`. Circunflexo e
cedilha com til dependem de duas palavras cada.

### A palavra estragou? Integridade e as quatro categorias

O planejamento previa dois eixos: a presença do acento (E1, o que está acima) e
a integridade da palavra (E2), se ela continua legível descontando o acento. O
E2 é medido em pares, com o mesmo escritor e a mesma semente: o erro de leitura
da palavra acentuada menos o do esqueleto dela. Antes de ler, os sinais das
duas imagens são apagados, para o leitor não ler o acento como letra. Assim o
rigor do leitor e a dificuldade da palavra se cancelam, e sobra o que o acento
fez com a letra.

| modelo | piora da letra (dCER médio) | acerto | omissão | degradação com diacrítico | degradação sem diacrítico |
|---|---|---|---|---|---|
| professor, 16 épocas | +0,20 | 48% | 6% | 38% | 8% |
| fora da máscara, 16 épocas | +0,07 | 53% | 31% | 10% | 5% |
| pares sem professor, 16 épocas | +0,05 | 66% | 15% | 14% | 5% |
| zona vogais, 40 épocas | +0,17 | 53% | 6% | 36% | 6% |

As quatro categorias são as do planejamento, com degradação definida como o
erro de leitura subir mais de 0,2 em relação ao esqueleto. O E2 mostra a troca
que o treino já tinha visto: as variantes que limpam a letra perdem o acento
(omissão alta), e o melhor modelo mantém o acento com a letra pior que a do
esqueleto em um terço das palavras. O E2 ainda não tem validação humana
direta. Detalhes em [avaliacao_diacriticos/README.md](../avaliacao_diacriticos/README.md).

## O que falta

- Acento a mais nas palavras com acento em cima, 15% pelo gabarito humano.
- Letra das palavras acentuadas, CER 0,39 contra 0,27 nas imagens da própria
  base. As tentativas apontam para a base, onde a palavra acentuada encolhe.
- Validar o E2 com pessoas, mostrando acentuada e esqueleto lado a lado.
- Na métrica, o tipo do sinal (til no lugar de agudo, por exemplo) e o acento
  que encosta na letra. A ideia é um leitor que leia a palavra com acento.

## Limitações

- Palavras com "i" e "j" ficam fora da avaliação automática, porque o pingo
  confunde o detector. Elas entraram só na anotação humana.
- Uma rodada de treino por configuração, sem repetição com outra semente.
- A validação humana é de um modelo só.
