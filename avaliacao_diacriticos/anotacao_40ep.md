# Anotação humana do modelo de 40 épocas

Registro da anotação feita por duas pessoas sobre imagens do melhor modelo de
outubro de 2026, o acento separado com professor e zona vogais de 40 épocas
(`modelos/zona_vogais_40ep.pt`). O objetivo é saber se os detectores
automáticos concordam com o que uma pessoa vê, e se as duas pessoas concordam
entre si.

O desempate ainda não foi feito. Os números abaixo usam só as imagens em que
as duas pessoas deram a mesma resposta. A seção Pendências diz o que falta.

## O que foi anotado

São 212 imagens. 180 vêm da avaliação do modelo pelo protocolo do
`scripts/avaliar_pt.py`, com palavras do split de validação, 20 escritores do
iam_test e as sementes 42 e 43. As outras 32 são palavras com i, que a
avaliação não tem, geradas aqui na CPU pelo mesmo protocolo com
`anotacao_dupla.py gerar_i_j`. Antes de gerar, uma palavra da avaliação foi
refeita na CPU e comparada com a da avaliação, com diferença média de 0,5 numa
escala de 0 a 255.

| grupo | imagens |
|---|---|
| palavras com acento em cima (agudo, til, circunflexo, cedilha e til) | 80 |
| palavras só com cedilha | 55 |
| palavras sem sinal pedido (esqueletos e palavras sem acento) | 45 |
| palavras com i, com acento e esqueletos | 32 |

Um esqueleto é a palavra acentuada pedida sem o acento, com o mesmo escritor e
a mesma semente. A palavra rapé ficou fora da amostra porque o esqueleto dela
é uma palavra ofensiva em inglês.

## Como foi anotado

Dois colegas da equipe, Bruno e Vini, anotaram as mesmas 212 imagens, cada um
sozinho. Eles receberam folhas com as imagens numeradas e embaralhadas, as
imagens soltas, uma planilha e um texto de instruções. A folha mostra, em cima
de cada imagem, só as letras da palavra sem acento, numeradas. Nada dizia se a
palavra tinha sido pedida com acento, qual era o modelo ou o que o detector
achou.

Para cada imagem eles responderam três perguntas.

| coluna | resposta |
|---|---|
| `marcas_em_cima` | os números das letras com acento ou marca solta em cima, ou nenhuma, ou não sei |
| `cedilha` | os números das letras com cedilha, ou nenhuma, ou não sei |
| `legivel` | sim, não ou não sei |

O pingo do i e do j, o traço do t e as hastes de letras altas não contavam
como marca.

Comparando as letras marcadas com a palavra pedida, o script põe cada imagem
numa de quatro classes.

| classe | quando |
|---|---|
| só o certo | marca em todas as letras pedidas e em nenhuma outra |
| certo e a mais | marca nas letras pedidas e em pelo menos mais uma |
| só a mais | marca em outra letra e não na pedida |
| nenhum | nenhuma marca em cima |

Numa palavra sem sinal em cima pedido, qualquer marca em cima conta como a
mais.

As duas planilhas voltaram completas, com 212 linhas e as letras batendo com a
chave. Ninguém usou não sei. Houve um erro de digitação (img_013 do Vini, com
cedilha 6 7 numa palavra de 5 letras) e 13 respostas de cedilha apontando uma
letra vizinha do c, o que não muda nada porque a cedilha é comparada só como
tem ou não tem.

## Métodos automáticos comparados

O detector do colega é o `scripts/medir_marcas.py` com a letra dada pelo
`scripts/medir_posicao_acento.py`. O primeiro acha manchas de tinta soltas,
pequenas, inteiras acima ou abaixo do corpo das letras. O segundo usa o
alinhador para dizer a que letra cada mancha pertence. Ele não distingue o
tipo do acento. A cedilha dele é uma mancha solta embaixo de um c.

A cedilha abaixo do c é o `experimentos/cedilha_abaixo.py`. Ela mede a tinta
abaixo da linha de base nas colunas do c, numa imagem só, com o limiar tirado
dos c sem cedilha do próprio modelo. Na prática, basta haver tinta abaixo do
c.

A marca do `paineis.tsv` é a coluna usada em `diagnostico/grafico_epocas.py`
para a medida de acento a mais nas palavras só com cedilha. Ela vale 1 se o
detector do colega acha qualquer mancha solta, em cima ou embaixo.

Nas palavras com i ou j o detector do colega conta o pingo como marca. Por
isso, nesse grupo, só as duas pessoas são comparadas.

## Kappa

O kappa de Cohen mede a concordância entre duas avaliações descontando a
concordância que viria por acaso. A conta é

```
kappa = (concordancia observada - concordancia por acaso) / (1 - concordancia por acaso)
```

A concordância por acaso sai das proporções de cada resposta. Se uma pessoa
diz sim em 70% das imagens e a outra em 60%, as duas concordariam em
0,7 x 0,6 + 0,3 x 0,4 = 54% das vezes respondendo ao acaso.

Usamos o kappa porque a concordância simples engana quando uma resposta
domina. Se quase nenhuma imagem tem cedilha, um detector que sempre diz não
concorda muito sem detectar nada. O kappa dele fica perto de zero.

A escala usual é a de Landis e Koch (1977).

| kappa | concordância |
|---|---|
| abaixo de 0 | pior que o acaso |
| 0 a 0,20 | leve |
| 0,21 a 0,40 | razoável |
| 0,41 a 0,60 | moderada |
| 0,61 a 0,80 | substancial |
| 0,81 a 1 | quase perfeita |

O kappa tem um limite que aparece neste conjunto. Quando quase todas as
respostas são iguais, ele vai a zero ou não pode ser calculado, mesmo com as
duas avaliações quase idênticas. Nas 45 imagens sem sinal pedido as duas
pessoas responderam nenhum em todas, e o kappa não existe. Nesses grupos vale
olhar as contagens.

## Precisão e recall

Precisão e recall servem quando um lado é a referência e a pergunta é sim ou
não. Aqui a referência é o consenso das duas pessoas, e a pergunta é, por
exemplo, se há cedilha.

| termo | o que é |
|---|---|
| VP, verdadeiro positivo | a pessoa vê e o detector acusa |
| FP, falso positivo | o detector acusa e a pessoa não vê |
| FN, falso negativo | a pessoa vê e o detector não acusa |
| precisão | VP / (VP + FP), das vezes que o detector acusa, quantas estão certas |
| recall | VP / (VP + FN), do que a pessoa vê, quanto o detector pega |

O kappa resume os dois lados num número só. Precisão e recall dizem para que
lado o detector erra.

## Resultados

### Concordância entre as duas pessoas

| pergunta | grupo | concordam | kappa |
|---|---|---|---|
| classe do acento em cima | com acento em cima | 69 de 80 | 0,64 |
| classe do acento em cima | com i | 28 de 32 | 0,78 |
| classe do acento em cima | só cedilha | 52 de 55 | 0,00 |
| classe do acento em cima | sem sinal pedido | 45 de 45 | não se calcula |
| cedilha | fora as palavras com i | 176 de 177 | |
| legível | com acento em cima | 62 de 80 | 0,44 |
| legível | só cedilha | 39 de 55 | 0,15 |
| legível | sem sinal pedido | 34 de 45 | 0,44 |
| legível | com i | 26 de 32 | 0,29 |

Na classe do acento e na cedilha a anotação é confiável, com concordância
substancial ou quase total. Os kappas de 0,00 e não se calcula são o limite do
kappa descrito acima, já que quase todas as respostas são nenhum. Na
legibilidade a concordância é leve a moderada, e esse número deve ser usado
com cuidado.

### Detectores contra o consenso

Imagens em que as duas pessoas deram a mesma resposta, sem as palavras com i.

| pergunta | detector | n | VP | FP | FN | precisão | recall | kappa |
|---|---|---|---|---|---|---|---|---|
| cedilha | detector do colega | 177 | 8 | 0 | 54 | 1,00 | 0,13 | 0,16 |
| cedilha | cedilha abaixo do c | 177 | 57 | 1 | 5 | 0,98 | 0,92 | 0,92 |
| alguma marca em cima | detector do colega | 166 | 58 | 6 | 7 | 0,91 | 0,89 | 0,84 |
| acento pedido presente | detector do colega | 69 | 48 | 1 | 12 | 0,98 | 0,80 | 0,45 |
| acento a mais | detector do colega | 69 | 6 | 11 | 2 | 0,35 | 0,75 | 0,38 |
| a mais nas só cedilha | marca do paineis.tsv | 52 | 0 | 14 | 0 | 0,00 | | 0,00 |
| a mais nas só cedilha | só marcas em cima | 52 | 0 | 3 | 0 | 0,00 | | 0,00 |

Contra cada pessoa separada, a classe do detector do colega nas 80 palavras
com acento em cima tem kappa 0,32 com as duas.

### Leitura

O detector do colega sabe bem se há marca em cima, com kappa 0,84. Quando ele
diz que o acento pedido está lá, quase sempre está, mas perde um em cada cinco
que as pessoas veem. O ponto fraco é o acento a mais. Ele acusa acento a mais
em 17 imagens e as pessoas veem em 8, então quase dois terços dos acentos a
mais que ele aponta não existem.

A cedilha abaixo do c funciona nas imagens geradas, com kappa 0,92, precisão
0,98 e recall 0,92. É o primeiro detector de cedilha que passa na anotação.
O detector de marcas soltas do colega perde 54 das 62 cedilhas, porque a
cedilha costuma sair presa ao c.

A medida de acento a mais nas palavras só com cedilha, tirada da marca do
`paineis.tsv`, não concorda com as pessoas. Nas 52 imagens em que elas
concordam, elas não veem acento a mais em nenhuma, e a medida acusa 14. Das
15 marcas dessa coluna nas 55 imagens só com cedilha, 9 são a própria cedilha
solta contada como marca. Contando só as marcas em cima, sobram 3 falsos
alarmes. Com zero casos positivos não há como medir o recall dessa medida
nesta amostra.

Nas 32 imagens com i, o detector do colega acusa acento a mais em 18, porque
conta o pingo. As pessoas veem acento a mais em 2 (Bruno) e 4 (Vini).

## Pendências

O desempate não foi feito. São 68 imagens com discordância em pelo menos uma
pergunta, 51 em legibilidade, 18 na classe do acento e 4 na cedilha. Elas
estão em `resultados/anotacao_40ep_desempate.csv`, com as respostas das duas
pessoas lado a lado e as colunas `decisao_marcas_em_cima`, `decisao_cedilha` e
`decisao_legivel` em branco, para preencher no mesmo formato da planilha.

O script ainda não lê as decisões do desempate. Depois de preenchidas, falta
montar o gabarito final, com a resposta comum onde as pessoas concordam e a
decisão onde discordam, e refazer a tabela de detectores contra esse
gabarito.

Também falta medir onde o detector do colega erra a posição do acento, letra a
letra, que era a terceira tarefa pedida.

## Arquivos

| arquivo | conteúdo |
|---|---|
| `anotacao_dupla.py` | gera as palavras com i, monta o pacote, grava as respostas dos detectores e calcula a concordância |
| `resultados/anotacao_40ep_chave.csv` | liga cada imagem à palavra pedida, ao tipo e ao painel original |
| `resultados/anotacao_40ep_detector.csv` | as respostas dos detectores para as 212 imagens |
| `resultados/anotacao_40ep_concordancia.txt` | a saída completa da concordância |
| `resultados/anotacao_40ep_desempate.csv` | as 68 discordâncias, para o desempate |
| `planilha_anotacao_bruno.csv`, `planilha_anotacao_vini.csv` | as planilhas preenchidas, na raiz do repositório |

A chave e as respostas dos detectores ainda não estão no git, porque mostram a
palavra pedida de cada imagem. Agora que as planilhas voltaram elas podem
entrar.

Para refazer a concordância, que só precisa do Python padrão e roda em
segundos,

```
python3 avaliacao_diacriticos/anotacao_dupla.py concordancia --planilhas planilha_anotacao_bruno.csv planilha_anotacao_vini.csv
```
