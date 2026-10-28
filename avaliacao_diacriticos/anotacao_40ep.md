# Anotação humana do modelo de 40 épocas

Registro da anotação feita por duas pessoas sobre imagens do melhor modelo de
outubro de 2026, o acento separado com professor e zona vogais de 40 épocas
(`modelos/zona_vogais_40ep.pt`). O objetivo é saber se os detectores
automáticos concordam com o que uma pessoa vê, e se as duas pessoas concordam
entre si.

Onde as duas pessoas discordaram, uma terceira pessoa desempatou. O gabarito
final tem a resposta comum onde elas concordam e a decisão do desempate onde
discordam, e é contra ele que os detectores são comparados.

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

Dois colegas da equipe, Bruno e Vinícius, anotaram as mesmas 212 imagens, cada um
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
chave. Ninguém usou não sei. Houve um erro de digitação (img_013 do Vinícius, com
cedilha 6 7 numa palavra de 5 letras) e 13 respostas de cedilha apontando uma
letra vizinha do c, o que não muda nada porque a cedilha é comparada só como
tem ou não tem.

## Desempate

As duas pessoas discordaram em pelo menos uma pergunta em 68 imagens, 51 em
legibilidade, 18 na classe do acento e 4 na cedilha, 73 respostas no total.
O autor do trabalho decidiu essas 68 imagens com duas folhas que mostram cada
imagem ampliada, as letras numeradas e as respostas das duas pessoas nas
perguntas em disputa, sem a palavra pedida. As decisões estão em
`resultados/anotacao_40ep_desempate.csv`.

| pergunta | igual ao Bruno | igual ao Vinícius | diferente dos dois |
|---|---|---|---|
| classe do acento em cima | 7 | 10 | 1 |
| cedilha | 3 | 1 | 0 |
| legível | 29 | 22 | 0 |

A resposta diferente dos dois é img_069 (pérsico), com marca nas letras 2 e 5.
Quem desempatou também respondeu as perguntas sem disputa dessas imagens, e
em todas deu a mesma resposta das duas pessoas.

Em img_101 (pedágio) a decisão marca cedilha na letra 5, que é o g. A palavra
tem i e fica fora da comparação com os detectores.

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
não. Aqui a referência é o gabarito final, e a pergunta é, por exemplo, se há
cedilha.

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

### Detectores contra o gabarito final

Todas as imagens, sem as palavras com i.

| pergunta | detector | n | VP | FP | FN | precisão | recall | kappa |
|---|---|---|---|---|---|---|---|---|
| cedilha | detector do colega | 180 | 8 | 1 | 54 | 0,89 | 0,13 | 0,15 |
| cedilha | cedilha abaixo do c | 180 | 57 | 2 | 5 | 0,97 | 0,92 | 0,91 |
| alguma marca em cima | detector do colega | 180 | 66 | 7 | 11 | 0,90 | 0,86 | 0,79 |
| acento pedido presente | detector do colega | 80 | 53 | 1 | 18 | 0,98 | 0,75 | 0,35 |
| acento a mais | detector do colega | 80 | 8 | 13 | 4 | 0,38 | 0,67 | 0,36 |
| a mais nas só cedilha | marca do paineis.tsv | 55 | 0 | 15 | 1 | 0,00 | 0,00 | -0,04 |
| a mais nas só cedilha | só marcas em cima | 55 | 0 | 4 | 1 | 0,00 | 0,00 | -0,03 |

Contra só o consenso, sem as imagens desempatadas, os números eram estes.

| pergunta | detector | n | VP | FP | FN | precisão | recall | kappa |
|---|---|---|---|---|---|---|---|---|
| cedilha | detector do colega | 177 | 8 | 0 | 54 | 1,00 | 0,13 | 0,16 |
| cedilha | cedilha abaixo do c | 177 | 57 | 1 | 5 | 0,98 | 0,92 | 0,92 |
| alguma marca em cima | detector do colega | 166 | 58 | 6 | 7 | 0,91 | 0,89 | 0,84 |
| acento pedido presente | detector do colega | 69 | 48 | 1 | 12 | 0,98 | 0,80 | 0,45 |
| acento a mais | detector do colega | 69 | 6 | 11 | 2 | 0,35 | 0,75 | 0,38 |
| a mais nas só cedilha | marca do paineis.tsv | 52 | 0 | 14 | 0 | 0,00 | | 0,00 |
| a mais nas só cedilha | só marcas em cima | 52 | 0 | 3 | 0 | 0,00 | | 0,00 |

As imagens desempatadas são as mais difíceis de julgar, e todos os kappas
caem um pouco quando elas entram. A maior queda é no acento pedido presente,
de 0,45 para 0,35, porque das 11 imagens que entraram nessa linha o detector
perde o acento em 6.

Contra cada pessoa separada, a classe do detector do colega nas 80 palavras
com acento em cima tem kappa 0,32 com as duas.

### O que o modelo desenhou

Pelo gabarito final, por grupo de imagens.

| grupo | imagens | sinal pedido presente | acento a mais | legível |
|---|---|---|---|---|
| com acento em cima | 80 | 71 | 12 | 61 |
| só cedilha | 55 | 54 | 1 | 45 |
| esqueletos | 30 | | 0 | 24 |
| sem acento | 15 | | 0 | 12 |
| com i, acentuadas | 16 | 14 | 2 | 13 |
| com i, esqueletos | 16 | | 2 | 15 |

Na linha com acento em cima, o sinal pedido presente é o acento em cima. Nas
8 dessas imagens que também pedem cedilha, a cedilha aparece em todas. Nas
palavras com i o pingo não conta como acento. A cedilha não aparece em
nenhuma imagem que não a pediu, fora a img_101 descrita no desempate. A
legibilidade tem concordância baixa entre as duas pessoas e a coluna deve ser
lida com cuidado.

### Leitura

O detector do colega sabe bem se há marca em cima, com kappa 0,79. Quando ele
diz que o acento pedido está lá, quase sempre está, mas perde um em cada
quatro que as pessoas veem, 18 de 71. O ponto fraco é o acento a mais. Ele
acusa acento a mais em 21 imagens e o gabarito tem 12, das quais ele pega 8.
Quase dois terços dos acentos a mais que ele aponta não existem.

A cedilha abaixo do c funciona nas imagens geradas, com kappa 0,91, precisão
0,97 e recall 0,92. É o primeiro detector de cedilha que passa na anotação.
O detector de marcas soltas do colega perde 54 das 62 cedilhas, porque a
cedilha costuma sair presa ao c.

A medida de acento a mais nas palavras só com cedilha, tirada da marca do
`paineis.tsv`, não concorda com as pessoas. Nas 55 imagens o gabarito tem
acento a mais em uma, img_087 (lançaremos), com acento sobre o o. A medida
acusa 15 e não pega essa. Das 15 marcas, 9 são a própria cedilha solta
contada como marca. Contando só as marcas em cima, sobram 4 falsos alarmes, e
img_087 continua perdida. Com um caso positivo só, o recall dessa medida não
diz muito.

Nas 32 imagens com i, o detector do colega acusa acento a mais em 18, porque
conta o pingo. O gabarito tem acento a mais em 4.

## Pendências

Falta medir onde o detector do colega erra a posição do acento, letra a
letra, que era a terceira tarefa pedida. O gabarito guarda só a classe de
cada imagem. As letras marcadas estão nas planilhas e no desempate.

## Arquivos

| arquivo | conteúdo |
|---|---|
| `anotacao_dupla.py` | gera as palavras com i, monta o pacote, grava as respostas dos detectores, calcula a concordância e monta o gabarito |
| `resultados/anotacao_40ep_chave.csv` | liga cada imagem à palavra pedida, ao tipo e ao painel original |
| `resultados/anotacao_40ep_detector.csv` | as respostas dos detectores para as 212 imagens |
| `resultados/anotacao_40ep_concordancia.txt` | a saída completa da concordância e da comparação com o gabarito |
| `resultados/anotacao_40ep_desempate.csv` | as 68 discordâncias, com as respostas das duas pessoas e as decisões |
| `resultados/anotacao_40ep_gabarito.csv` | o gabarito final das 212 imagens, com as perguntas que foram desempatadas |
| `resultados/planilha_anotacao_bruno.csv`, `resultados/planilha_anotacao_vini.csv` | as planilhas preenchidas |
| `amostras/anotacao_40ep/desempate/folha_desempate_{1,2}.png` | as folhas usadas no desempate, fora do git |

Para refazer a concordância e o gabarito, que só precisam do Python padrão e
rodam em segundos,

```
python3 avaliacao_diacriticos/anotacao_dupla.py concordancia \
    --planilhas avaliacao_diacriticos/resultados/planilha_anotacao_bruno.csv \
    avaliacao_diacriticos/resultados/planilha_anotacao_vini.csv
```

Se `resultados/anotacao_40ep_desempate.csv` já existe, o comando lê as
decisões dele e não o reescreve. Sem esse arquivo, ele grava a lista de
discordâncias com as colunas de decisão em branco.
