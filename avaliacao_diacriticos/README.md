# Métrica de diacríticos

Mede se um gerador de manuscrito desenha os diacríticos do português. A
métrica final responde, para cada palavra pedida com acento, três perguntas.

| pergunta | como | situação |
|---|---|---|
| 1. o sinal pedido está lá? | `sinais.py`: marca solta em cima para acento, tinta abaixo do c para cedilha | validada pela anotação humana |
| 2. está na letra certa, e só nela? | `sinais.py`: a mesma marca atribuída a uma vogal pelo alinhador | validada pela anotação humana |
| 3. é o sinal certo (agudo, til, circunflexo)? | um leitor que leia a palavra com acento, no eixo 2 | a fazer |

As seções a partir de "Como o eixo 1 funciona" registram as tentativas
anteriores, inclusive o eixo 1 original, que comparava a imagem acentuada com
o esqueleto e não passou na anotação.

## Métrica final, perguntas 1 e 2

`sinais.py` junta as duas medidas que passaram na anotação humana do modelo de
40 épocas (`anotacao_40ep.md`).

- Acento em cima: manchas de tinta soltas, inteiras acima do corpo da
  palavra, como no `scripts/medir_marcas.py` do colega.
- Cedilha: tinta abaixo da linha de base na coluna de cada c, como na sexta
  tentativa da seção "Cedilha".

Na posição, cada marca vai para a letra cuja coluna, pelo alinhador, contém o
centro dela. Duas regras foram acrescentadas ao detector do colega:

- alta: aceita uma marca mais alta que a altura-x quando ela está inteira
  acima do corpo. O detector original a descarta como se fosse haste.
- vogal: se a marca cai numa consoante, ela vai para a vogal cuja coluna mais
  se sobrepõe a ela, porque acento em português só cai em vogal.

O `scripts/medir_marcas.py` não foi alterado. Ele continua sendo a régua das
comparações de treino.

Cada palavra recebe uma categoria.

| categoria | quando |
|---|---|
| correto | todos os sinais pedidos, na letra certa, e nada a mais |
| a mais | os sinais pedidos estão certos, mas há marca em outra letra |
| letra errada | um acento pedido falta na letra dele, e há marca em outra |
| ausente | falta um sinal pedido, sem marca em outro lugar |
| limpa ou marca indevida | palavras sem nada pedido |

### Validação

Contra o gabarito humano das 180 imagens sem i e j do modelo de 40 épocas.
Detalhes em `resultados/presenca_40ep.txt`, `resultados/posicao_v2_40ep.txt`
e `resultados/categorias_40ep.txt`.

Presença.

| pergunta | concordam | precisão | recall | kappa |
|---|---|---|---|---|
| há marca em cima | 164 de 180 (91%) | 0,91 | 0,88 | 0,82 |
| há cedilha | 204 de 212 (96%) | 0,97 | 0,90 | 0,91 |
| todos os sinais pedidos presentes | 119 de 135 (88%) | 0,97 | 0,90 | 0,15 |

O kappa da última linha é baixo porque as pessoas veem o sinal pedido em 130
das 135 palavras, e quase não há casos negativos. Precisão e recall dizem mais.
Quando o detector diz que o sinal está lá, acerta 97% das vezes, e ele acha
90% dos sinais que as pessoas veem.

Posição, por vogal. As palavras foram divididas em duas metades. As regras
foram decididas olhando a metade A, e a metade B não foi usada em nenhuma
escolha.

| detector | kappa por vogal, metade A | kappa por vogal, metade B |
|---|---|---|
| original | 0,83 | 0,72 |
| com as regras alta e vogal | 0,87 | 0,75 |

Duas outras correções foram testadas e descartadas. Uma área mínima para a
marca melhorou a metade A e piorou a B. A tinta acima de cada vogal, desenho
igual ao da cedilha, achou acentos que encostam na letra, mas inventou marca
nas palavras sem acento, porque a haste de um l ou t vizinho invade a coluna
da vogal.

Categoria por palavra, nas 135 palavras pedidas com acento.

| categoria | pessoas | detector |
|---|---|---|
| correto | 87% | 75% |
| a mais | 6% | 6% |
| letra errada | 4% | 9% |
| ausente | 4% | 10% |

As duas concordam na categoria em 104 das 135 palavras (77%, kappa 0,33). O
erro principal é o acento que encosta na letra: as pessoas o veem, o detector
não acha mancha solta e chama a palavra de ausente (11 casos) ou de letra
errada (4). Por isso a métrica subestima o "correto" em uns 12 pontos. Para
comparar modelos entre si ela serve, desde que essa falha atinja os modelos de
forma parecida. Para dizer quantas palavras um modelo acerta, o número vem com
essa ressalva.

### Como usar

```bash
python avaliacao_diacriticos/avaliar_sinais.py --saida avaliacao_diacriticos/resultados/sinais \
    --modelo <rotulo>=<pasta das imagens>:<pasta com o paineis.tsv>
```

A entrada é uma avaliação do `scripts/avaliar_pt.py`. Sai uma linha por
imagem em `sinais_<rotulo>.tsv` e uma linha por modelo em
`resumo_sinais.tsv`. Leva cerca de um minuto por modelo na CPU.

### Por classe de sinal

`por_classe.py` separa o resultado pelo que a palavra pede: agudo, til,
circunflexo, cedilha, e cedilha com til. Sai em `resultados/sinais/por_classe.md`
para cada modelo. No protocolo, circunflexo e cedilha com til têm duas palavras
cada, e os números delas dependem muito dessas duas palavras.

No modelo de 40 épocas, contra o gabarito humano (`resultados/categorias_40ep.txt`):

| classe | imagens anotadas | correto pelas pessoas | correto pela métrica |
|---|---|---|---|
| cedilha | 55 | 96% | 89% |
| cedilha e til | 8 | 100% | 88% |
| til | 20 | 90% | 65% |
| agudo | 36 | 78% | 64% |
| circunflexo | 16 | 62% | 56% |

As pessoas ordenam as classes assim: cedilha melhor, depois til, agudo e
circunflexo. A métrica mantém a cedilha no topo e o circunflexo no fundo, mas
subestima o til em 25 pontos e o põe junto do agudo. Nas palavras com til em
que métrica e pessoas discordam, o til encosta na letra e não é visto, cobre
duas vogais (o "ão") e cai na vizinha, ou sai partido em dois pedaços. Por
isso a comparação entre classes pela métrica deve ser lida com cuidado, e a
diferença entre til e agudo só vale pelo gabarito humano.

### Limitações

- Palavras com i ou j ficam de fora, porque o pingo é uma marca solta em cima.
- O tipo do sinal não é julgado. Um til no lugar de um agudo conta como
  correto.
- A cedilha conta só como presença. Na anotação, as pessoas às vezes marcaram
  a letra vizinha do c, e o gabarito da letra dela não é confiável.
- A validação humana é de um modelo só, o de 40 épocas.

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

## Comparação com o detector de marcas soltas

O detector de `scripts/medir_marcas.py` olha uma imagem só e procura manchas de
tinta soltas acima e abaixo do corpo das letras. O alinhador de
`acentos_sinteticos/alinhamento.py` diz a que letra cada mancha pertence. Ele
foi comparado com o eixo 1 em dois conjuntos.

O primeiro é a base `iam_pt_alinhado`, com 5.188 pares em que o acento foi
desenhado por programa e a letra dele é conhecida. Os limiares do eixo 1 são o
p95 do controle negativo e não foram ajustados nesta base. A tabela dá a
fração dos sinais percebidos na letra certa.

| sinal | eixo 1 atual | eixo 1 com alinhador | marcas soltas | marca nova no par | n |
|---|---|---|---|---|---|
| agudo | 20,5% | 83,4% | 87,9% | 78,1% | 2.909 |
| til | 98,8% | 99,3% | 90,9% | 90,7% | 1.121 |
| circunflexo | 99,1% | 99,5% | 91,2% | 91,0% | 580 |
| cedilha | 15,9% | 26,9% | 7,8% | 7,6% | 1.093 |

O eixo 1 com alinhador está em `e1_alinhador.py`. Ele usa as colunas dadas
pelo alinhador no lugar das fatias iguais. A marca nova no par está em
`marcas_do_par.py` e conta as marcas soltas que a imagem acentuada tem e a
sem acento não tem, na mesma letra.

Com fatias iguais o eixo 1 acusa a letra vizinha em 64% dos tis e 68% dos
circunflexos. Com o alinhador isso cai para 4%, e a maior medida da palavra
fica na letra do sinal em 97 a 99% dos casos, contra 61 a 73%.

A cedilha não é medida por nenhum dos métodos. No eixo 1 a medida fica acima
de zero em 99% das cedilhas e abaixo do limiar, que vem alto do controle
negativo. No detector de marcas soltas a cedilha desenhada grudada no c não é
mancha solta.

O segundo conjunto são 200 imagens geradas pelo `model_iam_pt_peso5` de 16
épocas e anotadas por uma pessoa, com 80 pedidas com acento, 80 esqueletos e
40 palavras sem acento, todas do split de validação e sem i nem j. A pessoa
viu acento em 43 das 80 pedidas com acento e em 32 das 120 pedidas sem.

| nas 80 pedidas com acento | percebe acento visto | alarme falso | kappa |
|---|---|---|---|
| marcas soltas | 38 de 43 | 6 de 37 | 0,72 |
| marca nova no par | 31 de 43 | 6 de 37 | 0,55 |
| eixo 1 com alinhador | 23 de 43 | 6 de 37 | 0,36 |

Nas 120 pedidas sem acento o detector de marcas soltas tem kappa 0,88 com a
pessoa. Na pergunta sobre a letra certa os kappas são 0,44 para o eixo 1 com
alinhador, 0,29 para as marcas soltas e 0,22 para a marca nova no par. Baixar
o limiar do eixo 1 até ele dar os mesmos 6 alarmes falsos leva a 26 ou 27
acentos percebidos, ainda abaixo de 38.

As 7 imagens que a marca nova no par deixa passar e as marcas soltas percebem
têm acento visto pela pessoa também no esqueleto. Em 18 das 43 imagens com
acento visto o esqueleto do mesmo escritor também saiu com acento. As duas
medidas respondem a perguntas diferentes. As marcas soltas dizem se há acento
na imagem. A marca nova no par diz se o acento apareceu por causa do texto.

## Vazamento

Vazamento é marca em palavra que não pede marca. O eixo 1 não o mede, porque
compara duas imagens e a marca presente nas duas dá diferença zero.

A medida de `vazamento.py` olha uma imagem e conta a tinta acima das vogais a,
e, o, u e abaixo do c. Ela não funciona. Com 5% de falso alarme detecta 10%
das imagens acentuadas da base, com AUC 0,74.

O detector de marcas soltas funciona em palavras sem i nem j, com falso alarme
de 2,5 a 4,6% e detecção de 74,5%. Em palavras quaisquer o falso alarme sobe
para 31 a 38% por causa do pingo. Descontar a primeira marca sobre cada i e j
reduz o falso alarme para 10 a 12% e faz a detecção do agudo cair de 94,5%
para 75,4%, porque o acento sobre o i ocupa o lugar do pingo.

Os números completos estão em `resultados/comparacao_detectores.txt`. A
planilha anotada é `resultados/anotacao_peso5.csv` e as folhas são
`figuras/anotacao_peso5_folha1.png` a `folha5.png`. Para refazer

```bash
python avaliacao_diacriticos/experimentos/limiares_alinhador.py
python avaliacao_diacriticos/experimentos/medir_base_alinhada.py
python avaliacao_diacriticos/experimentos/comparar_detectores.py
python avaliacao_diacriticos/anotacao.py kappa
```

## Cedilha

A sexta tentativa, a tinta abaixo do c numa imagem só, é a medida da
cedilha. Ela passou na base e, nas imagens geradas pelo modelo de 40 épocas,
concorda com a anotação humana, com kappa 0,91, precisão 0,97 e recall 0,92
(ver `anotacao_40ep.md`).

Foram oito tentativas, mais uma varredura de limiares à mão. As porcentagens
são das 1.093 cedilhas da base `iam_pt_alinhado` que cada método percebe.

| método | cedilhas percebidas |
|---|---|
| eixo 1 com fatias iguais, limiar do modelo do IAM | 15,9% |
| eixo 1 com alinhador, limiar do modelo do IAM | 26,9% |
| detector de marcas soltas | 7,8% |
| marca nova no par | 7,6% |
| diferença pareada com limiar do próprio modelo de 40 épocas | 28,3% |
| tinta abaixo do c numa imagem só, limiar do modelo de 40 épocas | 95,0% |
| altura da tinta abaixo do c numa imagem só, limiar do modelo de 40 épocas | 93,9% |
| altura abaixo do c com linha de base local, limiar do modelo de 40 épocas | 93,6% |

O detector de marcas soltas não vê a cedilha presa ao c, porque ela faz parte
da mancha da letra. O eixo 1 vê tinta a mais embaixo do c em 99% das
cedilhas, com medida típica de 0,28, mas o limiar fica em torno de 0,40.

A última tentativa, em `experimentos/cedilha.py`, tirou o limiar dos c sem
cedilha das imagens do modelo de 40 épocas, como o c de percussão, onde a
acentuada e o esqueleto diferem só pelo ruído do gerador. O limiar saiu 0,40,
quase o mesmo 0,41 do modelo do IAM. Nas cedilhas pedidas ao modelo de 40
épocas, 27,7% passam dele.

Nas oito maiores medidas desses c sem cedilha, duas são a palavra inteira
redesenhada entre a acentuada e o esqueleto, e seis são uma cedilha de verdade
de outra letra caindo na coluna do primeiro c. As colunas vêm do esqueleto, e
na imagem acentuada as letras estão em outro lugar. Na base isso não acontece
porque as duas imagens só diferem no acento. Os números estão em
`resultados/cedilha_40ep.json`.

A sexta tentativa, em `experimentos/cedilha_abaixo.py`, não usa o esqueleto.
Como o detector do colega, ela acha o corpo das letras com
`acentos_sinteticos/geometria.py`, e as colunas de cada letra vêm do alinhador
lendo o esqueleto na própria imagem acentuada. A medida de cada c é a tinta
abaixo da linha de base mais 0,15 altura-x, a mesma folga do
`scripts/medir_marcas.py`, dividida pela tinta do corpo do c. O negativo e o
limiar são os da tentativa anterior: o p95 dos 280 c sem cedilha das
acentuadas do modelo de 40 épocas. O limiar saiu 0,0003, porque mais de 95%
desses c não têm tinta nenhuma ali. Na prática a regra é haver tinta abaixo
do c.

Com esse limiar, 95,0% das 1.093 cedilhas da base passam, todas alinhadas.
O mesmo c na imagem sem acento do par passa em 1,6%. Nas cedilhas pedidas ao
modelo de 40 épocas, 89,4% passam. Na anotação humana do mesmo modelo, ela
pega 57 das 62 cedilhas que as pessoas veem e acusa 2 que elas não veem.

As quatro folhas `resultados/cedilha_abaixo_folha_{A,B,C,D}.png` mostram, no
modelo de 40 épocas, 16 acertos sorteados (A), 16 cedilhas perdidas sorteadas
(B), todos os 14 c sem cedilha que passaram do limiar (C) e 16 c sem cedilha
que ficaram abaixo dele (D). Os painéis são numerados e listados no JSON.

Nos acertos, 15 dos 16 têm cedilha visível embaixo do c. A1 não tem.

Nos 14 falsos alarmes, quatro são cedilha de verdade desenhada no primeiro c
de recomeça em vez do segundo (C7, C11, C12, C13), e a medida acertou. Oito
são c sem cedilha: em C1, C2 e C3 só a borda de baixo do c passa do corte,
com medida até 0,03; em C6, C8, C9, C10 e C14 a volta de baixo do c desce
bem abaixo da base, com medida de 0,06 a 0,41, acima de muitas cedilhas
verdadeiras. C4 e C5 são letras borradas que não dá para julgar.

Nas perdidas, as cedilhas que o modelo desenhou são curtas e ficam acima do
corte (B5, B6, B8, B9, B11, B13, B14), ou a coluna do alinhador caiu fora do
c (B4, B7, B10, B15). Em B12 e B16 não há cedilha embaixo do c.

Os 280 c do negativo vêm de sete palavras, nenhuma com g, p, j ou q ao lado
do c. A rabeira dessas letras entrando na coluna do c não foi testada nas
imagens geradas. Os números estão em `resultados/cedilha_abaixo_40ep.json`.

A sétima tentativa, `experimentos/cedilha_abaixo.py --medida altura`, troca a
contagem de pixels pela altura. A medida é quantas fileiras seguidas, a partir
do corte para baixo, têm tinta na coluna do c, divididas pela altura-x. A
ideia é que a cedilha é estreita e funda, e o c um pouco mais baixo que a
palavra é largo e raso. Fileiras seguidas para que um pixel solto lá embaixo
não conte. Corte, negativo e regra do limiar ficaram iguais.

O limiar saiu 0,0018, de novo quase zero, porque os mesmos 95% dos c sem
cedilha não têm tinta abaixo do corte. Com limiar zero as duas medidas
decidem a mesma coisa, haver tinta abaixo do corte, e os 14 c sem cedilha que
passam são os mesmos da sexta tentativa. Na base, 93,9% das cedilhas passam e
o mesmo c no par passa em 1,6%. Nas cedilhas pedidas ao modelo de 40 épocas,
84,4% passam, contra 89,4% da sexta.

As folhas são `resultados/cedilha_abaixo_altura_folha_{A,B,C,D}.png`. A altura
mudou a ordem dos falsos alarmes. Os c sem cedilha ficaram rasos, de 0,036 a
0,136 altura-x (C1 a C7 e C10; o C10 é o c inteiro mais baixo de caçadora, que
na sexta media 0,412). As cedilhas de verdade no primeiro c de recomeça
ficaram fundas, de 0,33 a 0,44 (C11 a C14). A mediana das cedilhas da base é
0,35. Essa separação não muda o resultado porque a regra do p95 dá limiar
zero com qualquer medida neste negativo.

As perdas a mais vêm das fileiras seguidas. Em B5 e B7 a coluna está deslocada
e só pega a ponta curva da cedilha, que não encosta no corte; na sexta elas
eram acerto. Em B10 e B16 a cauda cruza a linha com uma fileira vazia logo
abaixo. Pelo mesmo motivo a cedilha solta do c, que a anotação humana conta
como cedilha, zera a medida de altura e a da oitava tentativa. A sexta, que
conta toda a tinta abaixo do corte, não tem esse problema e é a medida
adotada. Os números estão em `resultados/cedilha_abaixo_altura_40ep.json`.

Também testamos limiares fixos à mão, escolhidos numa grade antes de rodar e
fora da regra do p95. Os c sem cedilha que passam são nomeados pelos painéis
da folha C de cada medida.

| medida e limiar | base | mesmo c no par | modelo, pedidas | c sem cedilha que passam |
|---|---|---|---|---|
| altura 0,05 | 93,8% | 1,0% | 83,8% | C6 a C14 |
| altura 0,10 | 93,1% | 0,4% | 81,5% | C8 a C14 |
| altura 0,15 | 90,5% | 0,1% | 78,5% | C11 a C14 |
| altura 0,20 | 84,7% | 0,0% | 74,6% | C11 a C14 |
| altura 0,25 | 76,6% | 0,0% | 68,8% | C11 a C14 |
| altura 0,30 | 65,7% | 0,0% | 60,8% | C11 a C14 |
| tinta 0,05 | 93,2% | 0,9% | 82,9% | C5 a C14 |
| tinta 0,10 | 78,6% | 0,4% | 68,3% | C9 a C14 |
| tinta 0,15 | 58,8% | 0,3% | 53,5% | C11 a C14 |
| tinta 0,20 | 39,2% | 0,1% | 37,1% | C12 a C14 |

Na altura, a partir de 0,15 só passam C11 a C14, as cedilhas de verdade no c
errado de recomeça, e nenhum c normal. Na tinta, o c inteiro mais baixo de
caçadora (C14, 0,412) passa em qualquer limiar que ainda perceba metade das
cedilhas.

O custo aparece na folha `resultados/cedilha_abaixo_altura_folha_E.png`, com as
31 cedilhas pedidas ao modelo que o limiar 0,15 perde. Em 29 delas há cedilha
visível embaixo do c, só que curta; E23 é apagada demais para julgar e em E28
a coluna caiu sobre o a. As cedilhas curtas do modelo ficam na mesma faixa dos
c normais, de 0,04 a 0,15, e nenhum limiar separa as duas coisas. Subir o
limiar de zero para 0,15 troca 8 c normais acusados por cerca de 29 cedilhas
verdadeiras perdidas. Qual troca serve depende do uso da medida. A anotação
humana só foi comparada com o limiar zero da sexta tentativa, e os limiares
fixos não foram conferidos contra ela.

A oitava tentativa, `experimentos/cedilha_abaixo.py --medida base_local`, mede
a altura a partir de uma linha de base inclinada em vez da horizontal. A
linha é uma reta, por mínimos quadrados, pela fileira mais baixa com tinta de
cada letra sem descendente (ficam de fora g, j, p, q, y, f, z e os próprios
c). O corte de cada c é essa reta no meio da coluna dele mais 0,15 altura-x.
A ideia era que um c inteiro mais baixo que as vizinhas, como o de caçadora
que media 0,412 na sexta, deixasse de contar.

Esse c de caçadora deixou de passar, mas o resultado piorou. O limiar saiu
0,053, 93,6% das cedilhas da base passam, o mesmo c no par passa em 2,5% (era
1,6%) e as cedilhas pedidas ao modelo passam em 82,7% (eram 84,4%). Na folha
`resultados/cedilha_abaixo_base_local_folha_C.png` os oito c sem cedilha que
acusam (C1 a C7 e C9) são todos a primeira letra da palavra, em confusões,
caçadora, caçadoras e conexões. C11 a C14 são de novo a cedilha no c errado de
recomeça, e C8 e C10 são letras borradas.

O JSON de cada medida guarda quanto o fundo dos c sem cedilha desce abaixo da
linha de base usada. Com a linha horizontal, a mediana é 0 px e o p90 é 2 px,
tanto no c que abre a palavra quanto no c do meio. Com a reta, o c do meio
fica em -0,4 px de mediana e 3,0 px de p90, e o c que abre a palavra fica em
1,2 px e 4,1 px. A folga é de uns 3 px. A reta sai das letras do meio e, no
começo da palavra, onde é extrapolada, passa acima do fundo do c. As palavras
deste modelo quase não são inclinadas, então a reta não tinha o que corrigir
e só acrescentou esse erro na ponta. Os números estão em
`resultados/cedilha_abaixo_base_local_40ep.json`.

## Anotação do modelo de 40 épocas

Duas pessoas anotaram 212 imagens do modelo de 40 épocas. O processo, o que é
o kappa, a precisão e o recall, o desempate e os resultados estão em
`anotacao_40ep.md`.

Um cuidado que saiu dela. A coluna `marca` do `paineis.tsv` do
`scripts/avaliar_pt.py` vale 1 com qualquer mancha solta, em cima ou embaixo
do corpo. Numa palavra com ç, a mancha embaixo é quase sempre a cedilha solta,
então essa coluna não serve como medida de acento a mais. Nas palavras só com
cedilha do modelo de 40 épocas ela acusa 15 de 55 imagens, e as pessoas viram
acento a mais em 1. O lado do treino passou a usar só a marca em cima
(`diagnostico/marcas_so_cedilha.py`).

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
