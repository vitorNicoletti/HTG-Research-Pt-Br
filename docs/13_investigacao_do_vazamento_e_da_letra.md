# 13 Investigação do vazamento e da letra

Testes de geração sem treino e treinos curtos, feitos entre 7 e 9 de outubro
de 2026 numa máquina sem GPU utilizável, para entender os problemas do
[10](10_problemas_em_aberto.md). Não tem relação com os testes de hardware da
pasta `diagnostico/`. O método que saiu daqui, com os resultados na GPU, está
no [14](14_acento_separado_e_professor.md). Nada aqui passou pelo protocolo
completo do [08](08_protocolo_de_avaliacao.md). As medidas usam
5 palavras acentuadas de validação, seus esqueletos e 4 palavras sem acento,
com 4 escritores do `iam_test` e 2 sementes, 8 imagens por texto. Servem para
escolher o que treinar na GPU, não como resultado final.

## Testes de geração sem treino

Todos usam o `model_iam_pt_peso5` de 16 épocas e rodam na CPU.

### O sinal do diacrítico chega ao UNet

`diagnostico/diag_texto_por_posicao.py`. Distância relativa entre os vetores
de texto depois da `text_lin`, posição a posição, em 25 palavras.

| posição | ao tirar o acento | ao trocar a letra por outra |
|---|---|---|
| a letra acentuada | 1,19 | 1,12 |
| letras vizinhas | 0,40 | 0,31 |
| outras letras | 0,33 | 0,27 |
| preenchimento | 0,81 | 0,53 |

Na posição da letra, "á" difere de "a" tanto quanto "o" difere de "a". O
cosseno de 0,935 do [07](07_experimentos_com_acentos.md) vinha da média das
posições e escondia isso. O sinal não chega fraco. E o acento muda também os
vetores das outras letras e do preenchimento, porque o CANINE é contextual.

### O texto quase não decide a marca

`diagnostico/diag_casas_trocadas.py`. Na maioria das colunas a imagem sai
igual com "módulo" e com "modulo". Quem decide a marca é o par escritor e
ruído. Nos poucos casos em que o texto muda a imagem, a marca pareceu seguir
as posições de preenchimento nas palavras com agudo e circunflexo, e as
letras em "fogão". É um indício em poucas imagens.

### A marca segue o texto dos passos finais

`diagnostico/diag_quando_e_reforco.py`, 50 passos de geração.

| texto usado | imagens com marca |
|---|---|
| acentuado o tempo todo | 75% |
| sem acento o tempo todo | 35% |
| acentuado nos 10 primeiros passos, depois sem | 38% |
| sem acento nos 10 primeiros, depois acentuado | 75% |
| acentuado nos 25 primeiros, depois sem | 57% |
| sem acento nos 25 primeiros, depois acentuado | 70% |

### Reforço do texto na geração

Mesmo script. Em cada passo o modelo prevê com o texto pedido e com o texto
de contraste, e usa `e_contraste + w * (e_pedido - e_contraste)`.

| pedido | w = 1 | w = 3 | w = 6 |
|---|---|---|---|
| palavra acentuada, marca presente | 75% | 80% | 80% |
| esqueleto, marca presente | 35% | 15% | 20% |

Com w = 3 o vazamento cai para menos da metade nas palavras em que o modelo
reage ao texto. Com w = 6 a letra deforma. Com contraste inventado
(`diagnostico/diag_reforco_contraste.py`, por exemplo "tarde" contra "tárdé")
funciona em algumas palavras, não faz nada em outras e estraga palavras que já
saíam limpas. Não é método geral, é um termômetro de onde o modelo lê o texto.

## Treinos curtos

Todos partem dos pesos do IAM, com 16.670 amostras, 2 épocas (1.042 passos),
lr 2e-5 e lote 32. O IAM da máquina estava extraído pela metade, então as
bases usam só os 168 escritores com todas as imagens
(`scripts/base_subconjunto.py`). Os valores não são comparáveis aos do
[09](09_resultados_consolidados.md), só entre si. Até o braço 3 não há acento
nos dados, e as marcas acusadas são pedaços de letra quebrada.

| treino | o que entra | marca na acentuada | no esqueleto | sem acento |
|---|---|---|---|---|
| IAM original | | 0% | 5% | 0% |
| 1 | só palavras reais | 0% | 2% | 0% |
| 2 | mais geradas sem acento | 2% | 5% | 0% |
| 3 | mais os pares | 10% | 12% | 9% |
| 4 | mais as acentuadas, loss original | 22% | 18% | 19% |
| 5 | o 4 com peso 5 | 50% | 42% | 25% |
| A | o 4 com a imagem acentuada e o rótulo sem acento | 25% | 12% | 9% |
| B | o 4 com a imagem sem sinal e o rótulo acentuado | 12% | 8% | 0% |
| só leitura do texto | o 5 treinando só `attn2` e `text_lin` | 28% | 25% | 16% |

Leitura das folhas em `diagnostico/resultados/ingredientes/`,
`traco_vs_rotulo/` e `mini/`.

- Só palavras reais não muda a letra, como no controle da fase 7c.
- As imagens geradas borram um pouco o traço.
- O salto de piora vem quando as acentuadas entram, já com a loss original. O
  peso 5 não é a causa da piora da letra, só aumenta as marcas.
- Nem o traço sozinho (A) nem o rótulo sozinho (B) reproduzem o salto. A letra
  piora quando os dois aparecem juntos, isto é, quando o modelo aprende a
  ligação entre o caractere acentuado e a marca.
- Treinar só a leitura do texto, congelando 95% do UNet, estraga a letra do
  mesmo jeito e aprende o acento mais devagar.

## O que saiu disso

Duas ideias de treino, descritas no [14](14_acento_separado_e_professor.md).

- Se o acento muda os vetores de texto da palavra inteira, mandar ao CANINE a
  palavra sem acento e somar o diacrítico só na posição da letra.
- Se a letra piora quando o modelo aprende a desenhar marcas, por qualquer
  caminho, cobrar nas palavras sem acento a resposta do modelo original.

## Opções do train.py usadas aqui

Todas desligadas por padrão. Sem elas o caminho é o de antes.

| opção | o que faz |
|---|---|
| `--device cpu` | passa a funcionar |
| `--treinar_so texto` | treina só a atenção cruzada e a `text_lin` |
| `--cache_congelados arquivo.pt` | guarda a saída do VAE e do extrator de estilo por imagem. Na CPU o passo cai de 11 s para 3 s |

Os roteiros dos treinos curtos são `diagnostico/treino_ingredientes.sh`,
`treino_texto_vs_tudo.sh` e `treino_acento_separado.sh`. A avaliação pequena é
`diagnostico/avaliar_mini.py`. As bases reduzidas saem de
`scripts/base_subconjunto.py` e `scripts/base_troca_rotulo.py`.

## Limites

- Poucas imagens, leitura das folhas no olho, sem o leitor CTC.
- Em `casas_trocadas` e `reforco_contraste` entraram por descuido três palavras
  do treino ("pântano", "campo", "mundo") e uma do teste ("você"). Os outros
  testes usam só palavras de validação, conferidas contra as bases.
