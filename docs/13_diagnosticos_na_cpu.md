# 13 Diagnósticos de geração e treinos curtos na CPU

Feito entre 7 e 9 de outubro de 2026, numa máquina sem GPU utilizável, para
entender os três problemas do [10](10_problemas_em_aberto.md). Nada aqui passou
pelo protocolo completo do [08](08_protocolo_de_avaliacao.md). As medidas usam
5 palavras acentuadas de validação, seus esqueletos e 4 palavras sem acento,
com 4 escritores do `iam_test` e 2 sementes, 8 imagens por texto. Servem para
escolher o que treinar na GPU, não como resultado final.

## Diagnósticos sem treino

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

## Letra e diacrítico separados no condicionamento

É a proposta que saiu disso. Só há uma olhada com 1 época, descrita abaixo. O CANINE recebe sempre o
esqueleto ("implantacao"), a entrada que o modelo do IAM conhece. O diacrítico
entra por um vetor novo por tipo de sinal, iniciado em zero e somado só na
posição da letra (`diffusionpen_mods/utils/acento_separado.py`, opção
`--acento_separado` do `train.py`).

Conferido antes de treinar. Com o vetor em zero, o modelo modificado pedindo
"provável" gera a mesma imagem que o original pedindo "provavel", com
diferença zero.

### Resultado na GPU, 16 épocas

Receita do `model_iam_pt_peso5` (base `iam_pt_alinhado`, 30% de originais,
peso 5) com `--acento_separado`, avaliada pelo protocolo do
[08](08_protocolo_de_avaliacao.md) na validação.

| modelo | marca na acentuada | no esqueleto | sem acento | diferença pareada | CER sem acento pedido | o mesmo, só sem marca |
|---|---|---|---|---|---|---|
| peso 5, 16 épocas | 57% | 34% | 32% | +23 | 0,312 | 0,273 |
| acento separado, 6 épocas | 62% | 21% | 14% | +41 | | |
| acento separado, 16 épocas | 61% | 13% | 9% | +48 | 0,285 | 0,276 |

Bootstrap por palavra, separado menos peso 5, com IC95.

| medida | diferença | IC95 |
|---|---|---|
| marca no esqueleto | −21 pp | −26 a −17 |
| marca sem acento | −23 pp | −30 a −16 |
| diferença pareada | +26 pp | +14 a +37 |
| marca na acentuada | +5 pp | −6 a +14 |
| CER sem acento pedido | −0,027 | −0,037 a −0,016 |
| CER sem acento pedido, só sem marca | −0,002 | −0,014 a +0,010 |

Posição (`medir_posicao_acento.py`), fração dos sinais na letra certa.

| sinal | peso 5 | acento separado |
|---|---|---|
| agudo | 14% | 38% |
| til | 52% | 68% |
| circunflexo | 4% | 11% |

Leitura.

- O vazamento cai para perto do ruído do detector e continua caindo entre 6 e
  16 épocas.
- O acento cai bem mais na letra certa.
- O modelo não desenha mais acentos quando pedem. Em 39% das acentuadas não
  sai marca nenhuma, e em 26% a marca cai em letra que não a leva.
- A letra não muda. A queda do CER geral é só menos marca solta lida como
  letra. Nas imagens sem marca os dois modelos empatam em 0,27, contra 0,19 do
  original. Como aqui a palavra sem acento recebe a entrada de texto do modelo
  original, a piora da letra não vem do caminho do texto.

Esse treino rodou antes de uma correção. A linha do "sem sinal" da tabela de
vetores era treinável e chegou a norma 0,50, contra 7 a 10 dos vetores do
CANINE e 4,4 dos vetores de sinal, um deslocamento de cerca de 6% em toda
posição. Hoje ela fica presa em zero. Falta repetir o treino com a correção.

As avaliações estão na máquina de treino, em `avaliacao_separado_val/`, e as
folhas em `saidas/comparacao_separado/`.

### Na CPU, antes disso

Uma olhada com 1 época e a loss original já mostrava a letra deformada como no
treino 4 e marcas sem relação com o texto. O acento ainda não tinha sido
aprendido nesse ponto.

### Professor nas palavras sem acento

Para a letra. Opção `--professor arquivo.pt` do `train.py`. Uma cópia congelada
do modelo de partida prevê o ruído para a mesma entrada, e nas amostras cujo
texto não tem diacrítico essa previsão vira o alvo da loss, no lugar do ruído
sorteado. As acentuadas continuam com o alvo de sempre. `--peso_professor`
mistura os dois alvos, com 1 só o professor e 0 só o ruído.

Com `--acento_separado` o aluno e o professor recebem a mesma entrada de texto
nessas amostras. Na receita do peso 5 elas são cerca de 85% do treino. Custa
uma passada a mais pela rede, sem gradiente. Só foi testado que roda, sem
resultado ainda.

### Como rodar

Para a receita do peso 5 com o acento separado e o professor.

```bash
bash scripts/aplicar_mods.sh
python DiffusionPen/train.py --dataset iam_acentuado --model_name diffusionpen \
    --sample_every 0 --save_path ./model_iam_pt_separado_professor \
    --dataset_folder ./iam_pt_alinhado --max_samples 0 --iam_originais 0.3 \
    --style_path ./DiffusionPen/style_models/iam_style_diffusionpen.pth \
    --stable_dif_path stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --lr 2e-05 --batch_size 32 --adamw_eps 1e-06 --clip_grad_norm 1.0 \
    --ema_beta 0.995 --ema_inicio 2000 --texto_max_len 40 --device cuda:0 \
    --num_workers 8 --save_every_steps 500 --abort_after 300 \
    --peso_acento 5 --peso_zona 1 --zona vazia --acento_separado \
    --professor ./DiffusionPen/diffusionpen_iam_model_path/models/ema_ckpt.pt \
    --epochs 16 --pretrained_path ./DiffusionPen/diffusionpen_iam_model_path/models
```

O que ainda não conhece a opção.

- `scripts/treinar.py` não tem a chave no JSON, por isso a chamada direta.
- `scripts/gerar_amostras.py` não abre esses checkpoints.
- `comum/diffusionpen.py` abre, então `scripts/avaliar_pt.py` e os scripts de
  `diagnostico/` funcionam sem mudança.

O que olhar é o CER nas imagens sem marca, que no original é 0,19 e nos
modelos com acento fica em 0,27, sem perder o vazamento baixo e a posição.

## Outras opções novas do train.py

Todas desligadas por padrão. Sem elas o caminho é o de antes.

| opção | o que faz |
|---|---|
| `--device cpu` | passa a funcionar |
| `--treinar_so texto` | treina só a atenção cruzada e a `text_lin` |
| `--cache_congelados arquivo.pt` | guarda a saída do VAE e do extrator de estilo por imagem. Na CPU o passo cai de 11 s para 3 s |
| `--acento_separado`, `--lr_acento_mult` | descritas acima |
| `--professor`, `--peso_professor` | descritas acima |

Os roteiros dos treinos curtos são `diagnostico/treino_ingredientes.sh`,
`treino_texto_vs_tudo.sh` e `treino_acento_separado.sh`. A avaliação pequena é
`diagnostico/avaliar_mini.py`.

## Limites

- Poucas imagens, leitura das folhas no olho, sem o leitor CTC.
- Em `casas_trocadas` e `reforco_contraste` entraram por descuido três palavras
  do treino ("pântano", "campo", "mundo") e uma do teste ("você"). Os outros
  diagnósticos usam só palavras de validação, conferidas contra as bases.

## Trabalhos relacionados conferidos

- Diffusion-Based Ukrainian Handwritten Text Generation with Cross-Domain Style
  Transfer, arXiv 2605.27487. Retreina o DiffusionPen em ucraniano.
- Open-Set Personalized Handwriting Generation with DiffusionPen, relatório
  técnico 2026-43 da University at Buffalo. Registra que a atenção cruzada do
  código público tem `mask = None`, que em palavras curtas o preenchimento
  vence os caracteres reais, e que um fine-tune curto com a máscara corrige.
