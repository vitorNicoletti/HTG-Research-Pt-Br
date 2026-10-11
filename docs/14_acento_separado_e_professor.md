# 14 Acento separado e professor

Duas mudanças no treino que saíram da investigação do
[13](13_investigacao_do_vazamento_e_da_letra.md), avaliadas na GPU pelo
protocolo do [08](08_protocolo_de_avaliacao.md), na validação. A receita de
partida é a do `model_iam_pt_peso5` do [07](07_experimentos_com_acentos.md),
com a base `iam_pt_alinhado`, 30% de palavras originais do IAM, peso 5 no
acento e 16 épocas.

## Acento separado

No DiffusionPen o texto passa inteiro pelo CANINE, que é contextual. Trocar "a"
por "á" muda o vetor daquela posição e também o das outras letras e o do
preenchimento. Uma palavra acentuada chega ao UNet como uma entrada diferente
em todas as 40 posições.

Com `--acento_separado` o CANINE recebe sempre o esqueleto, a entrada que o
modelo do IAM conhece. O diacrítico entra por um vetor por tipo de sinal,
iniciado em zero e somado só na posição da letra.

```
palavra      i m p l a n t a ç ã o
CANINE lê    i m p l a n t a c a o
sinais       0 0 0 0 0 0 0 0 5 4 0     0 nada, 1 agudo, 2 grave,
                                       3 circunflexo, 4 til, 5 cedilha
```

Código em `diffusionpen_mods/utils/acento_separado.py`. A tabela de vetores é a
única peça nova e treina com a lr multiplicada por `--lr_acento_mult`, 30 por
padrão.

Conferido antes de treinar. Com os vetores em zero, o modelo modificado pedindo
"provável" gera a mesma imagem que o original pedindo "provavel", com diferença
zero.

Uma correção no meio do caminho. Na primeira versão a linha do "sem sinal" da
tabela era treinável. O primeiro treino na GPU rodou assim, e ela chegou a
norma 0,50, contra 7 a 10 dos vetores do CANINE e 4,4 dos vetores de sinal, um
deslocamento de cerca de 6% em toda posição. Hoje ela fica presa em zero.

## Professor

A letra piora quando dados acentuados entram no treino
([13](13_investigacao_do_vazamento_e_da_letra.md)). Com `--professor
arquivo.pt` uma cópia congelada do modelo de partida prevê o ruído para a mesma
entrada, e nas amostras cujo texto não tem diacrítico essa previsão vira o alvo
da loss, no lugar do ruído sorteado. As acentuadas continuam com o alvo de
sempre. `--peso_professor` mistura os dois alvos, com 1 só o professor e 0 só o
ruído.

Com o acento separado, aluno e professor recebem a mesma entrada de texto
nessas amostras. Na receita acima elas são 85% do treino. Custa uma passada a
mais pela rede, sem gradiente.

É destilação a partir do modelo original, na linha da destilação generativa de
Masip e outros e do Learning without Forgetting de Li e Hoiem. Não é o método
deles. Lá o professor gera as imagens na hora. Aqui elas são as do treino, em
parte palavras reais do IAM e em parte palavras que o próprio modelo original
escreveu quando a base foi montada.

## Resultados

Três treinos de 16 épocas. O peso 5 é o do [07](07_experimentos_com_acentos.md).

| modelo | marca na acentuada | no esqueleto | sem acento | diferença pareada | CER sem acento pedido | o mesmo, só sem marca |
|---|---|---|---|---|---|---|
| IAM original | 6% | 5% | 5% | +1 | 0,195 | 0,193 |
| peso 5 | 57% | 34% | 32% | +23 | 0,312 | 0,273 |
| acento separado | 61% | 13% | 9% | +48 | 0,285 | 0,276 |
| acento separado e professor | 71% | 5% | 8% | +66 | 0,190 | 0,187 |

Com 6 épocas o acento separado dava 62%, 21%, 14% e +41, e com o professor
81%, 6%, 6% e +75.

Bootstrap por palavra, com IC95.

| medida | separado menos peso 5 | professor menos separado |
|---|---|---|
| marca na acentuada | +5 pp, de −6 a +14 | +10 pp, de +6 a +14 |
| marca no esqueleto | −21 pp, de −26 a −17 | −8 pp, de −10 a −6 |
| marca sem acento | −23 pp, de −30 a −16 | −2 pp, de −4 a +2 |
| diferença pareada | +26 pp, de +14 a +37 | +18 pp, de +14 a +21 |
| CER sem acento pedido | −0,027, de −0,037 a −0,016 | −0,096, de −0,109 a −0,083 |
| o mesmo, só sem marca | −0,002, de −0,014 a +0,010 | −0,089, de −0,102 a −0,076 |

Posição, nas 1.200 imagens acentuadas (`scripts/medir_posicao_acento.py`).

| | peso 5 | acento separado | com professor |
|---|---|---|---|
| agudo na letra certa | 14% | 38% | 53% |
| til na letra certa | 52% | 68% | 66% |
| circunflexo na letra certa | 4% | 11% | 24% |
| todos os sinais na letra certa | | 31% | 38% |
| marca só em letra errada | | 26% | 30% |
| nenhuma marca | | 39% | 29% |
| alguma marca em letra que não a leva | | 38% | 46% |

No log do treino com professor, a distância entre aluno e professor nas
amostras sem acento foi de 0,0094 na primeira época para 0,0060 na última.

### Leitura

- O acento separado derruba o vazamento e melhora a posição. Não muda a letra.
  A queda do CER geral nesse passo é só menos marca solta lida como letra, e
  nas imagens sem marca ele empata com o peso 5. Como a palavra sem acento
  recebe a entrada de texto do modelo original, a piora da letra não vem do
  caminho do texto.
- O professor traz a letra das palavras sem acento de volta ao nível do modelo
  original e leva o vazamento ao ruído do detector, que no original é 5%. O
  acento aparece mais vezes e o agudo acerta mais a letra.

### O que sobra

Está dentro das palavras acentuadas.

- Em 29% o detector não vê marca, e em 30% ela cai em letra errada. A falta
  de marca é quase toda das palavras só com cedilha, que o detector não
  enxerga (tabela abaixo). Nos outros sinais sai marca em 86% dos pedidos.
- Em 46% há marca em alguma letra que não a leva, mais do que antes.
- O CER das acentuadas contra o esqueleto é 0,43, contra 0,17 do esqueleto da
  mesma palavra. Parte são marcas lidas como letras, mas as folhas mostram
  deformação. O professor não atua nessas amostras.
- O circunflexo segue fraco. A base tem 580 amostras com ele, contra 2.903 com
  agudo, e só 2 com grave. A avaliação tem só duas palavras com ele, "pôr" e
  "relâmpagos", então os números do circunflexo valem pouco.

Marca detectada por sinal da palavra, no modelo com professor de 16 épocas,
tirada do `paineis.tsv`.

| sinal na palavra | palavras | imagens | com marca |
|---|---|---|---|
| agudo | 10 | 400 | 88% |
| til | 5 | 200 | 86% |
| cedilha e til | 2 | 80 | 89% |
| circunflexo | 2 | 80 | 71% |
| só cedilha | 11 | 440 | 45% |

Das 350 imagens acentuadas sem marca, 242 são de palavras só com cedilha. Nas
folhas dessas palavras (13 palavras, 10 escritores), vistas no olho e sem
contagem, a cedilha aparece na maioria das imagens e quase sempre embaixo do
c. As marcas que o detector acusa são a cedilha quando sai solta do c ou um
acento a mais em outra letra. A deformação da letra não se concentra perto do
ç. A falta de marca nessas palavras é portanto da medida e não do modelo.

### Lastro das medidas

O detector e o leitor foram comparados com a anotação humana de 200 imagens
do modelo peso 5 (`avaliacao_diacriticos/resultados/kappa_peso5.txt`).

| medida | concordância com a pessoa | lastro |
|---|---|---|
| há marca, palavras sem acento | kappa 0,88 | forte |
| há marca, palavras acentuadas | kappa 0,72 | bom |
| marca na letra certa | kappa 0,29, com 80 imagens | fraco |
| legível, pelo CER | kappa de 0,07 a 0,40 conforme o corte | serve para comparar modelos, não como taxa de legibilidade |
| cedilha | detectada em 1 das 11 que a pessoa viu | nenhum |

A pessoa achou legíveis 171 das 200 imagens e o leitor lê sem erro 43. O CER
médio ainda separa os grupos, 0,28 nas legíveis contra 0,49 nas outras. A
anotação é de outro modelo, que deforma mais a letra, e cada sinal tem 20
imagens.

### Ressalvas

- Uma rodada de cada, na validação. O teste não foi aberto.
- O treino do acento separado rodou antes da correção da linha do "sem sinal",
  e o do professor depois. A comparação entre os dois mistura as duas
  mudanças. Falta o controle, acento separado corrigido e sem professor.
- A cedilha não é medida pelo detector e é 40% dos sinais avaliados.
- Os números de posição têm lastro fraco, como mostra a seção acima. A direção
  da melhora aparece nas folhas, o valor não está validado.

Registros em `diagnostico/resultados/acento_separado/`, com o resumo e o
`paineis.tsv` de cada avaliação, a posição, o `config.jsonl` de cada treino, as
linhas por época dos logs e quatro folhas de cada comparação. As imagens e os
modelos ficam na máquina de treino.

## Professor também nas acentuadas

Opção `--professor_fora_mascara`, ainda sem resultado. As amostras acentuadas
também passam pelo professor, com o texto sem acento. Dentro da máscara do
acento o alvo continua sendo o ruído sorteado, com o peso do acento. Fora dela
o alvo é o palpite do professor. Ataca a letra das palavras acentuadas. Exige
`--peso_acento` diferente de 1, que é o que carrega a máscara.

## Como rodar

Na máquina de treino, com o ambiente ativado.

```bash
bash scripts/aplicar_mods.sh
python -u DiffusionPen/train.py --dataset iam_acentuado --model_name diffusionpen \
    --sample_every 0 --save_path ./model_iam_pt_separado_professor \
    --dataset_folder ./iam_pt_alinhado --max_samples 0 --iam_originais 0.3 \
    --style_path ./DiffusionPen/style_models/iam_style_diffusionpen.pth \
    --stable_dif_path stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --lr 2e-05 --batch_size 32 --adamw_eps 1e-06 --clip_grad_norm 1.0 \
    --ema_beta 0.995 --ema_inicio 2000 --texto_max_len 40 --device cuda:0 \
    --num_workers 8 --save_every_steps 500 --abort_after 300 \
    --peso_acento 5 --peso_zona 1 --zona vazia --acento_separado \
    --professor ./DiffusionPen/diffusionpen_iam_model_path/models/ema_ckpt.pt \
    --epochs 16 --pretrained_path ./DiffusionPen/diffusionpen_iam_model_path/models \
    2>&1 | tee run_separado_professor.log
```

Variações.

| treino | o que muda no comando |
|---|---|
| controle, sem professor | tirar a linha do `--professor`, outro `--save_path` |
| professor também nas acentuadas | acrescentar `--professor_fora_mascara`, outro `--save_path` |
| triagem curta | `--epochs 6`. O `train.py` também guarda sozinho o modelo de 6 épocas de um treino longo, em `models/ema_ep5.pt` |

Para retomar um treino interrompido, trocar `--pretrained_path ...` por
`--load_check True` e `--epochs` pelo que falta. Os dois juntos fazem os pesos
de partida sobrescreverem o ponto retomado.

Avaliação e comparação.

```bash
python scripts/avaliar_pt.py --split val \
    --modelo meu_16ep=<pasta>/models/ema_ckpt.pt \
    --bases iam_pt_alinhado --leitor modelos/leitor_iam_transformer.pt \
    --saida avaliacao_meu_val
python scripts/comparar_avaliacoes.py \
    --modelo avaliacao_professor_val:professor_16ep --modelo avaliacao_meu_val:meu_16ep \
    --comparar meu_16ep professor_16ep
python scripts/medir_posicao_acento.py --avaliacao avaliacao_meu_val:meu_16ep \
    --saida diagnostico/resultados/posicao_meu
python scripts/comparar_paineis.py --modelo "professor=avaliacao_professor_val:professor_16ep" \
    --modelo "meu=avaliacao_meu_val:meu_16ep" --saida saidas/comparacao_meu
```

O que ainda não conhece as opções novas.

- `scripts/treinar.py` não tem as chaves no JSON, por isso a chamada direta.
- `scripts/gerar_amostras.py` não abre checkpoints com acento separado.
- `comum/diffusionpen.py` abre, então `scripts/avaliar_pt.py` e os scripts de
  `diagnostico/` funcionam sem mudança.

## Opções do train.py

Todas desligadas por padrão.

| opção | o que faz |
|---|---|
| `--acento_separado` | o CANINE lê o esqueleto e o diacrítico entra por um vetor somado na posição da letra |
| `--lr_acento_mult` | multiplicador da lr da tabela de vetores, 30 por padrão |
| `--professor arquivo.pt` | nas amostras sem diacrítico, o alvo é a previsão do modelo de partida |
| `--peso_professor` | mistura entre o professor (1) e o ruído sorteado (0) |
| `--professor_fora_mascara` | o professor também nas acentuadas, fora da máscara do acento |

## Próximos testes

Um por vez, com 6 épocas, mudando uma coisa em relação ao treino com professor.

| ordem | o que muda | ataca |
|---|---|---|
| 0 | controle sem professor, 16 épocas | atribuir o ganho |
| 1 | `--professor_fora_mascara` | a letra das palavras acentuadas |
| 2 | circunflexo repetido na base | o sinal raro |
| 3 | peso do acento de 5 para 10 | perdeu o motivo, os 29% sem marca são quase todos cedilha |

Entre 6 e 16 épocas a ordem entre os métodos não mudou nos dois treinos
avaliados nos dois pontos, o que sustenta a triagem curta.

## Trabalhos relacionados

- Masip, Rodríguez, Tuytelaars e van de Ven, Continual Learning of Diffusion
  Models with Generative Distillation, CoLLAs 2024, arXiv 2311.14028. Treinar
  um modelo de difusão em imagens geradas por ele mesmo, com o ruído sorteado
  como alvo, degrada a capacidade de remover ruído. Casar as previsões de
  professor e aluno corrige. A seção 4.3 testa a variante que usa os dados da
  tarefa nova como entrada.
- Ahitoliev e Berezin, Diffusion-Based Ukrainian Handwritten Text Generation
  with Cross-Domain Style Transfer, arXiv 2605.27487. Retreina o DiffusionPen
  em 126 mil palavras reais em ucraniano, 200 épocas, com descarte de texto em
  20% dos passos. O apóstrofo, marca pequena e rara, fica como a falha mais
  clara.
- Pandey, Open-Set Personalized Handwriting Generation with DiffusionPen,
  relatório técnico 2026-43 da University at Buffalo. A atenção cruzada do
  código público tem `mask = None`, em palavras curtas o preenchimento vence
  os caracteres reais, e um fine-tune curto com a máscara corrige.
