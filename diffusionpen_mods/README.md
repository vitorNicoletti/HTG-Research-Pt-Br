# Correções aplicadas ao DiffusionPen

O repositório do DiffusionPen é clonado à parte e fica no `.gitignore`
(`DiffusionPen/`). Estes arquivos são as versões **corrigidas** e
existem aqui para que as correções não se percam no re-clone e cheguem a quem
for rodar o treino em outra máquina.

Para usar: clone o DiffusionPen normalmente e sobrescreva com estes três
arquivos.

| arquivo | destino no clone |
|---|---|
| `train.py` | `DiffusionPen/train.py` |
| `style_encoder_train.py` | `DiffusionPen/style_encoder_train.py` |
| `utils/bressay_dataset.py` | `DiffusionPen/utils/bressay_dataset.py` |

## O que foi corrigido, e por quê

### `train.py`

**Gradiente não-finito envenenava o otimizador (causa do colapso do treino).**
O código original só checava se o *loss* era finito. Gradientes `inf`/`NaN`
passavam direto para o `optimizer.step()` e contaminavam de forma permanente o
`exp_avg_sq` do AdamW — depois disso, todo passo seguinte escrevia `NaN` nos
pesos, mesmo com loss finito. Foi o que destruiu o primeiro treino por volta da
época 28. Agora o batch é descartado **antes** do step, com contadores
separados para loss não-finito e gradiente não-finito.

**Métricas por época.** O `AvgMeter` e o contador de NaN nunca eram zerados, e
a média corria desde o passo 0 — dava para perder 90% dos batches de uma época
sem o MSE exibido se mexer.

**Retomada real.** Grava `models/estado.pt` com a época e o `ema.step`. Sem
isso, retomar reiniciava a contagem de épocas (sobrescrevendo imagens e
snapshots) e zerava o contador do EMA, o que fazia o `step_ema` **copiar** os
pesos do modelo por cima do EMA recém-carregado, destruindo-o.

**Novos argumentos:** `--max_samples` (limite de amostras por split; 0 = split
inteiro), `--sample_every`, `--save_every_steps`, `--abort_after` (sai com
código 3 após N batches seguidos sem passo válido, para um laço externo poder
relançar) e `--start_epoch`. Além disso `--num_workers` passou a ser
respeitado; antes era ignorado, com o valor 4 fixo no código.

### `utils/bressay_dataset.py`

O corte de 20.000 amostras era fixo no código e virou `--max_samples`. O
comentário original dizia "trava de RAM", mas o dataset é lazy — guarda só os
caminhos e abre a imagem no `__getitem__` — então o limite não economizava
memória nenhuma.

### `style_encoder_train.py`

**Crash determinístico no treino do extrator de estilo.** A seleção da amostra
positiva filtrava por transcrição com mais de 3 caracteres. No BRESSAY, que é
recortado em palavras de português, **50 dos 647 escritores do treino não têm
uma única palavra com mais de 3 letras** ("de", "do", "que"...), e a lista saía
vazia: `random.choice([])` → `IndexError`. Agora o filtro é preferência com
fallback.

**Dois crashes latentes.** `random.sample(k=5)` exige 5 elementos distintos e
estouraria nos escritores com menos de 5 amostras. Trocado por
`random.choices`, com reposição.

**Consumo de RAM.** O loader carregava as 74.882 imagens decodificadas na
memória do processo principal, e cada worker do DataLoader recopiava tudo —
estourava 32 GB. Agora guarda só o caminho e abre no worker.

**Velocidade.** Cada `__getitem__` varria o dataset inteiro **duas vezes** para
montar as listas de positivos e negativos (~150 mil comparações por amostra,
~11 bilhões por época). Substituído por mapas escritor→índices pré-computados.

**Seleção de checkpoint.** O "melhor modelo" era escolhido por
`classification_loss + triplet_loss`. Como os escritores de treino e validação
são disjuntos, a cabeça de classificação nunca viu as classes da validação e
aquele termo é inaprendível por construção — ele dominava a soma e tornava a
escolha praticamente aleatória. Agora a seleção usa só o triplet loss, que é o
que mede se o embedding de estilo generaliza para escritores novos.

## Aviso sobre o hardware

Todo o treino feito até aqui rodou numa AMD RX 6600 XT (ROCm), e essa placa
**não computa este modelo corretamente com lote maior que 2**: comparando cada
lote com o lote 1, a GPU erra 9,71e-02 no lote 4 e 3,09e-01 no lote 16,
enquanto a CPU fica em 5,19e-07 no mesmo teste — uma diferença de 187 mil
vezes. Lotes 1 e 2 estão corretos nos dois, e uma convolução isolada também
(1,1e-06). Veja `diagnostico/`.

Como o treino roda com `--batch_size 32`, todo forward e todo backward de todos
os treinos feitos nesta máquina caíram na faixa defeituosa.

Consequência prática: rode o teste mínimo de `diagnostico/` em qualquer placa
nova **antes** de treinar nela.
