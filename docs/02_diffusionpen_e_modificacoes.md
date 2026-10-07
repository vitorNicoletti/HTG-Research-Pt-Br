# 02 — O DiffusionPen e as mudanças que fizemos no treino

Este documento tem duas partes:
1. **como o DiffusionPen funciona**, no nível necessário para entender os
   experimentos;
2. **tudo o que mudamos** no código oficial de treino, e por quê.

---

## Parte 1 — Como o DiffusionPen funciona

### A ideia de difusão, em uma página

Um modelo de difusão aprende a **remover ruído** de uma imagem.

- **No treino:** pega-se uma imagem real, sorteia-se um "nível de ruído" `t`
  entre 0 e 999 e mistura-se ruído gaussiano `ε` na imagem, na proporção
  daquele nível. O modelo recebe a imagem ruidosa e o `t`, e precisa adivinhar
  **qual ruído foi misturado**. O erro é o quadrado da diferença entre o ruído
  verdadeiro e o previsto (MSE):

  ```
  loss = média( (ε − ε̂)² )
  ```

- **Na geração:** começa-se de ruído puro e aplica-se o modelo repetidas vezes.
  Cada passo remove um pouco do ruído previsto, até sobrar uma imagem. Usamos o
  agendador **DDIM com 50 passos**.

- **Condicionamento:** o modelo também recebe o **texto** e o **estilo**.
  Assim, "remover o ruído" passa a significar "remover o ruído de modo que
  sobre a palavra pedida, na caligrafia pedida".

### Difusão latente

Trabalhar direto nos pixels é caro. O DiffusionPen usa o **VAE do Stable
Diffusion 1.5** para comprimir a imagem:

```
imagem 3 × 64 × 256  ──VAE.encode──>  latente 4 × 8 × 32  (× 0,18215)
```

O ruído, a previsão e a loss acontecem **nesse latente**. Cada célula dele
corresponde a um bloco de 8×8 pixels da imagem. Um acento de poucos pixels
ocupa 1 a 4 células. Isso importa para os pesos na loss que criamos (parte 2).

O teste em `diagnostico/teste_vae_resolucao.py` (resultado no
[05_finetune_bressay.md](05_finetune_bressay.md)) mostra que **em 64×256 o VAE
preserva os acentos**: a compressão não é o gargalo.

### As entradas da rede (o UNet)

| entrada | de onde vem | treina no fine-tune? |
|---|---|---|
| latente ruidoso | VAE do SD 1.5 | o VAE fica congelado |
| nível de ruído `t` | sorteado | — |
| **texto** | **CANINE-C**: cada caractere vira seu número Unicode ("ã" = 227), depois um vetor; até 40 caracteres | o CANINE fica congelado; a projeção `text_lin`, dentro do UNet, treina |
| **estilo** | **MobileNetV2** treinada para reconhecer escritores; recebe 5 imagens do escritor e a média dos vetores vira o estilo | congelada (usamos a do IAM) |

O UNet tem ~170 milhões de parâmetros e é **a única parte que treina** nos
nossos fine-tunes.

**Por que o CANINE importa para nós:** ele não tem vocabulário fixo, então
"ã", "ç" e "é" chegam ao modelo sem erro. A sonda (fase 1) confirmou que o
caractere chega: "nacao" e "nação" geram imagens diferentes. O que falta é o
UNet ter aprendido a desenhar o sinal.

**Um limite do CANINE para nós:** "provável" e "provavel" diferem em um único
caractere e ficam muito parecidos depois do codificador (semelhança de cosseno
0,935 no diagnóstico do [07](07_experimentos_com_acentos.md)). O sinal de "tem
acento" chega fraco ao UNet.

### O EMA

Durante o treino mantém-se uma cópia dos pesos que é a **média móvel**
(β = 0,995) dos pesos treinados. A geração usa essa cópia (`ema_ckpt.pt`),
mais estável que os pesos crus. Antes do passo 2.000, o EMA é só uma cópia
direta do modelo.

### Sem *classifier-free guidance*

Muitos modelos de difusão treinam às vezes **sem** o texto, para depois, na
geração, "empurrar" a imagem na direção do texto (guidance). No DiffusionPen
original a linha que faria isso (`labels = None`) **não tem efeito**: o
modelo sempre treina com o texto. Consequência: não dá para usar guidance para
reforçar o acento sem treinar de novo. Isso aparece entre os problemas em
aberto.

### O pré-processamento do IAM

Cada palavra é redimensionada para 64 px de altura, mantendo a proporção, e
centralizada numa tela de 256 px de largura. Palavras mais largas são
encolhidas de 20 em 20 px até caber. **Mantivemos esse pré-processamento
copiado sem mudança** (`preprocessar_iam` em `utils/iam_acentuado_dataset.py`),
porque é o que o modelo viu no treino original.

---

## Parte 2 — O que mudamos no treino oficial

### Como as mudanças são aplicadas

O repositório do DiffusionPen é clonado à parte em `DiffusionPen/` e **não é
versionado** aqui. As versões modificadas dos arquivos ficam em
`diffusionpen_mods/` e são copiadas por cima do clone:

```bash
bash scripts/aplicar_mods.sh             # copia diffusionpen_mods/ para DiffusionPen/
bash scripts/aplicar_mods.sh --conferir  # só mostra o que está diferente
```

São **substituições de arquivo inteiro**, não um patch. O lançador
`scripts/treinar.py` roda o `aplicar_mods.sh` antes de cada treino.

| arquivo modificado | destino no clone |
|---|---|
| `diffusionpen_mods/train.py` | `DiffusionPen/train.py` |
| `diffusionpen_mods/style_encoder_train.py` | `DiffusionPen/style_encoder_train.py` |
| `diffusionpen_mods/utils/bressay_dataset.py` | `DiffusionPen/utils/bressay_dataset.py` (novo) |
| `diffusionpen_mods/utils/iam_acentuado_dataset.py` | `DiffusionPen/utils/iam_acentuado_dataset.py` (novo) |

A sonda (fase 1) usa outra coisa: um patch **aditivo**,
`sonda/patches_diffusionpen.diff`, que só acrescenta um modo de geração.

### O que **não** mudamos

- **A arquitetura do UNet:** ela precisa bater com o checkpoint do IAM.
- **O VAE, o CANINE e o extrator de estilo:** congelados.
- **O agendador de ruído e a geração (DDIM, 50 passos).**
- **O pré-processamento do IAM.**

### Tabela: original × nosso

| aspecto | DiffusionPen original | nosso `train.py` | motivo |
|---|---|---|---|
| ponto de partida | do zero | `--pretrained_path`: pesos do IAM (457/457 chaves) | fine-tune, não treino do zero |
| taxa de aprendizado | 1e-4, fixa no código | `--lr`, usamos **2e-5** | fine-tune pede passos menores, para não destruir o que o modelo sabe |
| AdamW `eps` | 1e-8 (padrão) | `--adamw_eps` **1e-6** | estabilidade numérica |
| corte da norma do gradiente | nenhum | `--clip_grad_norm` **1,0** | evita passos gigantes |
| gradiente não finito (`inf`/`NaN`) | só checava a loss | **lote descartado antes do `optimizer.step()`** | um único gradiente `inf` envenena para sempre o estado do AdamW (`exp_avg_sq`), e todo passo seguinte escreve `NaN` nos pesos. Foi o que destruiu o primeiro treino |
| métricas por época | média acumulada desde o passo 0 | zeradas a cada época | a média antiga escondia épocas inteiras perdidas |
| retomar treino | reiniciava a contagem de épocas e o EMA | `models/estado.pt` guarda a época e o passo do EMA; `--load_check` retoma | sem isso, retomar **apagava o EMA** (o passo voltava a ser < 2.000 e o EMA era sobrescrito) |
| checkpoint no meio da época | não | `--save_every_steps` (usamos 500) | a máquina às vezes desligava no meio de uma época de 18 min |
| processo travado | — | `--abort_after N`: sai com código 3 após N lotes seguidos sem passo válido; o lançador relança | herança do diagnóstico da placa defeituosa |
| grade de amostras durante o treino | a cada época, com `max_length=200` | `--sample_every 0` (desligada) | o treino usa 40 caracteres; a grade com 200 não representa o modelo. Usamos `scripts/gerar_amostras.py` |
| `num_workers` | ignorado (4 fixo) | respeitado | velocidade |
| tamanho do texto | 40 no treino | `--texto_max_len` 40 | só tornado explícito e registrado |
| EMA | β 0,995, início no passo 2.000 | `--ema_beta`, `--ema_inicio` (mesmos valores) | só tornado explícito |
| leitor de dados | IAM e GNHK | **+ `bressay`** e **+ `iam_acentuado`** (`--dataset`) | as nossas bases |
| pasta e tamanho dos dados | fixos | `--dataset_folder`, `--max_samples` | splits reduzidos e bases sintéticas |
| fração de palavras reais do IAM | — | `--iam_originais` (0 a 1) | misturar escrita real às bases sintéticas |
| pré-processamento do BRESSAY | — | `--preproc v1` ou `v2` | ver [05b](05b_pre_processamento_bressay.md) |
| **peso no acento** | — | **`--peso_acento λ`** | ver abaixo |
| **peso na zona** | — | **`--peso_zona λ`, `--zona vazia\|vogais`** | ver abaixo |
| registro | — | `SAVE_PATH/config.jsonl`: uma linha por lançamento, com argumentos, dados, ambiente e commit do git | reprodutibilidade |

### A mudança principal: pesos na loss

Na loss original todas as células do latente pesam igual. Um acento ocupa ~2%
do latente de um lote, então errar o acento quase não muda a loss
(diagnóstico no [07](07_experimentos_com_acentos.md), fase 6). Criamos dois
pesos:

```
loss = média( w · (ε − ε̂)² )

w = 1 + (peso_acento − 1) · máscara_acento
      + (peso_zona   − 1) · máscara_zona
```

**Máscara do acento (`--peso_acento`).**
- Para cada imagem acentuada da base existe o **par**: a mesma imagem antes de
  desenhar o acento.
- A máscara é onde as duas diferem: pixels com diferença maior que 40 níveis
  de cinza, levados para a grade 8×32 do latente e dilatados em 1 célula, já
  que o VAE espalha o sinal para as vizinhas.
- **A acentuada e o par recebem a mesma máscara.** Na acentuada, o peso cobra
  desenhar o acento; no par, cobra **não** desenhar ali.
- Exige que acentuada e par estejam na mesma tela, alinhados (ver fase 6 do
  [07](07_experimentos_com_acentos.md)).
- Código: `mascara_acento` em `utils/iam_acentuado_dataset.py`.

**Máscara da zona (`--peso_zona`, `--zona`).** Cobra tinta onde um acento que
o texto não pede apareceria. Há duas versões:
- **`vazia`:** toda a faixa sem tinta logo acima e logo abaixo do corpo da
  palavra, até 1 altura-x. **Ela encurtava as hastes** das letras
  ([07](07_experimentos_com_acentos.md), fase 8).
- **`vogais`:** só a faixa acima de cada vogal e abaixo de cada c, nas colunas
  da letra, localizadas pelo alinhador CTC. Fica longe das hastes (0,5% das
  células sobre letras com haste, contra 21,6% da `vazia`). É pré-calculada
  por `scripts/mascaras_vogais.py` (código em `acentos_sinteticos/zona_vogais.py`).

**Detalhes de implementação:**
- **Os pesos não são normalizados.** Fora das máscaras, a loss é exatamente a
  original.
- **Com `peso_acento = 1` e `peso_zona = 1`, o caminho antigo roda sem
  nenhuma mudança.** As máscaras nem são calculadas. Os experimentos antigos
  continuam reproduzíveis.
- O log de cada época mostra, além da MSE com peso:
  - a MSE sem peso, comparável à dos treinos antigos;
  - a MSE dentro e fora da máscara do acento;
  - a MSE na zona e a fração do latente coberta por cada máscara.

### O leitor `iam_acentuado` (`utils/iam_acentuado_dataset.py`)

Lê uma base de acentos sintéticos (`split.txt`, `manifesto.jsonl`,
`resumo.json`) e mistura uma fração das palavras reais do IAM:
- **Referências de estilo:** sempre de palavras **reais** do mesmo escritor,
  com mais de 3 letras, como no leitor original. O estilo vem da caligrafia
  real, e o acento só pode vir do texto.
- **Anti-vazamento:** se o `resumo.json` da base aponta o vocabulário
  português, todo rótulo é conferido contra o split de treino. Uma palavra de
  validação ou teste interrompe o treino com erro.
- **Formato de saída:** devolve os mesmos 6 elementos do leitor do IAM. Com
  pesos ligados, devolve também a máscara do acento (7º) e a da zona (8º).

### O lançador `scripts/treinar.py` e os experimentos

Todo treino é descrito por um JSON em `experimentos/`:
- **Todas as chaves são obrigatórias.** Chave desconhecida é erro, então o
  arquivo é a lista completa do que foi usado.
- **Ao começar,** o lançador copia o JSON para `SAVE_PATH/experimento.json`.
- **Ao retomar,** só aceita mudar a descrição, o total de épocas, a execução
  e as amostras. Mudar qualquer outra coisa exige um `save_path` novo.
- **Em blocos:** treina `epocas_por_bloco` épocas de cada vez. No fim de cada
  bloco ele:
  - salva o EMA como `models/ema_bloco_<N>ep.pt` (os modelos que avaliamos);
  - mede a deriva dos pesos em relação ao IAM (`scripts/medir_deriva.py`);
  - gera amostras em `SAVE_PATH/amostras/<N>ep/`.
- **Relançamento:** se o `train.py` sair com código 3, o lançador relança do
  último checkpoint.

```bash
python scripts/treinar.py experimentos/iam_acentuado_vogais.json --dry-run   # só mostra o comando
python scripts/treinar.py experimentos/iam_acentuado_vogais.json             # treina (ou retoma)
```

Chaves de `experimentos/*.json`:

| seção | chaves |
|---|---|
| `dados` | `dataset` (`bressay`/`iam_acentuado`), `split` (pasta), `imagens`, `max_samples`, `preproc` (`v1`/`v2`/`iam`), `iam_originais` |
| `modelo` | `pesos_iniciais`, `extrator_estilo`, `stable_diffusion` |
| `treino` | `epocas_por_bloco`, `epocas_total`, `lr`, `batch_size`, `adamw_eps`, `clip_grad_norm`, `ema_beta`, `ema_inicio`, `texto_max_len`, `peso_acento`, `peso_zona`, `zona` |
| `execucao` | `device`, `num_workers`, `save_every_steps`, `abort_after` |
| `amostras` | `palavras`, `estilos`, `seed`, `estilo_de` |

Valores comuns a todos os nossos fine-tunes: lr 2e-5, lote 32, AdamW eps
1e-6, clip 1,0, EMA 0,995 a partir do passo 2.000, texto até 40 caracteres e
extrator de estilo do IAM.

### Correções no treino do extrator de estilo (`style_encoder_train.py`)

Feitas na fase do BRESSAY, quando treinamos um extrator de estilo próprio.
Os treinos seguintes usam o extrator do IAM.
- **Crash determinístico:** 50 dos 647 escritores do BRESSAY não têm nenhuma
  palavra com mais de 3 letras, e a escolha da amostra positiva ficava vazia.
  Agora o filtro é uma preferência, com alternativa.
- **RAM:** o leitor carregava as 74.882 imagens decodificadas, e cada processo
  de carga recopiava tudo (mais de 32 GB). Agora guarda só o caminho.
- **Velocidade:** cada amostra varria o dataset inteiro duas vezes. Agora usa
  mapas escritor → índices.
- **Escolha do checkpoint:** a soma das losses incluía a de classificação, que
  não pode ser aprendida com escritores de validação diferentes dos de treino.
  Agora a escolha usa só a triplet loss.

Detalhes em `diffusionpen_mods/README.md`.
