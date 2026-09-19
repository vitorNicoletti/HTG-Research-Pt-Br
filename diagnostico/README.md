# Diagnóstico de hardware

**Rode isto em qualquer GPU nova antes de treinar nela.** Custa minutos e evita
semanas de resultados inexplicáveis.

Máquina onde o defeito foi encontrado: **AMD Radeon RX 6600 XT** (RDNA2,
gfx1032 rodando como gfx1030 via `HSA_OVERRIDE_GFX_VERSION=10.3.0`), ROCm,
PyTorch 2.12 / HIP 7.2.53211. Modelo: UNet do DiffusionPen, ~170M parâmetros,
difusão latente, latentes 4×8×32.

---

## Conclusão

**O forward está correto. O defeito está isolado no backward, e é de direção,
não de escala.**

### Forward: correto

Tensor de saída **inteiro** do UNet (não o loss, que é escalar e pode esconder
erros que se cancelam), CPU como referência — `teste_forward_completo.py`:

| lote | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| erro relativo | 2,0e-06 | 2,1e-06 | 2,5e-06 | 2,4e-06 | 2,2e-06 | 2,3e-06 |

Plano, sem dependência de lote. É ruído normal de fp32.

### Backward: errado

Mesmas entradas, mesmos pesos — `teste_gradiente_pareado.py`:

| lote 32, 5 passos | CPU | GPU |
|---|---|---|
| loss | 2,825638 | 2,825733 *(concorda na 4ª casa)* |
| \|grad\| | 6,8 a 10,9 | 51 a 74 — **5 a 9×** |

E o dado decisivo, `teste_direcao_gradiente.py`, lote 8:

```
cosseno entre o gradiente da CPU e o da GPU ........ 0,357
fração da norma da GPU ortogonal ao correto ........ 93,6%
```

**Não é erro de escala, é de direção.** Isso importa porque o treino usa
`clip_grad_norm_`, que normaliza a magnitude — se fosse só escala, o clip
corrigiria. Como é direção, 94% de cada passo de treino é ruído ortogonal à
descida correta. É por isso que treinar piorava o modelo.

### Acumulação de gradiente não resolve

Era a saída prática óbvia: treinar em lotes de 1 e somar, acionando só formatos
pequenos. `teste_acumulacao_gradiente.py`, com 4 amostras diferentes:

| comparação | erro relativo | normas |
|---|---|---|
| sanidade CPU: lote 4 vs acumulado | 3,62e-06 | 6,67 / 6,67 |
| GPU lote 4 vs CPU lote 4 | 5,60 | 37,9 / 6,67 |
| GPU acumulado vs CPU acumulado | **22,9** | 152,9 / 6,67 |

Acumular é o **pior** caminho, não o mais seguro. **Não há configuração segura
de treino nesta placa.** Gerar imagem, por outro lado, está liberado: é só
forward.

### A CPU está limpa

- `teste_gradiente_sintetico.py` com `DEVICE=cpu`: **0/10** gradientes
  não-finitos (a GPU dá 1/60 no mesmo teste), `|grad|` mediana 6,8 contra 57,1.
- Invariância ao lote: 4,83e-07 (lote 2), 5,19e-07 (lote 4).
- Treino real de 625 passos: **0 batches descartados**.

---

## O defeito do forward era dependente do estado da máquina

Registrado porque é a armadilha mais perigosa deste diagnóstico.

O mesmo script, mesmo checkpoint, mesma configuração:

| | antes do reinício (3,5 dias de uptime) | depois do reinício |
|---|---|---|
| lote 4 | 9,71e-02 | 3,20e-07 |
| lote 16 | 3,09e-01 | 3,16e-07 |
| pares bit-idênticos | 6/10 | 10/10 |
| transiente de 1ª chamada | presente | ausente |

O reinício eliminou o defeito do forward **e** o transiente de primeira
chamada. O defeito do backward sobreviveu — todas as medidas da seção anterior
são pós-reinício.

O cache do MIOpen (`~/.cache/miopen`) persiste entre reinícios, então não é ele.

**Consequência prática: anote o uptime da máquina junto de cada medida.** Sem
isso, duas execuções idênticas podem discordar e ninguém saberá por quê.

---

## O que continua em aberto

Declarado como pendência, não como fato:

1. **Qual operação do backward produz o erro.** Uma convolução isolada passa no
   teste; o defeito só aparece no modelo completo.
2. **Se a causa é física ou de software.** Testável trocando a versão do ROCm.
3. **Que estado de máquina, limpo por reinício, corrompia o forward.**
4. **Por que acumular lotes de 1 erra mais (23×) que um lote 4 (5,6×)**, se
   acumular envolve reduções menores.

---

## Validar uma máquina nova (roteiro para SSH)

```bash
nix develop .#cuda          # NVIDIA  |  .#rocm para AMD  |  .#cpu sem GPU

# 1. o mais barato: uma convolucao isolada. Segundos, sem checkpoint.
python diagnostico/teste_conv_isolada.py

# 2. comparavel entre maquinas: gradiente nao-finito, com o checkpoint
#    PUBLICO do DiffusionPen e dados sinteticos
CKPT=./DiffusionPen/diffusionpen_iam_model_path/models/ema_ckpt.pt \
    python diagnostico/teste_gradiente_sintetico.py
#    0/60 nao-finitos = saudavel | > 0 = mesmo problema desta maquina

# 3. o conclusivo: a direcao do gradiente bate com a CPU?
CKPT=... python diagnostico/teste_direcao_gradiente.py
#    cosseno ~1,0 = saudavel | cosseno baixo = nao treine nesta placa
```

Rode 2 ou 3 vezes cada um, em processos separados, e registre o uptime.

**Como ler os números.** Os limiares estão escritos dentro dos próprios
scripts, de propósito, para não serem ajustados depois de ver o resultado:

- **~1e-6 a 1e-5** — diferença normal de fp32 entre CPU e GPU. Saudável.
- **≥ 1e-2** — a GPU está calculando errado.
- **cosseno < 0,9 no gradiente** — a direção está errada; não treine.

---

## Os testes

| arquivo | o que mede | precisa de |
|---|---|---|
| `teste_conv_isolada.py` | uma `nn.Conv2d`, CPU vs GPU | nada |
| `teste_forward_completo.py` | tensor de saída inteiro, lotes 1 a 32 | checkpoint |
| `teste_gradiente_pareado.py` | loss e \|grad\|, entradas idênticas | checkpoint |
| `teste_direcao_gradiente.py` | **cosseno** entre gradientes CPU e GPU | checkpoint |
| `teste_acumulacao_gradiente.py` | acumular lotes de 1 resolve? | checkpoint |
| `teste_gradiente_sintetico.py` | gradientes não-finitos, dados sintéticos | checkpoint público |
| `teste_repeticao.py` / `_cpu.py` | repetibilidade e efeito do lote | checkpoint |
| `teste_lote_unet.py` / `_vs_cpu.py` | versões anteriores, superadas | checkpoint |
| `teste_gradiente_cpu_vs_gpu.py` | gradiente com batches reais do BRESSAY | checkpoint + dataset |
| `treinar_cpu.py` | treino de referência em hardware limpo | dataset |
| `vigia_termico.sh` | mata o treino se a CPU passar do limite | — |

---

## Armadilhas operacionais

Valem para qualquer máquina, não só para a defeituosa. Cada uma custou horas.

- **Nunca avalie o modelo pelo MSE.** Ele caiu de 0,0522 para 0,0404 na mesma
  época em que a geração morreu por completo. Gere amostras.
- **Use a deriva em relação ao pré-treinado como métrica de saúde.** Custa
  segundos e não mente: deriva 0,046% = modelo intacto; 0,351% = destruído.
- **Em qualquer comparação CPU vs GPU, use `eval()` nos dois lados.** O CANINE
  fica aninhado dentro do UNet e tem dropout 0,1; em `train()` cada lado sorteia
  máscaras diferentes e a comparação vira lixo. Esse bug invalidou duas medidas
  desta investigação.
- **Gere as entradas uma vez na CPU e copie para a GPU.** `torch.randn(device=)`
  usa geradores diferentes em cada dispositivo — sem isso você compara entradas
  diferentes achando que são as mesmas.
- **Aqueça cada formato de tensor antes de medir.** A primeira chamada de cada
  formato difere das seguintes.
- **Não confie em grade de amostra gerada dentro do treino.** O
  `sampling_loader` do DiffusionPen tokeniza com `max_length=200` enquanto o
  treino usa 40 — condicionamento diferente do treino.
- **`--pretrained_path` e `--load_check` são mutuamente exclusivos.** No
  `train.py` o bloco do `pretrained_path` roda **depois** do `load_check` e
  sobrescreve silenciosamente os pesos retomados.
- **`--load_check` é `type=bool` no argparse**, então `--load_check False` vira
  `True`. Só passe a flag quando quiser mesmo retomar.
- **O EMA precisa do `ema.step` restaurado ao retomar**, senão o `step_ema`
  copia os pesos do modelo por cima do EMA carregado e o destrói.
- **Escritores de train e val são disjuntos**, então a acurácia de validação do
  extrator de estilo é 0,0000 por construção. Não é defeito.
- **Comparar geração CPU vs GPU em lote 1 engana.** Uma diferença legítima de
  1e-6 é amplificada por 50 iterações do DDIM e produz imagens visivelmente
  distintas, **ambas válidas**. Isso não demonstra defeito.

---

## Um segundo problema, independente do hardware

Medido **na CPU**, com hardware comprovadamente limpo (0 batches descartados):
o fine-tune com `lr=2e-5` + AdamW destrói o modelo em uma época.

| | deriva do pré-treinado | geração |
|---|---|---|
| `lr=2e-5`, 625 passos | 0,351% | destruída (borrões) |
| `lr=1e-6`, 200 passos | 0,046% | legível, mas borrada |
| `lr=3e-7`, 200 passos | 0,017% | **nítida** |

A razão: o AdamW move cada parâmetro ~`lr` por passo, independente do tamanho
do gradiente. Em 625 passos com `lr=2e-5` isso dá 0,0125 de deriva por peso,
contra uma magnitude mediana de **0,0219** — mais da metade do peso típico numa
única época.

Nos três pontos medidos, nenhum produziu diacríticos: quanto mais o modelo se
afasta do pré-treinado, pior a imagem, e o acento nunca aparece. Foram apenas
200 passos por ponto, então a variável "mais passos com `lr` baixo" continua
não testada.

Resultados em `resultados/`.

---

## Correções de versões anteriores deste documento

Registradas porque os números aparecem no histórico do git e alguém pode
reproduzi-los:

1. *"Duas chamadas idênticas com o mesmo lote divergem"* — era o transiente de
   primeira chamada, medido sem aquecimento.
2. *"A falha é seletiva por formato: lotes 1 e 2 certos, 4 e 16 errados"* —
   medido antes do reinício da máquina. Depois do reinício, todos os lotes de 1
   a 32 dão ~2e-06 no forward.
3. *"Aquecimento por formato explica a discrepância entre execuções"* — a
   variável era o estado da máquina, não o aquecimento.
4. *"O forward está corrompido, logo a geração é contaminada pelo laço do
   DDIM"* — a premissa é falsa; o forward está correto.
5. *"Acumulação de gradiente é o caminho seguro"* — é o pior caminho.
