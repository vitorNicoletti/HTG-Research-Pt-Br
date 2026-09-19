# Diagnóstico de hardware

**Rode isto em qualquer GPU nova antes de treinar nela.** São minutos, e evita
semanas de resultados inexplicáveis.

> ## Caracterização final (2026-09-19)
>
> **O forward está correto. O defeito está isolado no backward.**
>
> Tensor de saída inteiro do UNet, CPU como referência, com aquecimento por
> formato, `eval()` nos dois lados e amostras diferentes dentro do lote
> (`teste_forward_completo.py`):
>
> | lote | 1 | 2 | 4 | 8 | 16 | 32 |
> |---|---|---|---|---|---|---|
> | erro relativo | 2,0e-06 | 2,1e-06 | 2,5e-06 | 2,4e-06 | 2,2e-06 | 2,3e-06 |
>
> Plano, sem dependência de lote. É ruído normal de fp32.
>
> O gradiente, nas mesmas condições (`teste_gradiente_pareado.py`,
> `teste_acumulacao_gradiente.py`):
>
> | medida | CPU | GPU |
> |---|---|---|
> | \|grad\| lote 32, 5 passos | 6,8 a 10,9 | 51 a 74 (**5 a 9×**) |
> | \|grad\| lote 4 | 6,67 | 37,9 (**5,6×**) |
> | \|grad\| acumulando lotes 1 | 6,67 | 152,9 (**23×**) |
>
> Como o forward está correto, as ativações que o backward consome estão
> corretas, e o erro nasce no próprio backward.
>
> **Acumulação de gradiente não resolve.** Era a saída prática óbvia — treinar
> em lotes de 1 e somar — e a medição a descarta: é o pior dos caminhos, 23×
> fora da referência contra 5,6× do lote 4. Não há configuração segura de
> treino nesta placa.
>
> ### Afirmações anteriores que caíram
>
> Duas caracterizações deste documento foram refutadas por medições melhores, e
> ficam registradas para quem for reproduzir:
>
> 1. *"Duas chamadas idênticas com o mesmo lote divergem"* — era o transiente de
>    primeira chamada, medido sem aquecimento.
> 2. *"A falha é seletiva por formato de lote: 1 e 2 certos, 4 e 16 errados"* —
>    medido no forward sem aquecimento por formato. Com aquecimento, todos os
>    lotes de 1 a 32 dão ~2e-06.
>
> O que sobrevive das duas é o **transiente de primeira chamada por formato**,
> que é real mas é um efeito separado e menor.
>
> ## Histórico: o que reproduz e o que não reproduz
>
> **Reproduz sempre:** gradiente não-finito no *backward*.
> `teste_gradiente_sintetico.py` dá **1/60 (2%)** nesta máquina, idêntico em
> execuções repetidas, usando só o checkpoint público do DiffusionPen e dados
> sintéticos. No treino real eram ~11% em todas as épocas. **É este o teste
> para comparar máquinas** — e é este o fenômeno que inviabilizou os treinos.
>
> **NÃO reproduz sob demanda:** o erro do *forward* por tamanho de lote. Com
> aquecimento de cada formato antes de medir, seis processos seguidos deram
> ~3e-07 em todos os lotes — limpos. O erro de 9,71e-02 no lote 4 aparece
> apenas em certas sequências de chamadas dentro do processo. O fenômeno é
> real e foi medido mais de uma vez, mas depende do histórico de formatos
> executados, o que é consistente com a hipótese de reuso de área temporária
> de memória (*workspace*) e **não permite afirmar "o lote 4 sempre erra"**.
>
> Consequência para o texto do TCC: descreva o achado principal como
> **gradientes não-finitos reprodutíveis no backward**, e trate a dependência
> de lote no forward como observação secundária, com a ressalva de que ela é
> intermitente.
>
> **Controle ainda pendente:** rodar `teste_gradiente_sintetico.py` com
> `DEVICE=cpu`. Se a CPU também produzir gradiente não-finito, a causa é do
> modelo ou da configuração, não da placa. É lento (minutos por passo), mas é
> a medida que fecha o argumento.

## Resumo do que foi encontrado

Todo o treino deste projeto até 19/09/2026 rodou numa **AMD Radeon RX 6600 XT**
(RDNA2, ROCm, `HSA_OVERRIDE_GFX_VERSION=10.3.0`, PyTorch 2.12 / HIP 7.2). Essa
placa **não computa este modelo de forma confiável**.

**A falha é seletiva por formato de lote.** Medido com `teste_repeticao.py`
(5 chamadas idênticas por lote, descartando a primeira) e o controle
`teste_repeticao_cpu.py`, ambos com o checkpoint `ema_ep11.pt`:

| lote | GPU vs lote 1 | CPU vs lote 1 | veredito |
|---|---|---|---|
| 1 | referência | referência | — |
| 2 | 2,91e-07 | 4,83e-07 | **correto** nos dois |
| 4 | **9,71e-02** | 5,19e-07 | GPU **errada**, 187 mil vezes a CPU |
| 16 | **3,09e-01** | — | GPU **errada** |

Não é "lote maior que 1 quebra": lotes 1 e 2 estão certos, lotes 4 e 16 estão
errados. Essa seletividade é a assinatura mais informativa do defeito.

**O lote 2 é o controle que descarta erro no próprio teste.** O script replica
a entrada com `.repeat(n, 1)`, inclusive o `style_extractor`, que tem formato
`(5, 1280)` por amostra. Se essa replicação estivesse errada, o lote 2 quebraria
tanto quanto o lote 4. Ele dá 2,91e-07 — precisão perfeita. Logo a replicação
está correta e o que falha no lote 4 não é o harness.

Outros dois fenômenos, menores e distintos do principal:

- **Efeito de primeira chamada, por formato.** A primeira chamada de cada
  tamanho de lote difere das seguintes; da segunda em diante o resultado é
  **bit a bit idêntico** (6 de 10 pares idênticos com 5 repetições — exatamente
  os 6 pares formados pelas repetições 2 a 5). Não é aleatoriedade geral: é um
  transiente de primeira execução.
- **Irreprodutibilidade real, só no lote 16.** Ali nem as chamadas aquecidas
  concordam entre si (dispersão de 9,45e-04 a 6,12e-02, nenhum par idêntico).

Uma convolução isolada passa no teste (1,1e-06 contra a CPU, e determinismo
perfeito), então o defeito não está no kernel de convolução sozinho — aparece
no modelo completo.

> **Correção de uma versão anterior deste documento.** A primeira redação
> afirmava "duas chamadas idênticas com o mesmo lote já divergem", com base numa
> medida de 1,75e-03 feita sem aquecimento adequado. Isso **não se sustenta**:
> com repetições e descarte da primeira chamada, os lotes 1, 2 e 4 são bit a bit
> determinísticos. O fenômeno medido era real, mas foi interpretado como
> instabilidade geral quando era o transiente de primeira chamada. A conclusão
> principal — a dependência do tamanho do lote — não só sobrevive à correção
> como fica mais forte, porque agora tem o lote 2 como controle interno.

## O que isso explicou

Esta era a causa raiz de praticamente tudo que travou o projeto:

- ~11% dos batches com gradiente não-finito, em todas as épocas de todos os
  treinos;
- a qualidade do modelo **piorando** conforme treinava, enquanto o MSE
  melhorava (o loss era calculado no mesmo regime corrompido);
- grades de amostra saindo em ruído colorido ou preto, sem que os pesos
  tivessem qualquer `NaN` (verificado: 0 em 170.908.868 parâmetros);
- o padrão aparente de "palavras curtas falham" — era artefato de geração em
  lote, não do modelo.

Nada disso era o código, o BRESSAY, o extrator de estilo ou a taxa de
aprendizado.

## Os testes

### `teste_conv_isolada.py`
O menor teste possível: uma `nn.Conv2d`, entrada sintética, CPU contra GPU.
Não precisa de checkpoint nem de dataset. **Comece por aqui.** Se falhar, a
placa está fora de questão. Se passar, ela ainda pode falhar no modelo
completo — foi o que aconteceu aqui.

### `teste_lote_unet.py`
Uma passada do UNet com a mesma entrada replicada, varrendo o tamanho do lote
de 1 a 32. Compara a linha 0 de cada lote contra o lote 1. Precisa de um
checkpoint (ajuste a constante `CKPT` no topo). **É o teste que pegou o
defeito.**

### `teste_lote_vs_cpu.py`
O mesmo, mas com a CPU como referência absoluta, para decidir qual dos dois
resultados está certo. Mais lento (a passada na CPU leva minutos).

### `teste_gradiente_cpu_vs_gpu.py`
Compara o **gradiente** (não só o forward) entre CPU e GPU, com batches reais do
BRESSAY, e tenta capturar um batch em que a GPU produza gradiente não-finito
para recalculá-lo na CPU. Define `CKPT_DIR` com uma pasta `models/`.

Foi o teste que revelou que a **primeira** chamada de cada processo diverge das
seguintes (loss 0,14865 contra 0,15532, com as chamadas 2 e 3 idênticas até
1e-12). Esse comportamento de primeira chamada **continua sem explicação** — só
se sabe que não vem de convolução quebrada, porque `teste_conv_isolada.py`
passa. Fica registrado como pendência.

Cuidado ao usá-lo: numa versão anterior deste teste o modelo ficava em
`train()`, o que deixava o dropout do CANINE ativo dentro do UNet; os dois
lados sorteavam máscaras diferentes e a comparação inteira virava lixo (quase
produziu um falso positivo de "GPU culpada"). Por isso o `monta()` força
`eval()`.

## Como interpretar

Os limiares estão declarados nos próprios scripts, de propósito, para não
serem ajustados depois de ver o resultado:

- **~1e-6 a 1e-5** — diferença normal de fp32 (ordem de redução distinta entre
  CPU e GPU). Placa saudável.
- **>= 1e-2** — a GPU está calculando errado.
- **Duas chamadas idênticas divergindo** — instabilidade; a placa não serve
  para treinar, em nenhuma configuração.

## Ambiente

Na RX 6600 XT era obrigatório:

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0        # gfx1032 usando kernels de gfx1030
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
```

Sem o `expandable_segments`, o treino entrava de forma **determinística** num
estado em que todo backward dava gradiente não-finito — sempre no passo 11 da
época, travando de vez no passo 71, reproduzido três vezes com ordens de dados
diferentes. Com ele, esse modo de falha específico desaparece, mas a
instabilidade acima permanece.
