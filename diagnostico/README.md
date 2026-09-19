# Diagnóstico de hardware

**Rode isto em qualquer GPU nova antes de treinar nela.** São minutos, e evita
semanas de resultados inexplicáveis.

## Resumo do que foi encontrado

Todo o treino deste projeto até 19/09/2026 rodou numa **AMD Radeon RX 6600 XT**
(RDNA2, ROCm, `HSA_OVERRIDE_GFX_VERSION=10.3.0`, PyTorch 2.12 / HIP 7.2). Essa
placa **não computa este modelo de forma confiável**.

| teste | resultado | veredito |
|---|---|---|
| uma `Conv2d` isolada, CPU vs GPU | 1,1e-06 | correto |
| UNet completo na CPU, lote 1 vs lote 4 | 5,2e-07 | a CPU é estável e independe do lote |
| UNet completo na GPU, lote 1, vs CPU | 9,7e-07 | correto |
| UNet completo na GPU, lote 4, vs CPU | 9,9e-02 | **errado** |
| UNet completo na GPU, entradas idênticas, vários lotes | **1,75e-03 a 1,08** | **instável** |

A varredura por tamanho de lote (`teste_lote_unet.py`) dá erros erráticos e
não-monotônicos — 1,75e-03 no lote 1, 4,07e-01 no lote 3, 1,00e-01 no lote 4,
1,08 no lote 16. Duas chamadas **idênticas** com o mesmo lote já divergem. Não
é um viés sistemático: é instabilidade.

Uma convolução isolada passa no teste, então o defeito não está no kernel de
convolução sozinho — aparece no modelo completo.

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
