# STATUS — onde a execução parou e como retomar

**Atualizado:** 2026-09-19
**Fase atual:** fine-tune executado; **bloqueado por hardware**
**Bloqueio ativo:** 1 — a GPU de desenvolvimento não computa o modelo de forma confiável

> O histórico das fases anteriores está no `LOG.md`. Este documento descreve
> só o estado atual e o que fazer a seguir.

---

## Resumo em uma linha

O pipeline completo roda de ponta a ponta — dados, extrator de estilo,
fine-tune, geração e métrica — mas **a RX 6600 XT usada no desenvolvimento
produz resultados numericamente instáveis**, e por isso nenhum modelo treinado
até agora é aproveitável. O próximo passo é repetir o treino numa GPU validada.

---

## O bloqueio

Medido em `diagnostico/`, com a CPU como referência:

| teste | resultado | veredito |
|---|---|---|
| uma `Conv2d` isolada, CPU vs GPU | 1,1e-06 | correto |
| UNet completo na CPU, lote 1 vs lote 4 | 5,2e-07 | a CPU independe do lote |
| UNet completo na GPU, lote 1, vs CPU | 9,7e-07 | correto |
| UNet completo na GPU, lote 4, vs CPU | 9,9e-02 | **errado** |
| UNet completo na GPU, entradas idênticas, lotes 1 a 32 | **1,75e-03 a 1,08** | **instável** |

Duas chamadas idênticas já divergem. Não é viés sistemático, é instabilidade.
Uma convolução isolada passa, então o defeito aparece só no modelo completo.

**Isso explicou todos os sintomas que travaram o projeto:** ~11% dos batches
com gradiente não-finito em todas as épocas; a qualidade piorando conforme
treinava enquanto o MSE melhorava; grades de amostra em ruído ou preto sem que
os pesos tivessem um único `NaN` (verificado: 0 em 170.908.868 parâmetros).

Nada disso era o código, o BRESSAY, o extrator de estilo ou a taxa de
aprendizado — todas essas hipóteses foram levantadas e descartadas por medição.

---

## O que está pronto e é aproveitável

**Correções no DiffusionPen** — `diffusionpen_mods/`, com README explicando
cada uma. São três arquivos que substituem os do clone. Sem eles o treino
corrompe o otimizador na primeira época com gradiente não-finito, o treino do
extrator de estilo quebra com `IndexError` em 50 dos 647 escritores, e o
carregamento do dataset estoura 32 GB de RAM.

**Testes de diagnóstico de GPU** — `diagnostico/`. Rode em qualquer placa nova
antes de treinar nela.

**Métrica de diacríticos** — `avaliacao_diacriticos/`, ver o `ESTADO.md` de lá.
Passos 1, 2, 3 e 6 prontos e verificados, 14 testes sintéticos passando. O
portão do Passo 1 passou: os gêmeos dos pares mínimos saem alinhados
(dx=dy=0, IoU 0,932), então o E1 por diferença de imagens é válido.

**Splits do BRESSAY** — `bressay_split/`. 74.882 amostras de treino, 647
escritores; train, val e test **disjuntos por escritor** (647/154/199,
sobreposição zero em todos os pares). Transcrições limpas: zero vazias, zero
acima de 40 caracteres, zero arquivos ausentes.

**Ambiente** — `flake.nix` com três shells: `rocm`, `cuda` e `cpu`.

---

## O que foi arquivado e por quê

Tudo em `~/repos/htg-tcc-arquivo/` (31 GB). **Movido, não apagado.**

| item | motivo |
|---|---|
| todos os modelos treinados | treinados na GPU defeituosa, não confiáveis |
| extrator de estilo do BRESSAY | idem |
| `saved_iam_data/` (duas cópias, 11,6 GB) | caches `.pt` que o código nunca lê — o `torch.load` está comentado no `style_encoder_train.py` |
| logs dos treinos | evidência dos diagnósticos, para documentar depois |
| 23 pastas de amostras e comparações | resultados dos runs defeituosos |

O repositório saiu de 45 GB para 7,6 GB.

---

## Como retomar, numa máquina nova

```bash
# 1. ambiente conforme a placa
nix develop .#cuda      # NVIDIA  (recomendado)
nix develop .#rocm      # AMD     (valide antes!)
nix develop .#cpu       # sem GPU (métrica e referência numérica)

# 2. VALIDE A PLACA ANTES DE QUALQUER TREINO
python diagnostico/teste_conv_isolada.py
CKPT=<um .pt do unet> python diagnostico/teste_lote_unet.py
# erros ~1e-6: pode treinar.  >=1e-2 ou erráticos: NÃO treine nesta placa.

# 3. clonar o DiffusionPen e aplicar as correções
#    (ver README.md, seção "Reprodução")
cp diffusionpen_mods/train.py                diffusionpen/DiffusionPen/
cp diffusionpen_mods/style_encoder_train.py  diffusionpen/DiffusionPen/
cp diffusionpen_mods/utils/bressay_dataset.py diffusionpen/DiffusionPen/utils/

# 4. baixar os pesos do DiffusionPen (huggingface.co/konnik/DiffusionPen)

# 5. treinar o extrator de estilo no BRESSAY (~20-25 epocas bastam;
#    depois disso ele overfita nos escritores vistos)
python diffusionpen/DiffusionPen/style_encoder_train.py \
  --dataset bressay --mode mixed --model mobilenetv2_100 \
  --epochs 25 --batch_size 64 --save_path ./style_models

# 6. fine-tune, em blocos com verificacao por geracao
./treinar_v2.sh
```

### Notas de quem já rodou

O `treinar_v2.sh` treina em blocos de 5 épocas e gera amostras ao fim de cada
um. Isso existe porque **o MSE não é sinal confiável de saúde**: já aconteceu
de ele melhorar (0,0522 → 0,0404) enquanto o modelo perdia completamente a
capacidade de gerar. O único teste que vale é gerar amostra e olhar.

A geração usa `gerar_amostras.py --ckpt <arquivo.pt> --out <pasta>`. Semente e
escritores são fixos, então checkpoints diferentes saem comparáveis.

Avalie pelas palavras com diacrítico (`ação`, `não`, `avó`, `coração`), não
pela legibilidade geral. O modelo do IAM puro gera "português" legível **sem o
circunflexo** e apaga til e cedilha ("não" → "no", "ação" → "aco") — é o
controle negativo, e é exatamente o que o trabalho quer superar.

---

## O que falta, em ordem

1. **Validar uma GPU.** A equipe tem uma RTX 3090 (Ampere, CUDA, 24 GB —
   caminho recomendado) e uma RX 9060 XT (RDNA4; é placa AMD de consumo, mesma
   classe de risco da RX 6600 XT, então **precisa passar pelo `diagnostico/`**).
2. **Re-treinar o extrator de estilo** no BRESSAY, na placa validada.
3. **Re-treinar o fine-tune** a partir do pré-treinado do IAM.
4. **Refazer o controle E1 do IAM puro** da métrica — o valor atual (0,141)
   veio de imagens geradas na placa defeituosa, em lote 4. Os números de
   recorte real (0,861) estão a salvo, porque não passam por geração.
5. **Fechar o E2** nos recortes reais e rodar o Passo 5 da métrica.

---

## Armadilhas já encontradas (para não repetir)

- **Nunca avalie um modelo pelo MSE.** Gere amostras.
- **Não confie em grade de amostra gerada dentro do treino.** O
  `sampling_loader` do DiffusionPen tokeniza com `max_length=200` enquanto o
  treino usa 40 — condicionamento diferente do treino. O `treinar_v2.sh`
  desliga essa amostragem e gera em processo separado.
- **`--pretrained_path` e `--load_check` são mutuamente exclusivos.** No
  `train.py` o bloco do `pretrained_path` roda **depois** do `load_check` e
  sobrescreve silenciosamente os pesos retomados.
- **`--load_check` é `type=bool` no argparse**, então `--load_check False`
  vira `True`. Só passe a flag quando quiser mesmo retomar.
- **O EMA precisa do `ema.step` restaurado ao retomar**, senão o `step_ema`
  copia os pesos do modelo por cima do EMA carregado e o destrói. O
  `estado.pt` cuida disso.
- **Escritores de train e val são disjuntos**, então a acurácia de validação
  do extrator de estilo é 0,0000 por construção — não é defeito. A seleção de
  checkpoint usa o triplet loss por causa disso.
