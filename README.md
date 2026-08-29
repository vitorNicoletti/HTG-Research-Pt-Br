# HTG-Research-Pt-Br

Fidelidade de diacríticos na geração de escrita manuscrita (HTG) em português brasileiro.

Projeto de Creative Experience: Transformative Project II — Ciência da Computação, PUCPR, Turma B, Equipe 13.
Vitor Nicoletti · Vinícius Y. Borges · Bruno H. O. M. Dutra · Leonardo Saito

---

## Problema

Modelos de HTG são treinados quase sempre em inglês. Quando se pede uma palavra
com diacrítico do português, o gerador falha — mas as métricas usuais (FID, KID,
CER agregado) não enxergam essa falha, porque diluem o erro sobre todo o alfabeto.

Este repositório contém a investigação empírica dessa falha e o desenvolvimento de
uma métrica de fidelidade estratificada por classe de caractere.

## Estado atual

| Fase | Situação |
|---|---|
| 0 — Ambiente (ROCm / GPU AMD) | concluída |
| 1 — *Smoke test* em inglês | concluída |
| 2 — Sonda de diacríticos | parcial (semente 0, 20 imagens) |
| 3 — Métrica + validação humana | em especificação |
| 4 — *Fine-tuning* em português | não iniciada |

## Resultado preliminar

`saidas/diffusionpen/folha_contato.png`

O grupo de controle ASCII sai legível. Os grupos acentuados exibem **dois modos de
falha distintos**:

- **Omissão limpa** — o diacrítico some e o resto da palavra sai correto:
  `café` → "cafe", `março` → "marco", `histórico` → "historico", `português` → "portugues".
- **Degradação da palavra inteira** — a palavra desmonta:
  `coração` → "ccrcemo", `praça` → "praxa", `põe` → "pae".

O par mínimo é o achado central: `nacao` é gerada corretamente e `nação` não, sob a
**mesma semente e o mesmo estilo**. Se o acento estivesse sendo filtrado na entrada,
as duas imagens seriam idênticas. Não são — logo o caractere chega ao modelo, e a
falha é de renderização, não de filtragem.

O controle de validade está em `saidas/diffusionpen/comparacao_real_vs_gerado.png`:
em inglês, as amostras geradas são quase indistinguíveis das reais do IAM.

## Estrutura

```
env/check_env.py              verificação de GPU/ROCm (aborta se cair para CPU)
comum/palavras.py             lista canônica da sonda (4 grupos, pares mínimos, 3 seeds)
comum/folha_contato.py        monta a folha de contato para inspeção visual
diffusionpen/smoke_test.py    Fase 1 — 5 palavras em inglês
diffusionpen/sonda_diacriticos.py   Fase 2 — 60 imagens da sonda
diffusionpen/patches_diffusionpen.diff   patch aditivo do DiffusionPen
saidas/                       imagens geradas, manifestos e folhas de contato
LOG.md                        diário de execução (reprodutibilidade)
ACHADOS.md                    achados técnicos consolidados
STATUS.md                     onde a execução parou e como retomar
```

## Reprodução

Pré-requisitos: GPU AMD com ROCm ≥ 7.0.2 (testado em RX 9060 XT / gfx1200) ou
GPU NVIDIA equivalente, e Python 3.12.

```bash
# 1. dependências
python3 -m venv venv-diffpen && source venv-diffpen/bin/activate
pip install -r requirements.txt

# 2. verificar a GPU antes de qualquer coisa
python env/check_env.py        # deve imprimir gfx1200 e sair com código 0

# 3. clonar o DiffusionPen e aplicar o patch
git clone --depth 1 https://github.com/koninik/DiffusionPen.git diffusionpen/DiffusionPen
cd diffusionpen/DiffusionPen && git apply ../patches_diffusionpen.diff && cd ../..

# 4. artefatos externos (não versionados — ver "Pesos e dados" abaixo)

# 5. executar
python diffusionpen/smoke_test.py            # Fase 1 (inspecionar as 5 imagens)
python diffusionpen/sonda_diacriticos.py     # Fase 2
python comum/folha_contato.py saidas/diffusionpen/sonda
```

Ambos os scripts aceitam `--dry-run`, que escreve o manifesto e imprime o comando
sem carregar o modelo.

## Pesos e dados (não versionados)

Não são commitados por tamanho (~7,4 GB). Procedência:

| Artefato | Origem | Destino |
|---|---|---|
| Pesos do DiffusionPen | `huggingface.co/konnik/DiffusionPen` | raiz de `diffusionpen/DiffusionPen/` |
| IAM `words.tgz` | `fki.tic.heia-fr.ch` (exige registro) | `diffusionpen/DiffusionPen/iam_data/words/` |
| VAE + scheduler do SD 1.5 | `huggingface.co/stable-diffusion-v1-5` | apontar com `--stable_dif_path` |

Arquivos necessários para amostragem: `saved_iam_data/train_word_IAM.pt`,
`diffusionpen_iam_model_path/models/{ema_ckpt.pt,ckpt.pt}`,
`style_models/iam_style_diffusionpen.pth`.

## O patch do DiffusionPen

`patches_diffusionpen.diff` é **aditivo**, e não altera nenhum caminho existente do
repositório original. Ele adiciona um `sampling_mode` chamado `sonda`, que lê um
manifesto JSON e gera cada item com **estilo e semente fixos**. Isso é necessário
porque o `single_sampling` original tem a lista de palavras fixa no código e sorteia
um estilo novo a cada palavra, o que inviabiliza a comparação por pares mínimos.

O patch também corrige três problemas encontrados no repositório original:
chamada a `save_single_images()`, função que não existe no repositório;
ausência de `map_location` ao carregar *checkpoints*, que quebra fora de CUDA; e
`torch.load` sem `weights_only=False`, incompatível com PyTorch ≥ 2.6 para os
`.pt` do IAM pré-processado.

