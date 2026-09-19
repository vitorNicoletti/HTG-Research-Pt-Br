# STATUS — onde a execução parou e como retomar

**Atualizado:** 2026-08-19
**Fase atual:** 0 (ambiente) — **bloqueada**
**Bloqueios ativos:** 2, ambos dependem de ação do orientando

---

## Resumo em uma linha

A GPU existe e o driver está correto, mas o PyTorch não a enxerga: em WSL não
há `/dev/kfd`, e os wheels do pytorch.org exigem essa interface. Solução
identificada: container `rocm/pytorch` + o pacote `rocdxg-roct` no host (um
`sudo dpkg -i`, sem trocar de distro). Em paralelo, o DiffusionPen precisa do
dataset IAM, que exige registro — o orientando fará o download depois.

---

## Máquina de trabalho

| Item | Valor |
|---|---|
| Acesso | `ssh -p 2222 dead@100.100.155.123` (chave autorizada, funcionando) |
| Host | `DESKTOP-KKFR30E` — WSL2 sobre Windows 11 Home build 26200 |
| Distro WSL | Ubuntu 26.04 (resolute) |
| CPU / RAM / disco | i5-14600K · 15 GiB · 948 G livres |
| GPU | **AMD Radeon RX 9060 XT** (gfx1200) ✅ |
| Driver Windows | Adrenalin 26.7.1 (`32.0.31035.1003`) ✅ acima do mínimo |
| Diretório do projeto | `~/htg-tcc/` |

---

## O que já está pronto

### Na máquina remota (`~/htg-tcc/`)

- [x] Estrutura de diretórios e scripts transferidos por `scp`
- [x] `uv` 0.12.5 instalado em `~/.local/bin` (contorno: sem sudo, sem `ensurepip`)
- [x] venv `venv-diffpen` com **Python 3.12.14**
- [x] DiffusionPen clonado (`diffusionpen/DiffusionPen/`)
- [x] VATr++ clonado (`vatr/VATr-pp/`)
- [x] `patch_sonda_train.diff` aplicado e validado (`ast.parse` OK; branch
      `sonda` em `train.py:750`)
- [ ] ⚠ `torch 2.13.0+rocm7.2` instalado mas **é o wheel errado** — precisa ser
      substituído pelo da AMD (ver bloqueio 1)

### Análise estática (concluída, não depende de GPU)

- [x] Charset do DiffusionPen mapeado; CANINE confirmado empiricamente aceitando
      `ã`/`ç` como codepoints
- [x] Ausência de normalização de acentos confirmada
- [x] **Filtro silencioso do VATr++ descoberto** (`generate/writer.py:70`)
- [x] Mecanismo `special_alphabet` (grego) identificado como alavanca para PT-BR
- [x] `xformers`/`bitsandbytes` confirmados ausentes — patch previsto no plano
      é desnecessário

### Scripts prontos e testados (sem GPU)

| Arquivo | Estado |
|---|---|
| `env/check_env.py` | ✅ executado no remoto; abortou corretamente |
| `comum/palavras.py` | ✅ testado — 60 imagens (4 grupos × 3 seeds) |
| `comum/folha_contato.py` | ✅ testado com imagens sintéticas |
| `diffusionpen/smoke_test.py` | ✅ dry-run OK |
| `diffusionpen/sonda_diacriticos.py` | ✅ dry-run OK, manifesto gerado |
| `vatr/*` | ❌ não escrito ainda (Fase 4) |
| `*/perfil_vram.py` | ❌ não escrito ainda (Fase 3) |

---

## BLOQUEIO 1 — ROCm não funciona nesta distro do WSL

### Diagnóstico

```
$ ./venv-diffpen/bin/python env/check_env.py
W agent.cpp:608] sysfs nodes path '/sys/class/kfd/kfd/topology/nodes' does not exist
torch: 2.13.0+rocm7.2      hip/rocm: 7.2.53211
cuda disponivel: False
FALHA: GPU nao visivel ao PyTorch          (exit 1)
```

Duas causas independentes:

1. **WSL não tem KFD.** Sem módulo `amdgpu`; a GPU chega por `/dev/dxg`. Os
   wheels do pytorch.org são compilados contra a interface KFD nativa. Não há
   configuração que contorne.
2. **Ubuntu 26.04 não tem repositório ROCm.** `resolute` → HTTP 404 em
   `repo.radeon.com` (verificado em `rocm/apt/{latest,7.2.4,7.2.3,7.2.1}` e em
   `amdgpu/latest/ubuntu`). Só `noble` (24.04) e `jammy` (22.04) existem.

### Solução adotada: **ROCm via pip, sem Docker** (revisão 2)

A imagem `rocm/pytorch` de 19,3 GB foi **descartada**. Inspeção do config da
imagem revelou a causa do tamanho:

```
AMDGPU_FAMILY=device-all
INDEX_URL=https://repo.amd.com/rocm/whl-multi-arch
```

`device-all` = kernels de GPU pré-compilados para **todas** as arquiteturas AMD
(Instinct gfx90a/942/950, RDNA2, RDNA3, APUs Strix...). Uma única camada da
imagem tem 18,82 GB. Precisamos de exatamente uma arquitetura: gfx1200.

A AMD publica índices **por família** em `repo.amd.com/rocm/whl/`. O nosso é
`gfx120X-all` (RDNA4). Tamanhos medidos individualmente:

| pacote | tamanho |
|---|---|
| `torch-2.9.1+rocm7.13.0` (cp312) | 340 MB |
| `rocm_sdk_libraries_gfx120x_all` | 1062 MB |
| `rocm_sdk_core` | 414 MB |
| `pytorch_triton_rocm` | 320 MB |
| `torchvision-0.26.0` | 1 MB |
| **total** | **≈ 2,2 GB** (9× menor) |

**Consequência importante:** `rocm-sdk-core` e `rocm-sdk-libraries` são wheels
Python. O ROCm vem pelo **pip**, não pelo apt — logo o repositório ausente para
`resolute` (Ubuntu 26.04) **deixa de ser um problema**, e o Docker torna-se
desnecessário. Sem Docker Desktop, sem integração WSL, sem segunda distro.

#### Passos

```bash
# 1. Unico passo com sudo (181 KB, sem dependencias) - FEITO em 2026-08-20
wget https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-roct_1.2.2_amd64.deb
sudo dpkg -i rocdxg-roct_1.2.2_amd64.deb

# 2. Resto em user-space
uv venv --clear --python 3.12 venv-diffpen
uv pip install --python ./venv-diffpen/bin/python \
  torch torchvision pytorch-triton-rocm \
  --index-url https://repo.amd.com/rocm/whl/gfx120X-all/
```

Verificado no host em 2026-08-20:

```
ii  rocdxg-roct  1.2.2  amd64  ROCDXG runtime libraries
OK  /opt/rocm/lib/librocdxg.so
OK  /opt/rocm/share/rocdxg/dids.conf
OK  /etc/ld.so.conf.d/x86_64-libhsakmt.conf
OK  /dev/dxg
```

#### Risco assumido

Este caminho (ROCm via pip + `librocdxg` no host) **não é o documentado pela
AMD** — a documentação oficial descreve o fluxo Docker. É dedução a partir da
organização dos pacotes. Ponto de falha possível: o `librocdxg` localizar o
ROCm dentro do `site-packages` em vez de `/opt/rocm`.

Fallback caso falhe: Docker com a imagem menor
`rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_2.10.0_ORT_1.23.2` (10,7 GB),
com os mounts documentados em `github.com/ROCm/librocdxg`.

### Critério de destravamento
Dentro do container: `python env/check_env.py` imprimir `gfx1200` e ~16 GB de
VRAM, com exit 0.

### Alternativa descartada (registrada)

Instalar `Ubuntu-24.04` como segunda distro WSL também funcionaria — está
disponível em `wsl.exe -l -o`. Foi preterida: exige instalar distro, ROCm
completo, sshd e reconfigurar acesso, contra um único `dpkg -i` no caminho
Docker.

---

## BLOQUEIO 2 — dataset IAM (aguardando, sem pressa)

O orientando informou que **fará o download depois**.

### Por que é necessário

`train.py:338-345`, dentro de `Diffusion.sampling()`:

```python
root_path = './iam_data/words'
for im_idx, random_f in enumerate(five_styles):
    file_path = os.path.join(root_path, random_f[0])
    img_s = Image.open(file_path).convert('RGB')
```

As 5 imagens de estilo few-shot são lidas **do disco**. Os caminhos vêm de
`utils/splits_words/iam_train_val.txt` (já no repo), mas os PNGs não.

`--img_feat False` existe, porém faz `style_images = None` e
`style_features = None`: gera sem condicionamento de estilo. Não é o modo
few-shot que a Fase 1 especifica, e o modelo foi treinado com o condicionamento.
**Não usar como substituto.**

### O que baixar

1. **IAM** — `words.tgz` (~1.2 GB), registro em
   https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
   Extrair para `~/htg-tcc/diffusionpen/DiffusionPen/iam_data/words/`
   (estrutura `a01/a01-000u/a01-000u-00-00.png`)

2. **Pesos do DiffusionPen** — https://huggingface.co/konnik/DiffusionPen
   (~7.4 GB dos ~10 GB totais; o resto não é usado em sampling)

   | arquivo | tamanho | necessário |
   |---|---|---|
   | `saved_iam_data/train_word_IAM.pt` | 5.30 G | sim |
   | `diffusionpen_iam_model_path/models/ema_ckpt.pt` | 0.68 G | sim |
   | `diffusionpen_iam_model_path/models/ckpt.pt` | 0.68 G | sim |
   | `style_models/iam_style_diffusionpen.pth` | 0.01 G | sim |
   | `saved_iam_data/test_word_IAM.pt` | 1.93 G | não |
   | `diffusionpen_iam_model_path/models/optim.pt` | 1.35 G | não |

   Colocar as pastas na raiz de `diffusionpen/DiffusionPen/`.

3. **Stable Diffusion v1.5** — subpastas `vae` e `scheduler` de
   https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5
   Apontar com `--stable_dif_path`.

---

## O que falta, em ordem

| # | Tarefa | Depende de |
|---|---|---|
| 1 | `dpkg -i rocdxg-roct` + Docker operacional | bloqueio 1 |
| 2 | `check_env.py` passar com `gfx1200` | 1 |
| 3 | Baixar IAM + pesos + VAE do SD1.5 | bloqueio 2 |
| 4 | **Fase 1** — smoke test em inglês (5 palavras) | 2, 3 |
| 5 | **Fase 2** — sonda de diacríticos (60 imagens) + folha de contato | 4 |
| 6 | **Fase 3** — escrever e rodar `perfil_vram.py` | 4 |
| 7 | **Fase 4** — VATr++: instalar, smoke, sonda, perfil | 2 |
| 8 | Consolidar `ACHADOS.md` com resultados visuais | 5, 6, 7 |

---

## Decisões pendentes do orientando

1. **Ambiente:** caminho Docker (recomendado — um `sudo dpkg -i` + container),
   segunda distro Ubuntu 24.04, ou migrar para Kaggle/Colab? (A Fase 3 pede
   perfil de VRAM da própria 9060 XT, o que só as opções locais entregam.)
2. **VATr++ / Fase 4:** o filtro de `writer.py:70` descarta os diacríticos antes
   da geração. Corrigi-lo altera o que é pedido ao modelo — não é patch de
   compatibilidade. **Não será mexido sem autorização explícita** (regra 4).
3. **VATr++ / versões:** o repo fixa PyTorch 1.13.1 + CUDA 11.7. Não existe
   equivalente ROCm. Vai precisar rodar em PyTorch 2.x — reportar antes de forçar.

---

## Armadilhas já encontradas (para não repetir)

- Timeout de SSH **não mata o processo remoto**: duas instâncias de `uv pip
  install` ficaram travadas no lock do cache. Usar `setsid` + polling.
- `python3 -m venv` falha na Ubuntu 26.04 (sem `ensurepip`); `python3.14-venv`
  exigiria sudo. `uv` resolve sem privilégio.
- Não há wheel `cp314` do PyTorch no repo da AMD — fixar **Python 3.12**.
- `sudo` no host remoto **pede senha** — nada que exija privilégio pode ser
  automatizado por mim.
