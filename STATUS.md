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

### Solução: **Docker** (revisada — supera a ideia de instalar outra distro)

O caminho documentado pela AMD em `github.com/ROCm/librocdxg` dispensa trocar de
distro. O host precisa apenas de **uma biblioteca**, não da pilha ROCm inteira.

Inspecionado o pacote `rocdxg-roct_1.2.2_amd64.deb` (181 KB, release de
2026-08-05): **não tem campo `Depends:`** e instala exatamente três coisas —
as mesmas que o container monta:

```
/opt/rocm/lib/librocdxg.so
/opt/rocm/share/rocdxg/dids.conf
/etc/ld.so.conf.d/x86_64-libhsakmt.conf
```

Ou seja: o problema do repositório ausente para `resolute` **deixa de existir**,
porque o host não precisa do apt da AMD. Toda a pilha ROCm + PyTorch vem dentro
da imagem, que é construída sobre Ubuntu 24.04 (uma distro suportada).

#### Passos

```bash
# 1. No host (Ubuntu 26.04 atual) - EXIGE SUDO, mas e um comando so
wget https://github.com/ROCm/librocdxg/releases/download/v1.2.2/rocdxg-roct_1.2.2_amd64.deb
sudo dpkg -i rocdxg-roct_1.2.2_amd64.deb

# 2. Docker precisa estar operacional dentro desta distro
#    O Docker Desktop esta instalado no Windows mas PARADO e sem integracao WSL.
#    Iniciar o Docker Desktop e marcar a integracao para a distro "Ubuntu",
#    ou instalar docker.io na distro (tambem exige sudo).

# 3. Rodar o container
docker run -it \
    -v /usr/lib/wsl/lib/libdxcore.so:/usr/lib/libdxcore.so \
    -v /opt/rocm/lib/librocdxg.so:/usr/lib/librocdxg.so \
    -v /opt/rocm/share/rocdxg/dids.conf:/usr/share/rocdxg/dids.conf \
    --device=/dev/dxg \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --ipc=host \
    --shm-size 8G \
    -v $HOME/htg-tcc:/workspace \
    rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.11.0
```

#### Imagem escolhida e por quê

`rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.11.0` (19.3 GB)

| critério | situação |
|---|---|
| ROCm 7.14 | ≥ 7.2.1 exigido para gfx1200 ✅ |
| ROCm ≥ 7.13 | dispensa `HSA_ENABLE_DXG_DETECTION=1` ✅ |
| Ubuntu 24.04 | distro suportada pela AMD ✅ |
| Python 3.12 | melhor cobertura de wheels da stack de ML ✅ |
| PyTorch 2.11 | recente, compatível com diffusers/transformers atuais ✅ |

Alternativa menor: `rocm7.2.4_ubuntu24.04_py3.12_pytorch_2.10.0_ORT_1.23.2` (10.7 GB).

#### O que ainda exige o orientando

- **`sudo dpkg -i`** do pacote rocdxg (1 comando)
- **Docker operacional** na distro (iniciar Docker Desktop + integração WSL, ou
  `apt install docker.io`)

#### Riscos a validar quando destravar

- Passagem de `/dev/dxg` através do Docker Desktop não foi testada nesta máquina.
- `torch 2.13.0+rocm7.2` instalado em `venv-diffpen` é o **wheel errado** para
  WSL; com o container ele deixa de ser usado (o venv pode ser descartado).

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
