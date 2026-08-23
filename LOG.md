# LOG de execução — HTG em GPU AMD

Diário de execução para a seção de reprodutibilidade do TCC.

---

## 2026-08-17 — FASE 0: Ambiente

### Estado: **BLOQUEADA** — hardware alvo ausente na máquina inspecionada

### Máquina inspecionada

| Item | Valor |
|---|---|
| Host | `archlinux` |
| Modelo | Samsung 550XBE/350XBE (notebook, chassis type 10) |
| SO | Arch Linux (rolling), kernel `6.19.8-arch1-1` |
| WSL | Não (`/proc/version` sem menção a microsoft) |
| CPU | Intel Core i5-8265U @ 1.60GHz (4C/8T, Whiskey Lake-U) |
| RAM | 15 GiB |
| Disco | `/` 219G (177G livres) · `/home` 916G (823G livres) |
| Python do sistema | 3.14.3 |

### GPU detectada

```
$ lspci -nn | grep -Ei 'vga|3d|display'
00:02.0 VGA compatible controller [0300]: Intel Corporation WhiskeyLake-U GT2 [UHD Graphics 620] [8086:3ea0] (rev 02)
```

- Total de dispositivos VGA/3D no barramento: **1**
- Dispositivos AMD/ATI (`1002:`) no PCI: **nenhum**
- `/sys/class/drm/card1/device/driver` → `i915` (driver Intel)
- Sem Thunderbolt (`/sys/bus/thunderbolt/devices/` inexistente) → eGPU externa não é possível nesta máquina

### ROCm

```
$ which rocminfo   → not found
$ which rocm-smi   → not found
$ ls /opt/rocm*    → inexistente
$ pacman -Qs rocm  → nenhum pacote
$ lsmod | grep amdgpu → não carregado
$ journalctl -k -b | grep -i amdgpu → sem menção
```

### Conclusão da Fase 0

A RX 9060 XT (gfx1200) **não está presente nesta máquina**. A única GPU é a
iGPU Intel UHD 620, que não é suportada por ROCm. Não se trata de ROCm
ausente/desatualizado — é ausência de hardware.

Conforme a regra 2 do protocolo (*nunca aceitar fallback silencioso para CPU*),
a execução foi interrompida. Nenhum pacote foi instalado.

**Pendência para o orientando:** confirmar em qual máquina a RX 9060 XT está
instalada. Este notebook aparenta ser a estação de trabalho de edição, não a de
treino.

### Divergência do plano

O plano previa Ubuntu 22.04/24.04 ou WSL2. A máquina inspecionada é **Arch
Linux**. Se a máquina de treino também for Arch, o caminho de instalação do ROCm
muda (ver seção de instalação abaixo) — o ROCm oficial da AMD não publica
pacotes para Arch; usa-se o repositório `extra` da distro.

---

## Verificações independentes de hardware (feitas em 2026-08-17)

Estas checagens não dependem da GPU e ficam validadas para quando o hardware
estiver disponível.

### PyTorch + ROCm — índices de wheels disponíveis

Sondagem direta em `download.pytorch.org` (não de memória):

| Índice | torch mais recente |
|---|---|
| `https://download.pytorch.org/whl/rocm6.4` | 2.8.0+rocm6.4 |
| `https://download.pytorch.org/whl/rocm7.0` | 2.10.0+rocm7.0 |
| `https://download.pytorch.org/whl/rocm7.1` | 2.10.0+rocm7.1 |
| **`https://download.pytorch.org/whl/rocm7.2`** | **2.11.0+rocm7.2** |
| `https://download.pytorch.org/whl/rocm7.3` | não existe (0 wheels) |

Wheels ROCm 7.2 existem para cp310–cp315. Comando previsto:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
```

> ROCm ≥ 7.0.2 é o requisito para gfx1200 conforme o plano; 7.2 atende com folga.
> A versão do runtime ROCm do sistema deve casar com a do wheel.

### VATr++ — URL do repositório **corrigida**

O plano instruía procurar VATr++ na organização `aimagelab`. **Não está lá.**

- `https://github.com/aimagelab/VATr` → é o **VATr** (CVPR 2023), não o VATr++.
  Requisitos declarados: Python 3.9, PyTorch 1.13.1, CUDA 11.7.
- `https://github.com/EDM-Research/VATr-pp` → **é o VATr++**. Requisitos
  declarados: Python 3.9, PyTorch 1.13.1 / torchvision 0.14.1 / torchaudio
  0.13.1, CUDA 11.7. Pesos: `resnet_18_pretrained.pth` via Google Drive
  (pré-treino em Font Square).
- `https://huggingface.co/blowing-up-groundhogs/vatrpp` → implementação HF do
  VATr++ (alternativa, referenciada pelo README do VATr).

**Risco antecipado (Fase 4):** o VATr++ fixa PyTorch 1.13.1/CUDA 11.7. Não
existe wheel ROCm 7.x para PyTorch 1.13.1 — a combinação declarada é
incompatível com gfx1200. Provável necessidade de rodar o VATr++ em PyTorch
2.x. Conforme a regra 4, isso será reportado antes de qualquer downgrade/upgrade
forçado.

### Instalação do ROCm na distro detectada (Arch Linux)

Caso a máquina de treino seja Arch, o ROCm vem do repositório oficial `extra`
(a AMD não publica `.deb`/`.rpm` para Arch):

```bash
sudo pacman -S rocm-hip-sdk rocm-opencl-sdk rocminfo rocm-smi-lib
sudo usermod -aG render,video "$USER"   # exige relogin
```

Se for Ubuntu 22.04/24.04 (caminho previsto no plano), usar o instalador
`amdgpu-install` da AMD apontando para ROCm ≥ 7.0.2.

Verificação pós-instalação: `rocminfo | grep gfx` deve imprimir `gfx1200`.

---

## 2026-08-17 — Análise estática dos geradores (sem GPU)

GPU confirmada indisponível pelo orientando. Prosseguiu-se com tudo que é
leitura de código e preparação — **nenhum modelo foi executado**.

### Repositórios clonados

| Repo | Commit | Comando |
|---|---|---|
| DiffusionPen | `--depth 1` de `main` | `git clone --depth 1 https://github.com/koninik/DiffusionPen.git` |
| VATr++ | `--depth 1` de `main` | `git clone --depth 1 https://github.com/EDM-Research/VATr-pp.git` |

### ACHADO 1 — DiffusionPen: charset ASCII, mas fora do caminho de sampling

`DiffusionPen/letter2index.json` e `train.py:32` definem um charset de **80
caracteres, ASCII puro**, sem nenhum diacrítico:

```python
c_classes = '_!"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '
```

Esse charset alimenta `label_padding()` (`train.py:37-47`), que faz
`letter2index[i]` — **lookup direto de dict, que levanta `KeyError` em `ã`**.

Porém `label_padding` só é chamado em `train.py:205` e `train.py:464`, ambos no
caminho de **treino** / `sampling_loader`. O caminho de sampling unitário
(`Diffusion.sampling`, `train.py:255`) tokeniza **apenas** via CANINE
(`train.py:263`) e nunca toca `letter2index`.

**Consequência:** a sonda da Fase 2 **não vai crashar** — falhará em silêncio,
exatamente como o plano hipotetizou. E o `KeyError` vai aparecer na fase
seguinte do TCC (fine-tuning em português), onde `label_padding` é usado.

### ACHADO 2 — CANINE aceita diacríticos (confirmado empiricamente)

`train.py:653` usa `CanineTokenizer.from_pretrained("google/canine-c")`.
Verificado com `transformers` 5.15.0 em venv CPU (sem torch):

| palavra | nº tokens | ids |
|---|---|---|
| `nacao` | 5 | `[110, 97, 99, 97, 111]` |
| `nação` | 5 | `[110, 97, 231, 227, 111]` |
| `coracao` | 7 | `[99, 111, 114, 97, 99, 97, 111]` |
| `coração` | 7 | `[99, 111, 114, 97, 231, 227, 111]` |

Cada caractere vira **o próprio codepoint Unicode** (`ã`→227, `ç`→231, `é`→233,
`ê`→234, `õ`→245). Sem OOV, sem erro, sem normalização. Os pares mínimos têm
comprimento idêntico e diferem só nas posições do diacrítico — condição ideal
para a comparação da Fase 2.

Comando de reprodução:
```bash
python -m venv venv-tok && ./venv-tok/bin/pip install transformers
# CanineTokenizer.from_pretrained("google/canine-c"); tok(w, add_special_tokens=False)
```

### ACHADO 3 — Nenhuma normalização de acentos no DiffusionPen

Busca por `unidecode`, `unicodedata`, `NFKD`, `NFD`, `encode('ascii')`:
**nenhuma ocorrência**. Os hits de "ascii" são o nome do diretório do IAM
(`iam_data/ascii/words.txt`) e um parâmetro de encoding de leitura de arquivo —
não removem acento. O plano pedia para reportar imediatamente se houvesse; **não há**.

### ACHADO 4 — VATr++ descarta caracteres desconhecidos silenciosamente ⚠

`VATr-pp/generate/writer.py:70`:

```python
text = "".join([c for c in text if c in self.model.args.alphabet])
```

E o alphabet padrão (`util/misc.py:561`, `train.py:36`) é o do IAM, sem diacríticos:

```
'Only thewigsofrcvdampbkuq.A-210xT5\'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%'
```

**`mão` vira `mo`. `coração` vira `corao`. `março` vira `maro`.** Sem aviso,
sem erro. Há um filtro equivalente em `models/model.py:267`.

Este é um modo de falha **diferente e pior** que o do DiffusionPen: o caractere
não é mal desenhado, ele **desaparece antes de chegar ao modelo**. Uma sonda
ingênua registraria "gerou sem erro" e a imagem teria uma palavra mais curta.

**Implicação metodológica:** comparar os dois modelos exige contabilizar isto.
Sem o ajuste, o VATr++ não está "errando o diacrítico" — não está sequer sendo
solicitado a desenhá-lo.

### ACHADO 5 — VATr++ tem mecanismo nativo para caracteres não vistos

`util/misc.py:508` define `special_alphabet` com o **alfabeto grego**:

```
'ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσςΤτΥυΦφΧχΨψΩω'
```

Ele é concatenado ao alphabet em dois pontos-chave:
- `models/model.py:123` → o `UnifontModule` (query embedding)
- `models/model.py:219` → o `strLabelConverter`

E `model.py:331` gera o `special_alphabet` explicitamente para visualizar
caracteres nunca vistos no treino.

O detalhe decisivo está em `models/unifont_module.py:29-42`: os arquétipos são
lidos de um pickle indexado por `ord(char)` e projetados por uma **`nn.Linear`
compartilhada** — não há embedding por caractere. Ou seja, **acrescentar um
caractere ao alfabeto não cria parâmetros novos**: basta o glifo existir no
Unifont. O grego é a prova de conceito disso no próprio repo.

**Esta é a alavanca do TCC.** A hipótese do "prior geométrico" tem um caminho de
implementação concreto: colocar `ãõçéêáóú` em `special_alphabet`. O DiffusionPen
não tem análogo — o CANINE dá representação de caractere, mas nenhuma informação
de forma do glifo.

> Pendência: `files/unifont.pickle` **não está no repo** (vem do Google Drive).
> Confirmar que ele cobre U+00E0–U+00FA antes de contar com isso. O GNU Unifont
> cobre Latin-1 Supplement, então é esperado que sim, mas não foi verificado.

### ACHADO 6 — `xformers` e `bitsandbytes`: não são problema

Busca por `xformers`, `bitsandbytes`, `memory_efficient` nos dois repos:
**nenhuma ocorrência**. O patch de atenção antecipado no passo 3 da Fase 1
**não será necessário**. O DiffusionPen usa `diffusers` (`AutoencoderKL`,
`DDIMScheduler`), que já cai em SDPA por padrão.

### Patch aplicado — DiffusionPen `train.py`

**Aditivo, não altera nenhum caminho existente** (regra 4). Diff completo em
`diffusionpen/patch_sonda_train.diff` (68 linhas).

1. Três argumentos novos: `--sonda_manifest`, `--sonda_out`, `--sonda_style`.
2. Um `sampling_mode` novo, `'sonda'`, que lê um manifesto JSON e gera cada item
   com **estilo fixo** e **seed controlada**.

Motivo: o `single_sampling` original tem a lista de palavras hardcoded
(`x_text = ['text', 'word']`) e sorteia o estilo a cada palavra
(`random.randint(0, 339)`) — inutilizável para uma sonda controlada, onde o
estilo precisa ser constante entre palavras para o par mínimo significar algo.

Isolamento das variáveis no branch novo:
- `random.seed(args.sonda_style)` antes de cada chamada → as 5 imagens de estilo
  few-shot sorteadas em `sampling()` são sempre as mesmas;
- `torch.manual_seed(seed)` → varia só o ruído inicial da difusão.

Sintaxe validada com `ast.parse`. **Não executado** (sem GPU).

---

## Artefatos criados nesta sessão

- Estrutura de diretórios: `env/`, `comum/`, `diffusionpen/`, `vatr/`, `saidas/{diffusionpen,vatr}/`
- `env/check_env.py` — verificação da Fase 0 (não executável aqui: sem GPU)
- `comum/palavras.py` — lista canônica da sonda. **Testado.** 4 grupos, 20
  palavras únicas, 3 seeds = 60 imagens. Compartilhado entre os dois modelos
  porque a Fase 4 exige palavras idênticas.
- `comum/folha_contato.py` — folha de contato. **Testado** com imagens
  sintéticas (51 presentes / 9 ausentes) — grid, rótulos acentuados e células
  "ausente" verificados visualmente.
- `diffusionpen/smoke_test.py` — Fase 1. Dry-run testado.
- `diffusionpen/sonda_diacriticos.py` — Fase 2. Dry-run testado, manifesto de 60 itens gerado.
- `diffusionpen/patch_sonda_train.diff` — patch para o registro.

> Desvio da estrutura do plano: foi criado `comum/` para a lista de palavras e a
> folha de contato. O plano previa duplicá-las em `diffusionpen/` e `vatr/`, mas
> a Fase 4 exige as **mesmas** palavras nos dois modelos — duplicar convida a
> divergência silenciosa que invalidaria a comparação.

### Notas sobre `check_env.py`

Além do especificado no plano, o script:
- aborta se `torch.version.hip` for `None` (pega o caso de instalar wheel CUDA/CPU por engano);
- imprime `props.gcnArchName`, que é o campo onde aparece `gfx1200` (o
  `get_device_name()` retorna o nome comercial, não a arquitetura);
- verifica `isfinite` no resultado do matmul fp16 — em arquiteturas recém
  suportadas, kernel quebrado costuma produzir NaN em vez de erro;
- testa `scaled_dot_product_attention`, que substituirá o `xformers` no patch da Fase 1.

### Notas sobre `check_env.py`

Além do especificado no plano, o script:
- aborta se `torch.version.hip` for `None` (pega o caso de instalar wheel CUDA/CPU por engano);
- imprime `props.gcnArchName`, que é o campo onde aparece `gfx1200` (o
  `get_device_name()` retorna o nome comercial, não a arquitetura);
- verifica `isfinite` no resultado do matmul fp16 — em arquiteturas recém
  suportadas, kernel quebrado costuma produzir NaN em vez de erro;
- testa `scaled_dot_product_attention`, que substituirá o `xformers` no patch da Fase 1.

---

## 2026-08-19 — FASE 0 na máquina com a RX 9060 XT (via SSH)

Acesso liberado pelo orientando via `ssh-copy-id`. Host: `dead@100.100.155.123:2222`.

### Máquina remota

| Item | Valor |
|---|---|
| Host | `DESKTOP-KKFR30E` |
| Ambiente | **WSL2** (kernel `6.18.33.2-microsoft-standard-WSL2`) |
| Distro WSL | **Ubuntu 26.04 (Resolute Raccoon)** |
| Windows | Windows 11 Home, build 26200 |
| CPU | Intel Core i5-14600K |
| RAM (dentro do WSL) | 15 GiB |
| Disco | 1007G, 954G livres |
| Python do sistema | 3.14.4 (único; sem `python3-venv`, sem `ensurepip`) |
| `sudo` | **exige senha** — não disponível de forma não-interativa |

### GPU

```
$ powershell.exe Get-CimInstance Win32_VideoController
Intel(R) UHD Graphics 770
AMD Radeon RX 9060 XT          <-- alvo, presente
```

- Driver AMD: `32.0.31035.1003`, de 23/07/2026 → **Adrenalin 26.7.1**
- `/dev/dxg` presente (paravirtualização de GPU do WSL)
- `/dev/kfd` e `/dev/dri` **ausentes** — esperado no WSL: ROCm ali não usa a
  interface KFD nativa, e sim o caminho DXG/ROCDXG.
- `/usr/lib/wsl/lib/` contém apenas `libd3d12.so`, `libd3d12core.so`,
  `libdxcore.so`. **Nenhuma lib HSA/ROCm injetada pelo driver.**
- `C:\Windows\System32\lxss\lib\` vazio.

### Requisitos de ROCm em WSL para RDNA4 (pesquisado, não de memória)

Da matriz de compatibilidade da AMD (*Use ROCm on Radeon and Ryzen*, WSL):

- RX 9060 XT (gfx1200) **é suportada** sob WSL2 — a partir do ROCm 7.2.1.
- Distros WSL suportadas: **Ubuntu 24.04.2** e **Ubuntu 22.04 LTS**.
- Driver Windows exigido: Adrenalin 26.1.1+ para WSL2 → o instalado (26.7.1) atende.
- PyTorch com suporte oficial de produção nessa combinação: **2.9.1**.

### ⚠ Descompasso de distro

Repositórios apt da AMD (`https://repo.radeon.com/rocm/apt/latest/dists/`):

| codinome | Ubuntu | HTTP |
|---|---|---|
| `jammy` | 22.04 | 200 |
| `noble` | 24.04 | 200 |
| **`resolute`** | **26.04** | **404** |
| `plucky` | 25.04 | 404 |
| `questing` | 25.10 | 404 |

Versões ROCm publicadas: até **7.2.4** (7.2.1+ é o necessário para gfx1200).

**A distro instalada (26.04 / resolute) não tem repositório ROCm da AMD.** O
driver Windows está em ordem; o problema é o lado Linux.

### Contorno de ambiente (sem sudo)

`python3 -m venv` falha (`ensurepip` ausente) e instalar `python3.14-venv`
exigiria senha de sudo. Usou-se **`uv` 0.12.5** (instalação local em
`~/.local/bin`, sem sudo), que também fornece um Python próprio:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 venv-diffpen      # Python 3.12.14
```

Escolhido 3.12 em vez do 3.14 do sistema: a stack de ML (diffusers, timm,
transformers em versões compatíveis) tem cobertura de wheels muito melhor.

### Teste em andamento

Instalando `torch`/`torchvision` do índice `rocm7.2` do pytorch.org para
responder empiricamente: **os wheels do PyTorch bastam sozinhos no WSL, ou é
mesmo necessário o ROCm de sistema (que não tem repo para 26.04)?**

### Resultado: PyTorch instalado, GPU **não** visível

```
$ uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
+ torch==2.13.0+rocm7.2
+ torchvision==0.28.0+rocm7.2
+ triton-rocm==3.7.1
INSTALL_EXIT=0

$ ./venv-diffpen/bin/python env/check_env.py
W agent.cpp:608] sysfs nodes path '/sys/class/kfd/kfd/topology/nodes' does not exist
torch: 2.13.0+rocm7.2
hip/rocm: 7.2.53211
cuda (build): None
cuda disponivel (API ROCm usa o mesmo nome): False
FALHA: GPU nao visivel ao PyTorch
EXIT=1
```

`check_env.py` cumpriu seu papel: abortou com código 1 em vez de cair para CPU.

**Diagnóstico.** Os wheels do pytorch.org são compilados contra a interface
**KFD nativa** (`/sys/class/kfd/kfd/topology/nodes`), que **não existe no WSL** —
lá o caminho é DXG/ROCDXG. Não é questão de configuração nem de
`HSA_OVERRIDE_GFX_VERSION`: é uma interface de kernel ausente. Os wheels
genéricos ROCm **não funcionam em WSL**, ponto.

O caminho suportado exige, em conjunto:
1. Ubuntu **24.04** ou **22.04** no WSL (não há repo para 26.04);
2. ROCm instalado via `amdgpu-install --usecase=wsl,rocm --no-dkms` (**exige sudo**);
3. wheels do PyTorch de `repo.radeon.com/rocm/manylinux/rocm-rel-7.2.x/`
   (torch 2.7.1 / 2.8.0 / 2.10.0 / 2.11.0, cp39–cp313 — **não há cp314**).

### Estado da Fase 0: **BLOQUEADA** (segunda vez, causa diferente)

| Requisito | Estado |
|---|---|
| GPU presente | ✅ RX 9060 XT |
| Driver Windows | ✅ Adrenalin 26.7.1 (≥ 26.1.1) |
| `/dev/dxg` | ✅ presente |
| Distro WSL suportada | ❌ 26.04 (precisa 24.04 ou 22.04) |
| ROCm runtime WSL | ❌ não instalado (exige sudo) |
| PyTorch com backend WSL | ❌ instalado o wheel errado (KFD) |

`wsl.exe -l -o` confirma que **`Ubuntu-24.04` está disponível para instalação**.

### Bloqueio adicional — DiffusionPen exige o IAM bruto

`train.py:338-345`, dentro de `Diffusion.sampling()`:

```python
root_path = './iam_data/words'
for im_idx, random_f in enumerate(five_styles):
    file_path = os.path.join(root_path, random_f[0])
    img_s = Image.open(file_path).convert('RGB')
```

As 5 imagens de estilo few-shot são lidas **do disco**. Os caminhos vêm de
`utils/splits_words/iam_train_val.txt` (presente no repo), mas os PNGs do IAM
**não** — e o IAM exige registro na FKI/Uni Bern.

`--img_feat False` faz `style_images = None` e `style_features = None`: o modelo
roda sem condicionamento de estilo. Não é o "few-shot de 5 amostras" que a Fase 1
especifica, e o modelo foi treinado com esse condicionamento. **Não é
substituto** — reportado, não aplicado (regra 4).

Pesos no HF `konnik/DiffusionPen` (~10 GB no total):

| arquivo | tamanho | necessário p/ sampling? |
|---|---|---|
| `saved_iam_data/train_word_IAM.pt` | 5.30G | sim (carregado no startup) |
| `saved_iam_data/test_word_IAM.pt` | 1.93G | não |
| `diffusionpen_iam_model_path/models/optim.pt` | 1.35G | não (só retomar treino) |
| `.../ema_ckpt.pt` | 0.68G | sim |
| `.../ckpt.pt` | 0.68G | sim |
| `style_models/iam_style_diffusionpen.pth` | 0.01G | sim |

Faltam ainda: VAE + scheduler do `stable-diffusion-v1-5`, e o `iam_data/words/`.

### Feito no remoto apesar dos bloqueios

- `~/htg-tcc/` criado, scripts transferidos via `scp`
- DiffusionPen e VATr++ clonados
- `patch_sonda_train.diff` aplicado (`git apply --check` OK, `ast.parse` OK,
  branch `sonda` em `train.py:750`)
- Filtro silencioso do VATr++ (`writer.py:70`) reconfirmado no host remoto
- `uv` 0.12.5 em `~/.local/bin`, venv `venv-diffpen` com Python 3.12.14

### Incidente registrado

Duas instâncias de `uv pip install` ficaram travadas no lock do cache: a
primeira sobreviveu ao timeout do meu SSH (o processo remoto não morre junto) e
a segunda, lançada com `nohup`, ficou esperando o lock. Cache parou de crescer em
5.9G. Resolvido com `pkill` e relançamento único usando `setsid`.
**Lição:** timeout de SSH não mata o processo remoto.

---

## 2026-08-20 — FASE 0 CONCLUÍDA ✅

### Solução: ROCm via pip, sem Docker

Descartada a imagem `rocm/pytorch` (19,3 GB). Inspeção do config revelou
`AMDGPU_FAMILY=device-all` + `INDEX_URL=.../whl-multi-arch`: kernels
pré-compilados para **todas** as arquiteturas AMD (uma camada de 18,82 GB).

A AMD publica índices por família em `repo.amd.com/rocm/whl/`. O nosso é
`gfx120X-all` (RDNA4). Total ≈ **2,2 GB**, 9× menor.

Como `rocm-sdk-core` e `rocm-sdk-libraries` são wheels Python, o ROCm vem pelo
pip — o repositório apt ausente para `resolute` **deixa de importar**, e o
Docker torna-se desnecessário.

```bash
# unico passo com sudo (feito pelo orientando)
sudo dpkg -i rocdxg-roct_1.2.2_amd64.deb    # 181 KB, sem dependencias

uv venv --clear --python 3.12 venv-diffpen
uv pip install --python ./venv-diffpen/bin/python \
  torch torchvision pytorch-triton-rocm \
  --index-url https://repo.amd.com/rocm/whl/gfx120X-all/
```

### Percalço: `libgomp.so.1` ausente

```
ImportError: libgomp.so.1: cannot open shared object file
```

A instalação do WSL é mínima e não traz o runtime OpenMP. Resolvido **sem sudo**
(`apt-get download` não exige root):

```bash
cd ~/htg-tcc/syslibs && apt-get download libgomp1 && dpkg -x libgomp1_*.deb ./root
export LD_LIBRARY_PATH="$HOME/htg-tcc/syslibs/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"
```

> **Esse `LD_LIBRARY_PATH` é obrigatório em toda execução.** Precisa ser embutido
> nos scripts, senão o import do torch quebra.

### Resultado

```
torch: 2.11.0+rocm7.13.0
hip/rocm: 7.13.99004
cuda disponivel (API ROCm usa o mesmo nome): True
device: AMD Radeon RX 9060 XT
arquitetura: gfx1200
VRAM total (GB): 16.97
matmul fp16 OK, pico VRAM (GB): 0.23
bf16 OK
scaled_dot_product_attention OK

AMBIENTE OK          (exit 0)
```

Todos os critérios de aceitação atendidos.

### Aviso a considerar na Fase 3

```
Flash Efficient attention on Current AMD GPU is still experimental.
Mem Efficient attention on Current AMD GPU is still experimental.
Enable it with TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
```

SDPA funciona, mas usa o backend matemático; os caminhos otimizados estão atrás
dessa flag. **Não ativada**: a Fase 3 mede tempo e VRAM, e ligar um backend
experimental alteraria justamente a medição. Fica como variável a testar depois,
com baseline para comparar.

### Dataset IAM — obtido via Kaggle

O orientando indicou `teykaicong/iamondb-handwriting-dataset`. O nome sugere
IAM-**OnDB** (dados de trajetória de caneta, que **não** serviriam), mas a
descrição do dataset confirma tratar-se do offline correto:

```
words.tgz : Contains words (example: a01/a01-122/a01-122-s01-02.png)
xml.tgz: Contains the meta-information in XML format
```

Download anônimo funciona (o 404 inicial era do método HEAD, não da ausência de
credencial). 788 MB; zip íntegro; contém `words.tgz` (820,9 MB) e `xml.tgz`.

`unzip` **não existe** na máquina e não há sudo → extração feita com o módulo
`zipfile`/`tarfile` do Python.

### Pesos baixados (só o necessário para sampling)

De `konnik/DiffusionPen`, via `huggingface_hub`, com symlink do cache para o
diretório do repo:

| arquivo | tamanho | baixado |
|---|---|---|
| `saved_iam_data/train_word_IAM.pt` | 5,30 GB | sim |
| `diffusionpen_iam_model_path/models/ema_ckpt.pt` | 0,68 GB | sim |
| `diffusionpen_iam_model_path/models/ckpt.pt` | 0,68 GB | sim |
| `style_models/iam_style_diffusionpen.pth` | 0,01 GB | sim |
| `saved_iam_data/test_word_IAM.pt` | 1,93 GB | **não** |
| `diffusionpen_iam_model_path/models/optim.pt` | 1,35 GB | **não** |

Poupados 3,3 GB: `optim.pt` só serve para retomar treino e `test_word_IAM.pt`
não é usado no sampling.

Do SD 1.5: `vae/` (safetensors, 334 MB), `scheduler/`, `model_index.json`.
