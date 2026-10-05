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

---

## 2026-09-19 — Fine-tune executado; bloqueio de hardware identificado

### Estado: **BLOQUEADA** — a GPU de desenvolvimento não computa o modelo de forma confiável

Sessão longa de execução e diagnóstico na RX 6600 XT (RDNA2, ROCm,
`HSA_OVERRIDE_GFX_VERSION=10.3.0`, PyTorch 2.12 / HIP 7.2).

### O que foi executado

- Extrator de estilo treinado no BRESSAY (50 épocas; melhor checkpoint na 20 —
  depois disso o triplet de validação piora, ou seja, overfita nos escritores
  vistos).
- Fine-tune do pré-treinado do IAM, várias tentativas: 40 épocas em 20.000
  amostras; 12 épocas no split inteiro (74.882); e um segundo run em blocos de
  5 épocas.

### Sintomas observados

- ~11% dos batches com gradiente não-finito, em **todas** as épocas de **todos**
  os runs.
- Qualidade da geração **piorando** conforme treinava, enquanto o MSE melhorava
  (0,0522 → 0,0404 numa época, com o modelo perdendo de vez a capacidade de
  gerar).
- Grades de amostra saindo em ruído colorido ou preto, sem que os pesos
  tivessem um único `NaN` — verificado: 0 em 170.908.868 parâmetros, incluindo
  o estado do AdamW.
- O primeiro run colapsou de vez por volta da época 28.

### Causa raiz

A placa produz resultados numericamente instáveis para este modelo. Medido com
a CPU como referência (`diagnostico/`):

| teste | resultado |
|---|---|
| uma `Conv2d` isolada, CPU vs GPU | 1,1e-06 — correto |
| UNet na CPU, lote 1 vs lote 4 | 5,2e-07 — CPU independe do lote |
| UNet na GPU, lote 1, vs CPU | 9,7e-07 — correto |
| UNet na GPU, lote 4, vs CPU | 9,9e-02 — **errado** |
| UNet na GPU, entradas idênticas, lotes 1 a 32 | **1,75e-03 a 1,08** — **instável** |

Duas chamadas idênticas divergem. Uma convolução isolada passa no teste, então
o defeito só aparece no modelo completo.

### Hipóteses levantadas e descartadas por medição

Registradas porque cada uma custou horas e nenhuma se sustentou:

1. `label_emb` indexado fora dos limites (647 escritores contra 339 classes) —
   **falso**: o `y` é sobrescrito pelas features de estilo em `unet.py:1287`, e
   o `label_emb` nunca é chamado nesta configuração.
2. Extrator de estilo do BRESSAY com features de magnitude anormal — **falso**:
   norma L2 mediana 46,7 contra 46,0 do IAM, praticamente idênticas.
3. Imagens degeneradas no dataset — **falso**: varredura das 74.882 imagens
   achou 6 com aspecto extremo, nenhuma ilegível ou sem contraste.
4. `set_timesteps` mutando o scheduler compartilhado — **falso**: `add_noise`
   bit a bit idêntico antes e depois.
5. Amostragem dentro do treino envenenando o processo — **falso**: sem amostrar
   nenhuma vez, os descartes continuaram em ~11% e o processo travou igual.
6. Convolução Winograd do MIOpen — **falso**: desligada, o piso de 11%
   permaneceu.

### Correções de código feitas no caminho (aproveitáveis)

Em `diffusionpen_mods/`, com README detalhando cada uma. As principais: guard
de gradiente não-finito antes do `optimizer.step()` (sem ele, um único
gradiente `inf` envenena o `exp_avg_sq` do AdamW de forma permanente); retomada
real com `estado.pt`; correção do crash determinístico no treino do extrator de
estilo; e o carregamento lazy do dataset, que antes estourava 32 GB de RAM.

### Reorganização

Repositório de 45 GB para 7,6 GB. Modelos, extrator de estilo, logs, amostras e
caches `.pt` não lidos foram **movidos** (não apagados) para
`~/repos/htg-tcc-arquivo/`.

### Próximo passo

Validar a RTX 3060 da equipe com `diagnostico/` e repetir o treino nela. A RX
9060 XT é RDNA4 — mesma classe de risco, precisa passar pelo mesmo teste antes
de qualquer treino.

---

## 2026-09-28 — RX 9060 XT validada; fine-tune no split de 25%

### Diagnóstico de GPU (`diagnostico/`) — **passou**

Máquina `DESKTOP-KKFR30E` (WSL2, Ubuntu 26.04), AMD RX 9060 XT (gfx1200,
17 GB), torch `2.11.0+rocm7.13.0` do índice `gfx120X-all`. Uptime de 49 min
no momento dos testes.

| teste | resultado | referência |
|---|---|---|
| `teste_conv_isolada.py` | CPU vs GPU: forward 1,08e-06, gradiente 1,16e-06; GPU repetida 0,0 | ~1e-6 = ok |
| `teste_gradiente_sintetico.py` (lote 32, 60 passos) | 0/60 não-finitos, \|grad\| mediana 7,1, máx 9,2 | 0 = ok; RX 6600 XT: 1/60, mediana 57 |
| `teste_direcao_gradiente.py` (lote 8) | cosseno 1,002, normas 9,505 / 9,505, 0,2% ortogonal | ~1,0 = ok; RX 6600 XT: 0,357 |

Rodado uma vez cada (o README pede 2 ou 3 em processos separados; pendente).
Ao contrário da RX 6600 XT, o gradiente desta placa bate com a CPU.

### Split reduzido

`scripts/reduzir_split.py --fracao 0.25` → `bressay_split_25/`: 18.724 palavras
de treino (de 74.882), os 647 escritores, 10,2% com diacrítico, piso de 5
palavras por escritor, seed 42. Validação e teste copiados sem mudança.

### Run `model_bressay_25`

```
SPLIT=./bressay_split_25 SAVE_PATH=./model_bressay_25 BLOCO=5 ALVO=40 NUM_WORKERS=8 bash scripts/treinar.sh
```

- IAM carregado por `--pretrained_path`: 457/457 chaves em `ckpt.pt` e `ema_ckpt.pt`.
- 586 passos/época, **2,08 passos/s** depois da compilação JIT do MIOpen
  (~4,7 min/época). Pico de RAM ~4,6 GB com 8 workers.
- MSE 0,15 → 0,07 na primeira época, sem nenhum batch descartado.
- Imagens do BRESSAY extraídas do `bressay.zip` (só `data/words` e `sets`):
  416.826 PNGs, 0 ausentes nos três splits.

---

## 2026-09-29 — Comparação IAM vs. fine-tune e resolução do BRESSAY

- O treino `model_bressay_25` parou às 22:17 de 2026-09-28, no passo 342/586
  da época 26: o WSL foi desligado quando o processo que o mantinha vivo saiu
  com a sessão. Última época completa: 25 (26 épocas), `estado.pt` íntegro.
- Deriva por bloco: 0,886% (5) · 1,314% (10) · 1,641% (15) · 1,913% (20) ·
  2,153% (25) · 2,198% (26 épocas). MSE 0,0667 → 0,0502. 0 batches descartados.
- Amostras do modelo original do IAM e do checkpoint de 26 épocas, com
  referência de estilo do BRESSAY e do IAM (`--seed 42 --styles 4`):
  `saidas/diffusionpen/fine_tune_25/comparacao_iam_vs_26ep.png`. O fine-tune
  ficou pior que o modelo de partida em todas as palavras, inclusive no
  controle ASCII.
- `diagnostico/resolucao_bressay.py` sobre o `bressay.zip`: palavras com altura
  mediana de 26 px no dataset e 31 px por página no split filtrado. Corte de
  35 px deixa 12 páginas de treino. Análise em `ACHADOS.md`, seção 7.

---

## 2026-09-30 — Run `model_bressay_25_v2` (pré-processamento v2)

- `python scripts/treinar.py experimentos/bressay_25_v2.json`: split
  `bressay_split_25_v2` (17.358 palavras, tinta ≥ 14 px), pré-processamento v2,
  demais parâmetros iguais aos do `bressay_25`. 543 passos/época, ~2,2 passos/s.
- O PC foi desligado no passo 153 da época 38; checkpoints íntegros, retomado
  com o mesmo comando a partir da época 38. 40 épocas, 0 batches descartados.
- Deriva: 0,937% (5) · 1,395% (10) · 1,744% (15) · 2,031% (20) · 2,285% (25) ·
  2,514% (30) · 2,726% (35) · 2,924% (40).
  O `medir_deriva.py` marca "FAIXA ALVO" a partir de 30 épocas.
- Comparação com as mesmas palavras, seed 42, 4 estilos, contra o IAM original
  e o `v1` de 26 épocas, com referência de estilo do IAM e do BRESSAY:
  `saidas/diffusionpen/fine_tune_25_v2/`. Análise no `ACHADOS.md`, seção 8.

---

## 2026-10-02 — Acentos sintéticos no IAM: alinhamento CTC das letras

- Pacote `acentos_sinteticos/` (gerador de acentos e cedilhas sobre palavras
  do IAM) e folhas em `saidas/acentos_sinteticos/`
  (`scripts/amostras_acentos.py --n 30 --seed 0`). Com fatias iguais por
  letra, ~24/30 amostras boas; as falhas vinham de letras de larguras muito
  diferentes.
- Reconhecedor CTC só convolucional para localizar as letras:
  `python scripts/treinar_alinhador.py --epocas 15 --saida modelos/alinhador_iam.pt`
  (`iam_training.txt` 47.981 palavras, validação `iam_val.txt` 7.554, alfabeto
  78). RX 9060 XT: ~103 s/época; a primeira execução levou 811 s por causa da
  compilação de kernels do MIOpen. Melhor CER de validação **0,127** na época
  14 (62% de palavras lidas certas). Os pesos ficam só em
  `~/HTG-Research-Pt-Br/modelos/alinhador_iam.pt` (gitignored) e são
  reconstruídos pelo comando acima.
- `scripts/avaliar_posicao_letras.py`: verdade automática em 2.017 palavras de
  val+test em que cada componente conexo é uma letra (2.969 letras-alvo).
  Acerto da letra certa (todas / letras do meio) e viés mediano em larguras de
  letra:
  - fatias iguais: 92,6% / 89,1%, viés −0,06;
  - iguais + vale: 94,9% / 92,5%;
  - CTC (meio entre disparos): 97,8% / 97,1%, viés +0,03;
  - **CTC + vale: 98,0% / 97,3%, viés −0,04**. É o que o gerador usa.
  - Disparo CTC direto: 98,6%, mas com viés de +0,23 para a direita.

  A confiança do alinhamento (`logp_medio`) separa os erros: no quartil
  inferior o acerto é 94,1%; nos outros quartis, 98,5% ou mais. Letras soltas
  são o caso fácil; na escrita cursiva o erro deve ser maior.
- Nas 30 amostras, CTC + vale e fatias iguais quase sempre caem no mesmo
  lugar. Ampliado, `bóttom` estava certo. `fõr` (logp −0,29, lido "fo") é um
  recorte ambíguo, e `stumblêd` (lido "stumblerd") continua com o circunflexo
  sobre o "d". Os dois de logp mais baixo são `fõr` e `bóttom`.

### Avaliação visual de 200 amostras (CTC + vales, seed 1)

- `python scripts/amostras_acentos.py --n 200 --seed 1 --alinhador modelos/alinhador_iam.pt --saida saidas/acentos_sinteticos/avaliacao_200`.
  Palavras de `iam_train_val`, em sua maioria cursivas.
- Julgamento feito pelo Claude, um avaliador só, às cegas para o `logp`, em
  folhas de 25 (`avaliacao_200/revisao/revisao_*.png`). Categorias:
  C = sinal na letra certa, E = letra errada, F = sinal fraco ou invisível,
  ? = não julgável. Cada amostra com o seu motivo:
  `avaliacao_200/julgamento.tsv`.
- Resultado: **C 182 · E 7 · F 10 · ? 1**. Sobre as 199 julgáveis:
  91,5% certas, 3,5% na letra errada, 5,0% fracas. O `?` é um rótulo errado do
  próprio IAM: `person`, mas a imagem mostra `people`.
- Corte por `logp_alinhamento`:
  | corte | fica | C | E | descartados (E/F/C) |
  |---|---|---|---|---|
  | nenhum | 199 | 91,5% | 3,5% | — |
  | −0,2 | 193 (97%) | 93,3% | 2,1% | 3 / 1 / 2 |
  | −0,1 | 184 (92%) | 92,9% | 2,2% | 3 / 1 / 11 |
  | −0,05 | 158 (79%) | 94,3% | 1,3% | 5 / 3 / 33 |
  | −0,03 | 118 (59%) | 96,6% | 0% | 7 / 6 / 68 |

  O corte em −0,2 pega 3 dos 7 erros perdendo só 2 amostras boas. Os outros 4
  erros têm `logp` entre −0,06 e −0,03, no meio das amostras boas. Os sinais
  fracos não se separam pelo `logp` (mediana −0,034).
- Padrões nas falhas:
  - **í:** 7 das 18 amostras com problema (2 E e 5 F). O agudo do i sai fino
    ou some.
  - **Letra logo depois de um `h`:** 3 dos 7 erros (`háppens`, `hélp`,
    `húll`); o sinal cai na haste do h.
  - **"Sem sinal visível"** em 5 casos (`fór`, `wíth`, `sidê`, `strêwn`):
    provavelmente o sinal foi desenhado sobre tinta já existente, e a mistura
    por mínimo o escondeu.

### Filtro de visibilidade e base completa

- `desenho.visibilidade` mede quanto do sinal virou tinta nova (pixels novos ÷
  comprimento × espessura). Calibrado nas mesmas 200 amostras, que a semente 1
  reproduz idênticas:
  - os 4 sinais escondidos em tinta existente têm 0,33–0,54;
  - das 182 boas, só `perfõrm` fica abaixo de 0,6 (0,42); as outras estão acima
    de 0,68.

  Limiar **0,6**, com até 3 sorteios por palavra.
- **Correção da avaliação de 200:** ampliados, 6 dos 10 sinais "fracos" (5 com
  `í` e o `bút`) estavam visíveis, com 1,8–4,6 px de espessura na escala do
  modelo. Pareciam fracos só porque as palavras de caneta fina apareciam muito
  reduzidas nas folhas. Os fracos de verdade são os 4 escondidos.
- O pingo do `i` agora é apagado por inpainting (`cv2.inpaint`). A cor única
  deixava um quadrado claro em papel com textura.
- `python scripts/gerar_base_acentos.py --alinhador modelos/alinhador_iam.pt --saida iam_acentuado --workers 16`
  (seed 0, `iam_train_val`): **29.732 amostras** de 30.658 elegíveis (97,0%),
  339 escritores, 233 MB, 125 s.
  - Descartes: confiança 771 (2,5%) e invisível 155 (0,5%). 357 palavras
    precisaram de mais de um sorteio.
  - Letras: é 6.228 · ã 4.881 · ê 4.121 · í 3.261 · õ 2.666 · ó 1.642 · á 1.598 ·
    ç 1.197 · ô 1.158 · â 1.107 · ú 1.076 · à 797.
  - Sem `OMP_NUM_THREADS=1` e `cv2.setNumThreads(1)`, os 16 workers disputavam
    threads (load 300, ~4 palavras/s); com eles, ~250 palavras/s.
  - A base fica só no WSL (`~/HTG-Research-Pt-Br/iam_acentuado/`, gitignored).
    Resumo em `saidas/acentos_sinteticos/base/resumo.json`.
- Conferência de 25 amostras aleatórias (`base/conferencia_25.png`):
  - cerca de 21–22 boas;
  - `hér` com o agudo no laço do `h` (de novo, letra depois de `h`);
  - `thê` fundido ao arco do `h` e `háve` espremido contra o `h`, ambos na
    letra certa;
  - `satisfíes` certo: o agudo trocou um pingo em forma de traço.

### Fine-tune na base de acentos sintéticos — lançado

- Leitor novo `diffusionpen_mods/utils/iam_acentuado_dataset.py`
  (`--dataset iam_acentuado`):
  - amostras: as acentuadas mais uma fração `--iam_originais` das palavras
    originais do `iam_train_val`;
  - referências de estilo: sempre palavras originais do mesmo escritor;
  - pré-processamento: o do `IAMDataset`, copiado sem mudança.
- `scripts/treinar.py` ganhou `dados.dataset` e `dados.iam_originais`. Os
  `experimento.json` antigos valem como `bressay`/`0`; o `--dry-run` do
  `bressay_25_v2` não acusou diferenças.
- Teste do leitor:
  - 85.267 amostras (29.732 acentuadas e 55.535 originais), 339 escritores;
  - 22 ms por amostra, com as 5 referências;
  - imagens conferidas como o modelo as recebe.
- `python scripts/treinar.py experimentos/iam_acentuado.json`: lr 2e-5, batch
  32, 10 épocas em blocos de 2, a partir dos pesos do IAM.
  - Os pesos carregaram com as 457/457 chaves.
  - 2.665 passos por época, ~2 passos/s, ~22 min por época.
  - MSE inicial ≈ 0,047.

## 2026-10-04 — Fine-tune `model_iam_acentuado` (10 épocas)

- Interrompido em 2026-10-02, depois de ~6 min. Relançado em 2026-10-03 com o mesmo
  comando; como nenhuma época tinha terminado, recomeçou dos pesos do IAM.
- 10 épocas, sem erro. ~20 min por época; ~1,35 passo/s no fim, ~2,2 no começo.
  MSE 0,043 → 0,030.
- Deriva: 1,009% (2) · 1,380% (4) · 1,640% (6) · 1,844% (8) · 2,016% (10).
- Amostras de 2 a 10 épocas e do IAM original (mesma semente e estilos) em
  `saidas/diffusionpen/fine_tune_iam_acentuado/`. Análise no ACHADOS, seção 10:
  - o til e a cedilha aparecem em `nação`, `coração` e `pão`;
  - o acento vaza para palavras sem acento (`the` → `thé`);
  - `ó` e `ê` não aparecem.

## 2026-10-04 — Base com teto por palavra e run `model_iam_acentuado_teto25`

- `gerar_base_acentos.py --teto_por_palavra`: cada palavra recebe no máximo
  `max(1, round(teto × ocorrências))` versões acentuadas, sorteadas com a seed.
  O padrão 1,0 reproduz a base anterior.
- `--teto_por_palavra 0.25 --saida iam_acentuado_teto25`:
  - 10.232 ocorrências selecionadas de 5.223 palavras distintas;
  - **9.892 geradas** (descartes: confiança 301, invisível 39), 45 s;
  - `the` acentuado: 718 contra 2.907 sem acento (eram 2.858). `thé`/`thê`
    continuam sendo as mais frequentes (436 e 282).
- `python scripts/treinar.py experimentos/iam_acentuado_teto25.json`:
  - 65.427 amostras, 15% acentuadas (eram 35%);
  - mesmos hiperparâmetros do `iam_acentuado`, 2.045 passos por época,
    ~16 min por época;
  - nas amostras de cada bloco entram os controles `and` e `with`, que na
    base viram `ãnd` e `wíth`.
- Interrompido a pedido no passo 1.556/2.045 da 2ª época. `estado.pt` registra 1
  época completa (`epoch` 0, `ema_step` 3.545). Retomar com o mesmo comando
  continua da 2ª época.
- Retomado em 2026-10-04 a partir de 1 época e concluído com 10 épocas. Os blocos
  passaram a fechar nas épocas 3, 5, 7, 9 e 10. O ritmo caiu de ~2,1 para ~1,4
  passo/s ao longo da sessão.
- Deriva: 1,099% (3) · 1,320% (5) · 1,491% (7) · 1,635% (9) · 1,698% (10).
- `and` e `with` gerados também para o IAM original e para o
  `model_iam_acentuado` 10 ép. (mesma semente). Comparações em
  `saidas/diffusionpen/fine_tune_iam_acentuado/`: `comparacao_teto.png`,
  `comparacao_teto_4estilos.png` e `evolucao_teto25.png`. Análise no
  ACHADOS, seção 10 (adendo).

## 2026-10-04 — Medição de marcas soltas (80 amostras por palavra)

- `gerar_amostras.py --paineis` salva cada painel separado.
  `scripts/medir_marcas.py` conta as marcas soltas acima e abaixo do corpo.
- 17 palavras × 40 escritores do IAM (`--classes_iam`: o 12 e mais 39 sorteados
  com `Random(0)`) × sementes 42 e 43 × 3 modelos (IAM, `model_iam_acentuado`
  10 ép., `model_iam_acentuado_teto25` 10 ép.), `--em_lote`. ~2,5 min por
  modelo e semente; a primeira chamada levou 5 min por causa da compilação
  do MIOpen.
- Tabela e análise no ACHADOS, seção 10 (adendo 2). Painéis no WSL em
  `~/HTG-Research-Pt-Br/medicao_marcas/`.

## 2026-10-04 — Base portuguesa (palavras do português geradas pelo IAM e acentuadas)

Ordem seguida para não vazar dados: a partição foi congelada antes de gerar
qualquer imagem.

1. **Vocabulário.** `scripts/preparar_vocabulario_pt.py`:
   - fonte: FrequencyWords `pt_br_50k` (OpenSubtitles 2018, CC-BY-SA 4.0), com o
     sha256 em `vocabulario_pt/resumo.json`;
   - 48.658 palavras de 2 a 12 letras, em 41.054 grupos;
   - **grupo** = esqueleto sem acento com o plural dobrado; o grupo inteiro vai
     para um split só;
   - forçados no teste: as 97 palavras de avaliação do repositório (sonda,
     pares da métrica, amostras dos experimentos, medição de marcas) e as 1.083
     acentuadas do teste do BRESSAY. Forçadas na validação: as 875 acentuadas
     da validação do BRESSAY. O resto vai por hash do grupo (10% teste, 5% val);
   - treino: 4.952 acentuadas e 35.248 sem acento; val: 532 e 2.105; teste:
     1.536 e 4.285;
   - commit `c0cfc38`. Não regenerar: o script também lê os
     `experimentos/*.json`.
2. **Várias marcas por palavra.** `gerador.acentuar_palavra` desenha todos os
   sinais; se um sair invisível, a amostra é descartada. O desenho de um sinal
   foi extraído para `_desenhar_sinal`, e as 40 primeiras amostras da base
   inglesa saem idênticas pixel a pixel (regressão conferida).
3. **Geração.**
   - `comum/diffusionpen.py`: DiffusionPen em lote, com referências pelo
     pré-processamento do `IAMDataset`.
   - `scripts/gerar_base_pt.py`: só palavras do treino (conferido antes de
     gerar e a cada gravação), só escritores do `iam_train_val`, palavras
     amostradas uniformemente com 2 imagens por acentuada.
   - Cada imagem é recortada na tinta e ampliada 2×. Sai se o alinhador não
     ler o esqueleto com `logp ≥ −0,2`.
   - Tipos de amostra: `acentuada`, `par` (a mesma imagem sem os sinais,
     rótulo = esqueleto) e `sem_acento`.
   - Teste com 40 + 20 palavras: 51% aproveitadas (as gerações ilegíveis do
     IAM em palavras longas e raras saem); folha em
     `saidas/acentos_sinteticos/base_pt_teste.png`.
4. **Avaliação fixada antes do modelo novo.** `scripts/avaliar_pt.py`:
   - palavras de um split (val para escolher checkpoint, teste uma vez no fim):
     30 acentuadas, os 30 esqueletos e 20 sem acento, sem `i`/`j`; no teste
     entram também os pares da sonda;
   - 20 escritores do `iam_test` × sementes 42 e 43;
   - mede a taxa de marca solta, a diferença pareada (acentuada − esqueleto) e
     o CER de um leitor **separado** (cabeça LSTM, semente 1), não do alinhador
     que filtrou a base;
   - para com erro se uma palavra avaliada não for do split, se o grupo dela
     estiver numa base de treino, ou se um escritor estiver no treino.
5. O leitor do treino (`iam_acentuado_dataset.py`) confere os rótulos de base
   com vocabulário contra o split de treino. `experimentos/iam_pt.json`:
   `iam_originais` 0,3, 16 épocas em blocos de 4, amostras de bloco com
   palavras de validação.
- Lançado: `gerar_base_pt.py --saida iam_pt --sem_acento 12000` (21.904
  gerações, ~2,7 por segundo). Em seguida, automaticamente,
  `treinar_alinhador.py --cabeca lstm --seed 1 --saida modelos/leitor_iam_lstm.pt`.

## 2026-10-05 — Base `iam_pt`, leitor e run `model_iam_pt`

- `gerar_base_pt.py --saida iam_pt --sem_acento 12000`: 21.904 gerações em 7.681 s.
  - Aproveitadas: 11.559 (descartes: confiança 10.052, invisível 29).
  - Amostras: 5.188 acentuadas, 5.188 pares e 6.516 sem acento. A base fica no WSL.
  - Folha de 24 amostras: `saidas/acentos_sinteticos/base_pt_amostras.png`.
- Leitor da avaliação:
  - a cabeça **LSTM** ficou presa no patamar do branco (perda ~3,5, CER ~0,95 por 7
    épocas), enquanto a convolucional saía dele na 2ª época. Provável defeito do
    LSTM do MIOpen na gfx1200; interrompida;
  - `--cabeca transformer --seed 1`: CER val **0,084** (76% das palavras certas) em
    15 épocas, `modelos/leitor_iam_transformer.pt`.
- `python scripts/treinar.py experimentos/iam_pt.json`:
  - 33.552 amostras (16.892 da base e 16.660 originais);
  - os rótulos foram conferidos contra o vocabulário, todos do treino;
  - 1.049 passos por época, ~8 min por época, 16 épocas;
  - deriva: 0,963% (4) · 1,370% (8) · 1,683% (12) · 1,949% (16).
- `avaliar_pt.py --split val` em IAM e pt 4/8/12/16 épocas (~25 min por modelo).
  Tabela e análise no ACHADOS, seção 11. Checkpoint escolhido: 16 épocas. O teste
  não foi aberto.

## 2026-10-05 — Diagnóstico do peso do acento e alinhamento dos pares

- `diagnostico/diag_peso_acento.py` (96 pares, IAM e pt 16 ép.): análise no
  ACHADOS, seção 12. Resumo:
  - um acento faltando custa 12–26% da perda da amostra em ruído médio a alto,
    mas só ~15% das amostras têm acento;
  - com o texto errado, a perda na região do acento sobe +10–14% no pt contra
    +1,5% no IAM;
  - o CANINE distingue a palavra com e sem acento;
  - 64% dos pares estavam desalinhados.
- Correção dos pares: `gerador.par_na_tela` põe o par na tela da acentuada,
  com a mesma função de margem e a mesma cor de papel. `gerar_base_pt.py` já
  grava assim.
- `scripts/alinhar_pares.py --origem iam_pt --destino iam_pt_alinhado`:
  - 3.296 pares refeitos e 1.892 que já estavam alinhados; os 5.188 agora têm o
    tamanho da sua acentuada;
  - demais imagens por hardlink; `iam_pt` não foi alterada;
  - em 500 pares, depois do pré-processamento do treino, os pixels diferentes
    entre a acentuada e o par caíram de mediana **12,8%** (p95 30%) para
    **0,40%** (p95 1,0%), o tamanho de um acento (0,46%);
  - figuras em `diagnostico/resultados/peso_acento/`: `pares_desalinhados.png`
    (base antiga) e `pares_alinhados_base_nova.png` (a coluna 4 é o que o
    treino recebe; a 5 reaplica a correção e não vale para a base nova).

## 2026-10-05 — Peso no acento na loss: controle e peso 5

- `--peso_acento λ` no `train.py` (só `iam_acentuado`; chave
  `treino.peso_acento` nos experimentos, 1 nos antigos):
  `loss = média((1 + (λ−1)·máscara) · (ε − ε̂)²)`, sem normalizar pelos pesos.
  Com λ = 1 o caminho antigo roda sem mudança.
- Máscara (`mascara_acento` em `utils/iam_acentuado_dataset.py`): pixels em que
  a acentuada e o par diferem mais de 40 níveis, depois do pré-processamento
  do treino; max-pool para 8×32 e dilatação de 1 célula. Exige base com
  `pares_alinhados`. A acentuada e o par recebem a mesma máscara; sem_acento e
  originais do IAM, zeros.
- `diagnostico/conferir_mascara_acento.py` na `iam_pt_alinhado`:
  - os 5.188 pares têm máscara não vazia; a do par é idêntica à da acentuada;
  - a máscara cobre em média 7,6% do latente (mediana 7,0%, p95 13%); no lote,
    2,3%;
  - com λ = 5, a máscara passa de 7% para 27% da perda de uma acentuada
    mediana e de 2,3% para 11% da perda do lote;
  - figura: `diagnostico/resultados/peso_acento/mascaras.png`, a máscara pega
    o sinal e só ele.
- Teste curto na GPU (640 amostras, 1 época, λ = 5): nenhum lote descartado.
  Com os pesos do IAM, o erro de ruído dentro da máscara é **0,319** e fora
  **0,088** (3,6×).
- Treinos em sequência, mesmos parâmetros do `iam_pt` com a base trocada para
  `iam_pt_alinhado`:
  - `experimentos/iam_pt_alinhado.json` (controle, λ = 1) → `model_iam_pt_alinhado`;
  - `experimentos/iam_pt_peso5.json` (λ = 5) → `model_iam_pt_peso5`.
  Logs em `~/treino_pt_alinhado.out` e `~/treino_pt_peso5.out` no WSL.
