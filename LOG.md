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
