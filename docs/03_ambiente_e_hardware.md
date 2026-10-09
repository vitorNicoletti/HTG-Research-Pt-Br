# 03 — Ambiente, hardware e armadilhas

Boa parte do tempo do projeto foi gasta em problemas de máquina. Este
documento existe para que ninguém repita esses erros.

## Máquinas usadas

| máquina | GPU | uso | situação |
|---|---|---|---|
| notebook (Arch Linux) | Intel UHD 620 | análise estática do código, testes de tokenização | sem GPU útil |
| desktop antigo | **AMD RX 6600 XT** (RDNA2, gfx1032) | primeiros treinos no BRESSAY | **defeituosa para este modelo: todo treino feito nela é inválido** |
| desktop atual | **AMD RX 9060 XT** (RDNA4, gfx1200, 17 GB) em **Windows 11 + WSL2 (Ubuntu 26.04)**, 15 GB de RAM | todos os resultados válidos | validada |

## O defeito da RX 6600 XT (e por que validar qualquer GPU nova)

Os primeiros fine-tunes no BRESSAY tinham sintomas estranhos:
- ~11% dos lotes com gradiente não finito, em todas as épocas;
- a MSE melhorava enquanto a geração piorava;
- amostras em ruído colorido, sem nenhum `NaN` nos pesos.

A investigação (`diagnostico/`, detalhes em `diagnostico/README.md`) usou a
CPU como referência:

| teste | RX 6600 XT | correto seria |
|---|---|---|
| uma convolução isolada | 1,1e-06 | ~1e-6 ✓ |
| forward do UNet inteiro, lote 1 a 32 | ~2e-06 depois de reiniciar a máquina | ✓ |
| **direção do gradiente (cosseno GPU × CPU, lote 8)** | **0,357** | ~1,0 |
| fração do gradiente ortogonal ao correto | 93,6% | ~0% |

**O backward estava errado na direção, não na escala.** O corte de norma
(`clip_grad_norm`) não conserta isso: 94% de cada passo era ruído. Por isso
treinar piorava o modelo. Acumular lotes de 1 não resolveu, e foi até pior.
Gerar imagens, que só usa o forward, funcionava.

**Também descobrimos que o estado da máquina mudava os resultados:** depois de
3,5 dias ligada, o forward também errava (9,7e-02); depois de reiniciar, voltou
ao normal. Por isso anotamos o tempo ligado (uptime) junto de cada medida.

**Regra do projeto:** antes de treinar em qualquer GPU nova, rodar:

```bash
python diagnostico/teste_conv_isolada.py           # segundos, sem checkpoint
python diagnostico/teste_gradiente_sintetico.py    # gradiente não finito com dados sintéticos
python diagnostico/teste_direcao_gradiente.py      # cosseno do gradiente GPU × CPU: tem de dar ~1,0
```

A RX 9060 XT passou: forward 1,08e-06, 0 de 60 gradientes não finitos e
**cosseno 1,002**.

## A máquina atual: Windows + WSL2

O código roda dentro do WSL2 (Linux), na pasta `~/HTG-Research-Pt-Br`. O
repositório do Windows (`C:\Users\...\HTG-Research-Pt-Br`) é onde se edita e
faz commit; o WSL faz `git pull` do GitHub. **Dados, bases, modelos e
avaliações só existem no WSL** (são grandes e ficam fora do git).

Ambiente Python:
- **venv:** `~/htg-tcc/venv-diffpen`, Python 3.12.
- **torch:** `2.11.0+rocm7.13.0`, do índice `repo.amd.com/rocm/whl/gfx120X-all/`.
- **variáveis:** antes de rodar, exportar
  - `LD_LIBRARY_PATH` para `~/htg-tcc/syslibs/root/usr/lib/x86_64-linux-gnu`
    (a libgomp);
  - `CPLUS_INCLUDE_PATH` para os headers `include/c++/16` da mesma raiz. O
    MIOpen compila kernels em tempo de execução e precisa deles.
- O script `~/env_htg.sh` faz isso e ativa o venv: `source ~/env_htg.sh`.
- **Nunca** definir `HSA_OVERRIDE_GFX_VERSION` nesta placa. Ele só servia para
  a RX 6600 XT (gfx1032 → 10.3.0) e atrapalha qualquer outra.

Desempenho de referência: ~2,2 passos/s com lote 32, ou seja, ~8,5 min por
época de 33 mil amostras e ~18 min por época de 75 mil.

Há também um `flake.nix` com shells Nix para ROCm, CUDA e CPU
(`nix develop .#rocm`), usado na máquina antiga. Na máquina atual usamos o venv.

## Armadilhas que custaram horas

| armadilha | sintoma | solução |
|---|---|---|
| **O WSL desliga sozinho** quando nenhuma sessão está aberta | o treino morre no meio, sem erro | lançar com `setsid nohup ... &` **e** manter uma sessão aberta (um terminal do Ubuntu, ou um processo `sleep` longo pelo `wsl.exe`) |
| **O `wsl.exe` expande `$VAR`** do lado do Windows | comandos com variáveis quebram | escrever o comando num `.sh` e rodar `wsl.exe -d Ubuntu -- bash arquivo.sh` |
| **O `pkill -f` acerta o próprio comando** | o comando de limpeza se mata (código 15) | filtrar por PID com `ps` + `awk`, ou usar padrão com colchete: `grep "[t]rain.py"` |
| **O MIOpen compila um kernel para cada formato novo** | cada lote de largura diferente leva segundos | usar larguras fixas (múltiplos de 128) no alinhador e no leitor |
| **LSTM no gfx1200** | o leitor com cabeça LSTM ficava parado no platô do "branco" do CTC | usamos a cabeça Transformer (CER 0,084) |
| **Muitas threads por processo** | carga 300 na CPU com 16 processos | `OMP_NUM_THREADS=1` e `cv2.setNumThreads(1)` em cada processo |
| **Disco do Windows cheio** | o WSL grava arquivos **vazios**, sem erro: corrompeu objetos do git e 24 arquivos de código | ver abaixo |
| **Flags booleanas do `train.py`** | `--load_check False` vale **True** (qualquer texto não vazio é verdadeiro) | omitir a flag em vez de passar `False` |
| **Stable Diffusion 1.5 saiu do Hugging Face** | `runwayml/stable-diffusion-v1-5` não existe mais | usar `stable-diffusion-v1-5/stable-diffusion-v1-5` (padrão nos nossos scripts) |

### O episódio do disco cheio (7/10/2026)

O disco virtual do WSL é um arquivo (`ext4.vhdx`) que **cresce dentro do C:**
do Windows. Com o C: em 0,2 GB livres, o WSL não conseguia crescer, e as
gravações saíam vazias **sem nenhum erro**.

Conserto:
1. Apagamos os pesos salvos a cada época (`ema_ep*.pt`, 47 GB) e o cache do
   `uv` (28 GB). Os `ema_bloco_*.pt`, que são os modelos avaliados, ficaram.
2. Rodamos `fstrim -av` como root e `wsl --shutdown`. O C: voltou a 99 GB
   livres.
3. Consertamos o git do WSL: os objetos vazios foram movidos para
   `~/git_objetos_vazios/` e baixados de novo com `git fetch`. Os arquivos
   truncados foram restaurados com `git checkout -- .`.

**Prevenção:**
- Cada checkpoint tem ~650 MB e cada treino guarda vários.
- Mantenha folga no C:.
- Apague `ckpt.pt` e `optim.pt` de treinos que não serão retomados.
