#!/usr/bin/env bash
# Treino do iam_acentuado para maquina alugada (Colab, Kaggle): a sessao pode
# cair a qualquer momento, entao ao fim de cada epoca o estado do treino e
# copiado para uma pasta duravel, e uma sessao nova retoma dali.
#
#   NOME=sem_pares EPOCAS=16 DESTINO=/content/drive/MyDrive/htg-tcc/treinos \
#     ARGS="--acento_separado --professor_sem_pares" bash scripts/treino_nuvem.sh
#
# NOME      o treino fica em ./model_<NOME> e a copia em <DESTINO>/model_<NOME>
# EPOCAS    total de epocas; chamadas seguintes treinam so o que falta
# DESTINO   pasta duravel (Drive montado, /kaggle/working ...)
# ARGS      o que muda em relacao ao treino com professor (peso 5 no acento,
#           professor = modelo do IAM); pode sobrepor qualquer opcao de baixo
# BASE      base de acentos (padrao iam_pt_alinhado)
# DEVICE    padrao cuda:0;  WORKERS  padrao = numero de nucleos
#
# Grava so no fim da epoca (--save_every_steps 0): os arquivos de models/ ficam
# parados durante a epoca seguinte e a copia nunca pega um estado pela metade.
set -u
cd "$(dirname "$0")/.."
: "${NOME:?defina NOME}" "${EPOCAS:?defina EPOCAS}" "${DESTINO:?defina DESTINO}"
ARGS=${ARGS:-}
BASE=${BASE:-iam_pt_alinhado}
DEVICE=${DEVICE:-cuda:0}
WORKERS=${WORKERS:-$(nproc)}
PY=${PY:-python -u}
IAM=./DiffusionPen/diffusionpen_iam_model_path/models
SALVAR=./model_$NOME
COPIA=$DESTINO/model_$NOME
LOG=$SALVAR/treino.log

bash scripts/aplicar_mods.sh > /dev/null || { echo "aplicar_mods falhou"; exit 1; }
mkdir -p "$SALVAR/models" "$COPIA/models"

copiar() {  # models/ + registros -> pasta duravel; nome temporario e troca no fim
  local f
  for f in "$SALVAR"/models/*.pt; do
    [ -f "$f" ] || continue
    [ "$f" -nt "$COPIA/models/$(basename "$f")" ] || continue
    cp "$f" "$COPIA/models/.$(basename "$f").tmp" || return 1
  done
  for f in "$COPIA"/models/.*.tmp; do
    [ -f "$f" ] || continue
    local nome; nome=$(basename "$f"); nome=${nome#.}; nome=${nome%.tmp}
    mv "$f" "$COPIA/models/$nome" || return 1
  done
  cp "$LOG" "$SALVAR"/config.jsonl "$COPIA"/ 2>/dev/null
  echo "=== copia em $COPIA ($(date +%H:%M)) ==="
}

# sessao nova: o disco local esta vazio e o estado vem da copia
if [ ! -f "$SALVAR/models/estado.pt" ] && [ -f "$COPIA/models/estado.pt" ]; then
  echo "=== restaurando de $COPIA ==="
  rm -f "$COPIA"/models/.*.tmp
  cp "$COPIA"/models/*.pt "$SALVAR/models/" || exit 1
  cp "$COPIA"/treino.log "$COPIA"/config.jsonl "$SALVAR"/ 2>/dev/null
fi

feitas=0
inicio="--pretrained_path $IAM"
if [ -f "$SALVAR/models/estado.pt" ]; then
  feitas=$($PY -c "import torch;print(torch.load('$SALVAR/models/estado.pt',map_location='cpu',weights_only=True)['epoch']+1)" | tail -1)
  inicio="--load_check True"
fi
if [ "$feitas" -ge "$EPOCAS" ]; then
  echo "=== $NOME: $feitas epocas feitas, nada a treinar ==="
  copiar; exit 0
fi
echo "=== $NOME: $feitas epocas feitas, treinando ate $EPOCAS ($(date +%H:%M)) ==="

# vigia: a cada epoca nova no log, copia. Para sozinho quando aparece .fim, e o
# script espera por ele: duas copias ao mesmo tempo estragariam os arquivos.
rm -f "$SALVAR/.fim"
(
  vistas=$(grep -ac "^epoca [0-9]*: MSE" "$LOG" 2>/dev/null); vistas=${vistas:-0}
  while [ ! -f "$SALVAR/.fim" ]; do
    sleep 20
    agora=$(grep -ac "^epoca [0-9]*: MSE" "$LOG" 2>/dev/null); agora=${agora:-0}
    if [ "$agora" -gt "$vistas" ]; then vistas=$agora; copiar; fi
  done
) &
VIGIA=$!
trap 'kill $VIGIA 2>/dev/null' EXIT

$PY DiffusionPen/train.py --dataset iam_acentuado --model_name diffusionpen --sample_every 0 \
  --save_path "$SALVAR" --dataset_folder "./$BASE" --max_samples 0 --iam_originais 0.3 \
  --style_path ./DiffusionPen/style_models/iam_style_diffusionpen.pth \
  --stable_dif_path "${SD:-stable-diffusion-v1-5/stable-diffusion-v1-5}" \
  --lr 2e-05 --batch_size 32 --adamw_eps 1e-06 --clip_grad_norm 1.0 --ema_beta 0.995 --ema_inicio 2000 \
  --texto_max_len 40 --device "$DEVICE" --num_workers "$WORKERS" --save_every_steps 0 --abort_after 300 \
  --peso_acento 5 --peso_zona 1 --zona vazia --professor "$IAM/ema_ckpt.pt" \
  --epochs $((EPOCAS - feitas)) $inicio $ARGS 2>&1 | tee -a "$LOG" | tr '\r' '\n' | grep -a --line-buffered "^epoca\|^professor\|^acento_separado\|Retomando\|Traceback\|Error\|zona vogais"
codigo=${PIPESTATUS[0]}
touch "$SALVAR/.fim"
wait $VIGIA
copiar
[ "$codigo" -eq 0 ] && echo "=== $NOME pronto: $SALVAR/models/ema_ckpt.pt ($(date +%H:%M)) ===" || echo "=== $NOME parou com codigo $codigo ==="
exit "$codigo"
