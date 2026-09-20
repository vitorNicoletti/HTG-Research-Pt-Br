#!/usr/bin/env bash
# Fine-tune do DiffusionPen (a partir do pre-treinado no IAM) no BRESSAY,
# em blocos, medindo a deriva e gerando amostras ao fim de cada bloco.
#
# O ALVO e a deriva, nao o numero de epocas. Medido nos treinos ja feitos:
#
#     0,10% - 0,16%   preso entre IAM e BRESSAY: ilegivel
#     2,1%  - 2,7%    BRESSAY legivel, com diacriticos  <-- o que queremos
#     12,9%           degradado de novo
#
# Numa placa sadia o modelo caminha mais rapido do que caminhou na RX 6600 XT
# (onde 64% de cada gradiente era ruido ortogonal), entao NAO fixe o numero de
# epocas: olhe a deriva impressa ao fim de cada bloco e as amostras.
#
# O MSE nao serve como sinal de saude -- ja melhorou (0,0522 -> 0,0404) na
# mesma epoca em que a geracao morreu. O unico teste que vale e gerar amostra.
set -u

export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# A RX 6600 XT e gfx1032 e o ROCm so tem kernels ate gfx1030; sem o override os
# resultados sao silenciosamente errados. Em QUALQUER outra placa o override e
# nocivo -- forcaria kernels de RDNA2 numa arquitetura diferente. Por isso e
# condicional: so entra se a placa detectada for mesmo gfx1032.
if [ -z "${HSA_OVERRIDE_GFX_VERSION:-}" ] && command -v rocminfo >/dev/null 2>&1; then
  if rocminfo 2>/dev/null | grep -q gfx1032; then
    export HSA_OVERRIDE_GFX_VERSION=10.3.0
    echo "placa gfx1032 detectada: HSA_OVERRIDE_GFX_VERSION=10.3.0"
  fi
fi

SAVE_PATH=${SAVE_PATH:-./model_bressay_longo}
BLOCO=${BLOCO:-5}
ALVO=${ALVO:-40}
STYLE=${STYLE:-./DiffusionPen/style_models/iam_style_diffusionpen.pth}
IAM_BASE=./DiffusionPen/diffusionpen_iam_model_path/models
LOG=${LOG:-run_bressay_longo.log}

comum=(
  --dataset bressay
  --model_name diffusionpen
  --save_path "$SAVE_PATH"
  --style_path "$STYLE"
  --max_samples 0            # split inteiro: o unico run bom usou 74.882
  --sample_every 0           # a grade interna tokeniza com max_length=200
                             # enquanto o treino usa 40 -- nao e o mesmo
                             # condicionamento. As amostras saem do
                             # gerar_amostras.py, em processo separado.
  --lr 2e-5                  # valor do unico run que produziu diacriticos
  --batch_size 32
  --num_workers 12
  --save_every_steps 500
  --stable_dif_path runwayml/stable-diffusion-v1-5
)

epocas_feitas() {
  if [ -f "$SAVE_PATH/models/estado.pt" ]; then
    python - "$SAVE_PATH" <<'PY'
import sys, torch
print(torch.load(f'{sys.argv[1]}/models/estado.pt', map_location='cpu', weights_only=True)['epoch'] + 1)
PY
  else
    echo 0
  fi
}

while true; do
  feitas=$(epocas_feitas)
  if [ "$feitas" -ge "$ALVO" ]; then
    echo "=== $feitas epocas concluidas, teto de $ALVO atingido ==="
    break
  fi

  restam=$((ALVO - feitas))
  n=$(( restam < BLOCO ? restam : BLOCO ))
  echo "=== bloco: $feitas epocas feitas, treinando mais $n ==="

  if [ "$feitas" -eq 0 ]; then
    # --pretrained_path e --load_check sao mutuamente exclusivos: no train.py o
    # bloco do pretrained_path roda DEPOIS e sobrescreve o que o load_check
    # retomou. Por isso cada ramo passa so um dos dois.
    python DiffusionPen/train.py "${comum[@]}" \
      --pretrained_path "$IAM_BASE" --epochs "$n" 2>&1 | tee -a "$LOG"
  else
    python DiffusionPen/train.py "${comum[@]}" \
      --load_check True --epochs "$n" 2>&1 | tee -a "$LOG"
  fi

  codigo=${PIPESTATUS[0]}
  if [ "$codigo" -eq 3 ]; then
    echo "=== processo entrou no estado ruim; relancando do ultimo checkpoint ==="
    sleep 10
    continue
  elif [ "$codigo" -ne 0 ]; then
    echo "=== saiu com codigo $codigo; parando para voce olhar ==="
    exit "$codigo"
  fi

  agora=$(epocas_feitas)
  cp "$SAVE_PATH/models/ema_ckpt.pt" "$SAVE_PATH/models/ema_bloco_${agora}ep.pt"
  echo "=== snapshot: ema_bloco_${agora}ep.pt ==="

  python scripts/medir_deriva.py "$SAVE_PATH/models/ema_ckpt.pt" 2>&1 | tee -a "$LOG"

  echo "=== gerando amostras de ${agora} epocas ==="
  python scripts/gerar_amostras.py \
    --ckpt "$SAVE_PATH/models/ema_ckpt.pt" \
    --out "./amostras_${agora}ep" \
    --style "$STYLE" 2>&1 | tail -3 | tee -a "$LOG"
  echo "=== olhe ./amostras_${agora}ep: o criterio e o diacritico aparecer ==="
done
