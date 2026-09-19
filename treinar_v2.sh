#!/usr/bin/env bash
# Fine-tune do zero (a partir do pre-treinado no IAM), em blocos de 5 epocas,
# gerando amostras ao fim de cada bloco.
#
# Por que em blocos: nesta maquina o MSE NAO e sinal confiavel de saude. Ja
# aconteceu de o MSE melhorar (0.0522 -> 0.0404) enquanto o modelo perdia
# completamente a capacidade de gerar. O unico teste que vale e gerar amostra.
# Cada bloco deixa um snapshot proprio, entao da para voltar atras.
#
# NAO toca em ./model_bressay_full_12ep -- o modelo bom (ema_ep11.pt) fica la.
set -u

export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

# A RX 6600 XT e gfx1032 e o ROCm so tem kernels para gfx1030; sem isto os
# resultados sao silenciosamente errados. Costuma vir do perfil do shell, mas
# uma sessao tmux herda o ambiente do servidor tmux -- entao garantimos aqui.
: "${HSA_OVERRIDE_GFX_VERSION:=10.3.0}"
export HSA_OVERRIDE_GFX_VERSION

# Winograd fica LIGADA de proposito: o unico modelo que comprovadamente gera
# bem (ema_ep11) foi treinado com ela, e as 3 epocas treinadas com ela
# desligada foram as que mataram a geracao.

SAVE_PATH=./model_bressay_v2
BLOCO=5
ALVO=50
LOG=run_bressay_v2.log

comum=(
  --dataset bressay
  --model_name diffusionpen
  --save_path "$SAVE_PATH"
  --style_path ./style_models/mixed_bressay_mobilenetv2_100.pth
  --max_samples 0
  --sample_every 999999      # amostragem dentro do treino corrompe o processo
  --lr 2e-5
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
    echo "=== $feitas epocas concluidas, alvo de $ALVO atingido ==="
    break
  fi

  restam=$((ALVO - feitas))
  n=$(( restam < BLOCO ? restam : BLOCO ))
  echo "=== bloco: $feitas epocas feitas, treinando mais $n ==="

  if [ "$feitas" -eq 0 ]; then
    python diffusionpen/DiffusionPen/train.py "${comum[@]}" \
      --pretrained_path ./diffusionpen/DiffusionPen/diffusionpen_iam_model_path/models \
      --epochs "$n" 2>&1 | tee -a "$LOG"
  else
    python diffusionpen/DiffusionPen/train.py "${comum[@]}" \
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

  # Bloco fechou: snapshot com nome proprio (para poder voltar atras) e amostras.
  agora=$(epocas_feitas)
  cp "$SAVE_PATH/models/ema_ckpt.pt" "$SAVE_PATH/models/ema_bloco_${agora}ep.pt"
  echo "=== snapshot salvo: ema_bloco_${agora}ep.pt ==="

  echo "=== gerando amostras de ${agora} epocas ==="
  SAVE_PATH="$SAVE_PATH" OUT_DIR="./amostras_v2_${agora}ep" CKPT=ema_ckpt.pt \
    python gerar_amostras.py 2>&1 | tail -3 | tee -a "$LOG"
  echo "=== veja ./amostras_v2_${agora}ep e compare com ./amostras_controle_ep11 ==="
done
