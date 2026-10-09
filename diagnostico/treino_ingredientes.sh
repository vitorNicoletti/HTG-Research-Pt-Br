#!/usr/bin/env bash
# Qual ingrediente da base de acentos estraga a letra? Um treino curto por
# ingrediente acrescentado, todos com 16.670 amostras, o UNet inteiro, a partir
# dos pesos do IAM, na CPU:
#
#   1_reais       so palavras reais do IAM                         (ing_1_reais)
#   2_geradas     + palavras pt geradas pelo modelo, sem acento    (ing_2_geradas)
#   3_pares       + os pares (esqueleto de palavra acentuada)      (ing_3_pares)
#   4_acentuadas  + as acentuadas, loss original                   (iam_pt_sub)
#   5 (peso 5)    o mesmo do 4 com peso 5: ja treinado, model_cpu_tudo
#
# Cada braco roda EPOCAS epocas (521 passos cada) e guarda o modelo em
# model_ing_<braco>/models/etapa_<EPOCAS>ep.pt. Braco pronto e pulado; braco
# interrompido retoma de onde parou. No fim, avalia tudo com avaliar_mini.py.
#
#   nohup bash diagnostico/treino_ingredientes.sh > run_ingredientes.log 2>&1 &
#   grep -a "===" run_ingredientes.log          # acompanhar
set -u
cd "$(dirname "$0")/.."
EPOCAS=${EPOCAS:-2}
EXTRA=${EXTRA:-}
ARGS=${ARGS:-}              # argumentos a mais para o train.py, ex.: ARGS="--acento_separado --peso_acento 5"
PREFIXO=${PREFIXO:-model_ing}
SEM_AVALIAR=${SEM_AVALIAR:-}
PY="env -u LD_LIBRARY_PATH nix develop .#cpu --command python -u"

bash scripts/aplicar_mods.sh > /dev/null || { echo "aplicar_mods falhou"; exit 1; }

treinar() {  # braco, base
  local salvar=./${PREFIXO}_$1 base=$2 feitas=0 inicio
  [ -f "$salvar/models/etapa_${EPOCAS}ep.pt" ] && { echo "=== $1: ja pronto ==="; return 0; }
  inicio="--pretrained_path ./DiffusionPen/diffusionpen_iam_model_path/models"
  if [ -f "$salvar/models/estado.pt" ]; then
    feitas=$($PY -c "import torch;print(torch.load('$salvar/models/estado.pt',map_location='cpu',weights_only=True)['epoch']+1)" 2>/dev/null | tail -1)
    inicio="--load_check True"
  fi
  if [ "$feitas" -lt "$EPOCAS" ]; then
    echo "=== $1: $feitas epocas feitas, treinando ate $EPOCAS ($(date +%H:%M)) ==="
    $PY DiffusionPen/train.py --dataset iam_acentuado --model_name diffusionpen --sample_every 0 \
      --save_path "$salvar" --dataset_folder "./$base" --max_samples 0 --iam_originais 0.0 \
      --style_path ./DiffusionPen/style_models/iam_style_diffusionpen.pth \
      --stable_dif_path stable-diffusion-v1-5/stable-diffusion-v1-5 \
      --lr 2e-05 --batch_size 32 --adamw_eps 1e-06 --clip_grad_norm 1.0 --ema_beta 0.995 --ema_inicio 2000 \
      --texto_max_len 40 --device cpu --num_workers 4 --save_every_steps 500 --abort_after 300 \
      --peso_acento 1 --peso_zona 1 --zona vazia --treinar_so tudo \
      --cache_congelados "./${PREFIXO}_cache.pt" --epochs $((EPOCAS - feitas)) $inicio $ARGS $EXTRA \
      || { echo "=== FALHOU $1 ==="; return 1; }
  fi
  cp "$salvar/models/ema_ckpt.pt" "$salvar/models/etapa_${EPOCAS}ep.pt"
  rm -f "$salvar/models/ckpt.pt" "$salvar/models/optim.pt"      # 2 GB por braco; o braco pronto nao e retomado
  echo "=== $1 pronto: $salvar/models/etapa_${EPOCAS}ep.pt ($(date +%H:%M)) ==="
}

# BRACOS="nome:base ..." troca a lista; AVALIAR e SAIDA trocam o que entra na folha final
BRACOS=${BRACOS:-"1_reais:ing_1_reais 2_geradas:ing_2_geradas 3_pares:ing_3_pares 4_acentuadas:iam_pt_sub"}
for nb in $BRACOS; do treinar "${nb%%:*}" "${nb#*:}" || exit 1; done
AVALIAR=${AVALIAR:-"1_reais 2_geradas 3_pares 4_acentuadas"}
SAIDA=${SAIDA:-diagnostico/resultados/ingredientes}

[ -n "$EXTRA$SEM_AVALIAR" ] && { echo "sem avaliacao"; echo FIM; exit 0; }
echo "=== avaliando ($(date +%H:%M)) ==="
M="--modelo iam=DiffusionPen/diffusionpen_iam_model_path/models/ema_ckpt.pt"
for b in $AVALIAR; do M="$M --modelo ${b}=${PREFIXO}_${b}/models/etapa_${EPOCAS}ep.pt"; done
[ -f model_cpu_tudo/models/etapa_${EPOCAS}ep.pt ] && M="$M --modelo 5_peso5=model_cpu_tudo/models/etapa_${EPOCAS}ep.pt"
$PY diagnostico/avaliar_mini.py --saida "$SAIDA" \
  --bases ing_1_reais ing_2_geradas ing_3_pares iam_pt_sub iam_pt_alinhado $M
echo "FIM ($(date +%H:%M))"
