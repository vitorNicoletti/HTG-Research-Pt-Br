#!/usr/bin/env bash
# Dois fine-tunes iguais na CPU, a partir dos pesos do IAM, na base iam_pt_sub
# com peso 5 no acento. A unica diferenca e o que treina:
#   tudo   o UNet inteiro (a receita do model_iam_pt_peso5)
#   texto  so a atencao cruzada e a text_lin (--treinar_so texto)
# Roda em etapas alternadas e guarda o modelo de cada etapa em
# model_cpu_<braco>/models/etapa_<epocas>ep.pt, para comparar os dois com o
# mesmo numero de passos (521 passos por epoca). O cache do VAE e do extrator de
# estilo (congelados) e dividido pelos dois bracos: ${PREFIXO}_cache.pt.
#
#   bash diagnostico/treino_texto_vs_tudo.sh "2 4 8"     # epocas acumuladas de cada etapa
set -u
cd "$(dirname "$0")/.."
ETAPAS=${1:-"2 4 8"}
EXTRA=${EXTRA:-}            # ex.: EXTRA="--max_samples 64" para um teste rapido
PREFIXO=${PREFIXO:-model_cpu}

treinar() {  # braco, epocas a rodar agora
  local braco=$1 n=$2 salvar=./${PREFIXO}_$1
  local inicio="--pretrained_path ./DiffusionPen/diffusionpen_iam_model_path/models"
  [ -f "$salvar/models/estado.pt" ] && inicio="--load_check True"
  python -u DiffusionPen/train.py --dataset iam_acentuado --model_name diffusionpen --sample_every 0 \
    --save_path "$salvar" --dataset_folder ./iam_pt_sub --max_samples 0 --iam_originais 0.0 \
    --style_path ./DiffusionPen/style_models/iam_style_diffusionpen.pth \
    --stable_dif_path stable-diffusion-v1-5/stable-diffusion-v1-5 \
    --lr 2e-05 --batch_size 32 --adamw_eps 1e-06 --clip_grad_norm 1.0 --ema_beta 0.995 --ema_inicio 2000 \
    --texto_max_len 40 --device cpu --num_workers 4 --save_every_steps 500 --abort_after 300 \
    --peso_acento 5 --peso_zona 1 --zona vazia --treinar_so "$braco" \
    --cache_congelados "./${PREFIXO}_cache.pt" --epochs "$n" $inicio $EXTRA
}

feitas=0
for alvo in $ETAPAS; do
  for braco in tudo texto; do
    echo "=== $braco: de $feitas para $alvo epocas ($(date +%H:%M)) ==="
    treinar "$braco" $((alvo - feitas)) || { echo "FALHOU $braco"; exit 1; }
    cp "./${PREFIXO}_$braco/models/ema_ckpt.pt" "./${PREFIXO}_$braco/models/etapa_${alvo}ep.pt"
    echo "=== etapa pronta: ${PREFIXO}_$braco/models/etapa_${alvo}ep.pt ($(date +%H:%M)) ==="
  done
  feitas=$alvo
done
echo "FIM"
