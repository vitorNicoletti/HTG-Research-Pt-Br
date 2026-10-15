#!/usr/bin/env bash
# Treina (ou continua), avalia pelo protocolo e empacota o resultado. Para a
# maquina com GPU, dentro do tmux.
#
#   NOME=zona_vogais EPOCAS=30 PONTOS="21 30" ENVIAR=leo@192.168.0.10:~/repos/htg-tcc/ \
#     ARGS="--acento_separado --peso_zona 5 --zona vogais" bash scripts/treino_completo.sh
#
# NOME, EPOCAS, ARGS  os mesmos de scripts/treino_nuvem.sh, que faz o treino
# PONTOS   quais modelos avaliar, em numero de epocas (padrao: so o final).
#          Existem os pontos que o train.py guarda (2, 6, 11, 21, 41, 61), o
#          ultimo de cada chamada e o final
# ENVIAR   destino do scp; vazio = nao envia, so diz onde ficou o pacote
# DESTINO  pasta da copia de seguranca (padrao ./copias)
#
# Ao continuar um treino, o modelo que ja existia e guardado antes como
# ema_<N>ep.pt, porque o ema_ckpt.pt e sobrescrito.
set -u
cd "$(dirname "$0")/.."
: "${NOME:?defina NOME}" "${EPOCAS:?defina EPOCAS}"
export NOME EPOCAS ARGS=${ARGS:-} DESTINO=${DESTINO:-./copias}
PONTOS=${PONTOS:-$EPOCAS}
ENVIAR=${ENVIAR:-}
PY=${PY:-python -u}
M=./model_$NOME/models

if [ -f "$M/estado.pt" ]; then
  feitas=$($PY -c "import torch;print(torch.load('$M/estado.pt',map_location='cpu',weights_only=True)['epoch']+1)" | tail -1)
  if [ "$feitas" -lt "$EPOCAS" ] && [ ! -f "$M/ema_${feitas}ep.pt" ]; then
    cp "$M/ema_ckpt.pt" "$M/ema_${feitas}ep.pt" && echo "=== guardado o modelo de $feitas epocas em $M/ema_${feitas}ep.pt ==="
  fi
fi

bash scripts/treino_nuvem.sh || { echo "=== treino falhou, sem avaliacao ==="; exit 1; }

modelos=""
for n in $PONTOS; do
  if [ "$n" -eq "$EPOCAS" ]; then f=$M/ema_ckpt.pt
  elif [ -f "$M/ema_${n}ep.pt" ]; then f=$M/ema_${n}ep.pt
  else f=$M/ema_ep$((n - 1)).pt; fi
  [ -f "$f" ] || { echo "=== ponto de $n epocas nao existe ($f), pulando ==="; continue; }
  modelos="$modelos --modelo ${NOME}_${n}ep=$f"
done
[ -n "$modelos" ] || { echo "=== nenhum ponto para avaliar ==="; exit 1; }

SAIDA=avaliacao_${NOME}_${EPOCAS}ep_val
echo "=== avaliando$modelos ($(date +%H:%M)) ==="
# shellcheck disable=SC2086
$PY scripts/avaliar_pt.py --split val $modelos --bases "${BASE:-iam_pt_alinhado}" \
  --leitor modelos/leitor_iam_transformer.pt --device "${DEVICE:-cuda:0}" --saida "$SAIDA" \
  || { echo "=== avaliacao falhou ==="; exit 1; }
cat "$SAIDA/resumo.md"

PACOTE=${TMPDIR:-/tmp}/${NOME}_${EPOCAS}ep.tgz
tar czf "$PACOTE" "$SAIDA" "model_$NOME/config.jsonl" "model_$NOME/treino.log" || exit 1
echo "=== pacote em $PACOTE ($(du -h "$PACOTE" | cut -f1)) ==="
if [ -n "$ENVIAR" ]; then
  scp "$PACOTE" "$ENVIAR" && echo "=== enviado para $ENVIAR ===" || echo "=== scp falhou; envie a mao: scp $PACOTE $ENVIAR ==="
else
  echo "para enviar: scp $PACOTE leo@IP:~/repos/htg-tcc/"
fi
