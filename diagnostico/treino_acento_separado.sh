#!/usr/bin/env bash
# Letra e diacritico separados no condicionamento (--acento_separado), no mesmo
# formato curto dos outros bracos: iam_pt_sub, 2 epocas, UNet inteiro, CPU.
#   C_separado        loss original   -> compara com 4_acentuadas
#   D_separado_peso5  peso 5          -> compara com 5_peso5 (model_cpu_tudo)
set -u
cd "$(dirname "$0")/.."
ARGS="--acento_separado" BRACOS="C_separado:iam_pt_sub" SEM_AVALIAR=1 \
  bash diagnostico/treino_ingredientes.sh || exit 1
ARGS="--acento_separado --peso_acento 5" BRACOS="D_separado_peso5:iam_pt_sub" \
  AVALIAR="4_acentuadas C_separado D_separado_peso5" SAIDA=diagnostico/resultados/acento_separado \
  bash diagnostico/treino_ingredientes.sh
