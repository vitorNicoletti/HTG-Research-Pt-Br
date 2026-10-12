#!/usr/bin/env bash
# Monta pacote_nuvem.tar com tudo que um treino fora daqui precisa e que nao
# esta no git: imagens do IAM, pesos do DiffusionPen do IAM, extrator de
# estilo, base de acentos e os modelos auxiliares. Cerca de 3 GB.
#
#     bash scripts/pacote_nuvem.sh [base]        (padrao: iam_pt_alinhado)
#
# O pacote e extraido na raiz do repositorio: tar xf pacote_nuvem.tar
# O IAM exige cadastro para uso em pesquisa: guarde o pacote em pasta privada.
set -eu
cd "$(dirname "$0")/.."
BASE=${1:-iam_pt_alinhado}
ITENS="DiffusionPen/iam_data/words
DiffusionPen/diffusionpen_iam_model_path/models/ckpt.pt
DiffusionPen/diffusionpen_iam_model_path/models/ema_ckpt.pt
DiffusionPen/style_models/iam_style_diffusionpen.pth
modelos/alinhador_iam.pt
modelos/leitor_iam_transformer.pt
$BASE"
for i in $ITENS; do [ -e "$i" ] || { echo "falta $i"; exit 1; }; done
n=$(find DiffusionPen/iam_data/words -name '*.png' | wc -l)
[ "$n" -ge 115000 ] || { echo "IAM incompleto nesta maquina: $n imagens"; exit 1; }
[ -f "$BASE/zona_vogais.npz" ] || echo "aviso: $BASE sem zona_vogais.npz (--zona vogais nao vai rodar)"
# shellcheck disable=SC2086
tar cf pacote_nuvem.tar $ITENS
ls -lh pacote_nuvem.tar
