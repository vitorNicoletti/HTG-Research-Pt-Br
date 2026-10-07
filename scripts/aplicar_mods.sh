#!/usr/bin/env bash
# Copia as correcoes de diffusionpen_mods/ para dentro do clone DiffusionPen/.
#
# Por que isto existe: o clone e gitignored, entao um `git pull` atualiza
# diffusionpen_mods/ e NAO atualiza DiffusionPen/. Sem este passo a maquina
# roda uma versao velha do train.py em silencio -- foi assim que um treino
# quebrou com ZeroDivisionError num `--sample_every 0` que ja estava corrigido
# no repositorio ha semanas.
#
# Rode depois de todo `git pull` e depois de todo clone novo.
#
#     bash scripts/aplicar_mods.sh            aplica
#     bash scripts/aplicar_mods.sh --conferir so relata o que esta diferente
set -u

RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
CLONE="${CLONE:-$RAIZ/DiffusionPen}"
CONFERIR=0
[ "${1:-}" = "--conferir" ] && CONFERIR=1

if [ ! -d "$CLONE" ]; then
  echo "clone nao encontrado em $CLONE"
  echo "git clone --depth 1 https://github.com/koninik/DiffusionPen.git DiffusionPen"
  exit 1
fi

status=0
while IFS= read -r rel; do
  origem="$RAIZ/diffusionpen_mods/$rel"
  destino="$CLONE/$rel"
  if [ ! -f "$destino" ]; then
    estado="AUSENTE no clone"
  elif cmp -s "$origem" "$destino"; then
    echo "  ok        $rel"
    continue
  else
    estado="DIFERENTE"
  fi
  status=1
  if [ "$CONFERIR" -eq 1 ]; then
    echo "  $estado  $rel"
  else
    mkdir -p "$(dirname "$destino")"
    cp "$origem" "$destino"
    echo "  copiado   $rel  ($estado)"
  fi
done < <(cd "$RAIZ/diffusionpen_mods" && find . -name '*.py' | sed 's|^\./||' | sort)

if [ "$CONFERIR" -eq 1 ]; then
  [ "$status" -eq 0 ] && echo "clone em dia" || echo "clone DESATUALIZADO: rode sem --conferir"
  exit "$status"
fi

# Verificacao do defeito concreto que motivou este script.
if grep -q 'args.sample_every > 0' "$CLONE/train.py"; then
  echo "clone em dia (guarda do --sample_every presente)"
else
  echo "ATENCAO: train.py no clone nao tem a guarda do --sample_every"
  exit 1
fi
