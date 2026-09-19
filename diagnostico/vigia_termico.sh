#!/usr/bin/env bash
# Mata o treino se a CPU passar do limite. Roda ao lado da corrida, sem
# interferir nela.
#
#   ./scripts/vigia_termico.sh <sessao_tmux> [limite_C] [log]
#
# Checa a cada 60 s. Precisa de 3 leituras seguidas acima do limite para agir,
# para nao derrubar a corrida por um pico isolado. Registra todas as leituras,
# entao de manha da para ver o historico termico mesmo que nada aconteca.
set -u

SESSAO="${1:-cpuref}"
LIMITE="${2:-85}"
LOG="${3:-./vigia_termico.log}"
ACIMA=0

le_temp() {
  for z in /sys/class/thermal/thermal_zone*; do
    if [ "$(cat "$z/type" 2>/dev/null)" = "x86_pkg_temp" ]; then
      echo $(( $(cat "$z/temp") / 1000 ))
      return
    fi
  done
  # sem x86_pkg_temp: usa a maior zona disponivel
  local max=0 t
  for z in /sys/class/thermal/thermal_zone*; do
    t=$(cat "$z/temp" 2>/dev/null) || continue
    t=$((t / 1000))
    [ "$t" -gt "$max" ] && max=$t
  done
  echo "$max"
}

echo "=== vigia iniciado $(date '+%F %T') | sessao=$SESSAO | limite=${LIMITE}C ===" >> "$LOG"

while tmux has-session -t "$SESSAO" 2>/dev/null; do
  T=$(le_temp)
  echo "$(date '+%F %T') ${T}C" >> "$LOG"

  if [ "$T" -ge "$LIMITE" ]; then
    ACIMA=$((ACIMA + 1))
    echo "  ACIMA DO LIMITE ($ACIMA/3)" >> "$LOG"
    if [ "$ACIMA" -ge 3 ]; then
      echo "$(date '+%F %T') MATANDO a sessao $SESSAO -- ${T}C por 3 leituras" >> "$LOG"
      tmux kill-session -t "$SESSAO" 2>/dev/null
      exit 1
    fi
  else
    ACIMA=0
  fi

  sleep 60
done

echo "=== sessao $SESSAO terminou sozinha $(date '+%F %T') | vigia encerrado ===" >> "$LOG"
