#!/usr/bin/env bash
BC=/home/daniel/LovellAI/odyssey-ur5e/examples/ur5e-drugsort/browser_capture
TARGET=${TARGET:-150}
DEADLINE=$(( $(date +%s) + ${MAXWAIT:-10800} ))
while true; do
  n=$(ls -d $BC/out_full/inst_*/raw/ep*/meta.json 2>/dev/null | wc -l)
  now=$(date +%s)
  echo "$(date -u +%FT%TZ) completed=$n target=$TARGET" >> $BC/out_full/watch.log
  if [ "$n" -ge "$TARGET" ]; then echo "WATCH_DONE reason=target n=$n"; exit 0; fi
  if [ "$now" -ge "$DEADLINE" ]; then echo "WATCH_DONE reason=deadline n=$n"; exit 0; fi
  sleep 120
done
