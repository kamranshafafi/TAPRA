#!/bin/bash
# ns-3 validation sample: plans with Python, runs ns-3 in in-slot mode, compares.
# Several ns-3 runs in parallel (each ns-3 process uses one core).
#   bash ns3_validation_batch.sh            (defaults: 4 parallel runs, 30 slots)
#   PAR=6 SLOTS=40 bash ns3_validation_batch.sh
PAR=${PAR:-4}
SLOTS=${SLOTS:-30}
AI=$HOME/AI-DTOPA
NS3=$HOME/workspace/ns-3-dev
BIN=$(ls $NS3/build/scratch/ns3.*-tapra_validate-default | head -1)
OUT=$AI/results/ns3_validation.csv
mkdir -p $AI/runs $AI/results
cd $AI

# 1) plans (fast, sequential)
JOBS=()
for sc in "urban 500 20 homogeneous" "urban 500 20 heterogeneous" "suburban 500 20 homogeneous"; do
  set -- $sc
  for seed in 1 2 3; do
    for m in tapra vtopa full; do
      name="${1}${2}_${3}n_${4:0:4}_s${seed}_${m}"
      if [ ! -f runs/$name/params.txt ]; then
        python tapra_export.py --env $1 --area $2 --nodes $3 --traffic $4 --seed $seed \
               --method $m --slots $SLOTS --out runs/$name > /dev/null || { echo "export failed: $name"; continue; }
      fi
      JOBS+=("$name")
    done
  done
done
echo "${#JOBS[@]} plans ready; running ns-3 with $PAR in parallel"

# 2) ns-3 runs in parallel, skipping finished ones
run_one () {
  # ns-3.38 aborts now and then inside the Wi-Fi module (PhyEntity assertion,
  # association retry without BSSID). A crashed run is resumed from the first
  # unfinished slot with a new random stream, up to 8 attempts.
  name=$1
  d=$AI/runs/$name
  node_csv=$d/ns3_slot_node_inslot_servable_rts.csv
  flow_csv=$d/ns3_flow_inslot_servable_rts.csv
  for attempt in 1 2 3 4 5 6 7 8; do
    last=$(grep -o "slot [0-9]*/$SLOTS done" $d/ns3.log 2>/dev/null | tail -1 | sed 's#slot \([0-9]*\)/.*#\1#')
    last=${last:-0}
    if [ "$last" -ge "$SLOTS" ]; then echo "done $name"; return; fi
    # each snapshot resets the random stream, so a crash repeats with the same --run:
    # a resumed run must use a new one
    rng=$attempt
    if [ "$last" -gt 0 ]; then
      rng=$((attempt + 1))
      # drop rows of the slot that was cut off, keep the finished slots
      python3 - "$node_csv" "$flow_csv" "$last" <<'PY'
import sys, pandas as pd
for f in sys.argv[1:3]:
    try:
        df = pd.read_csv(f)
        df[df.slot <= int(sys.argv[3])].to_csv(f, index=False)
    except FileNotFoundError:
        pass
PY
    else
      rm -f $d/ns3.log
    fi
    (cd $NS3 && LD_LIBRARY_PATH=$NS3/build/lib:$LD_LIBRARY_PATH \
      "$BIN" $(cat $d/params.txt) --slots=$SLOTS --mode=inslot \
      --firstSlot=$((last + 1)) --run=$rng >> $d/ns3.log 2>&1) \
      || echo "retry $name from slot $((last + 1)) (attempt $attempt)"
  done
  echo "FAILED $name after 8 attempts (see $d/ns3.log)"
}
export -f run_one; export AI NS3 BIN SLOTS
printf "%s\n" "${JOBS[@]}" | xargs -P $PAR -I{} bash -c 'run_one {}'

# 3) comparisons -> one summary file
rm -f $OUT
for name in "${JOBS[@]}"; do
  python compare_ns3.py runs/$name inslot_servable_rts --summary $OUT > /dev/null 2>&1 || echo "compare failed: $name"
done
echo "summary: $OUT"
