#!/bin/bash
# Rev. 2 ns-3 runs. Put this file and rev2_policy_export.py in ~/AI-DTOPA and run:
#   PAR=4 bash rev2_ns3.sh cont       Part B, start now (longest, independent)
#   PAR=4 bash rev2_ns3.sh adm        Part A, after I confirm the threshold
#   bash rev2_ns3.sh sum              summaries + tar to send back
# PAR = parallel ns-3 processes (one core each).
# Part A (Reviewer 1.1): admission rules at fixed TAPRA waypoints, in-slot snapshots.
# Part B (Reviewer 1.2): continuous simulations over the whole 250 s mission, UEs and
#   UAV moving continuously, association and queues kept across slots.
PAR=${PAR:-4}
SLOTS=50
AI=$HOME/AI-DTOPA
NS3=$HOME/workspace/ns-3-dev
BIN=$(ls $NS3/build/scratch/ns3.*-tapra_validate-default | head -1)
cd $AI
WHAT=${1:-sum}
THR=${THR:-11}

run_inslot () {   # same resume logic as ns3_validation_batch.sh
  name=$1; d=$AI/runs/$name
  node_csv=$d/ns3_slot_node_inslot_servable_rts.csv; flow_csv=$d/ns3_flow_inslot_servable_rts.csv
  for attempt in 1 2 3 4 5 6 7 8; do
    last=$(grep -o "slot [0-9]*/$SLOTS done" $d/ns3.log 2>/dev/null | tail -1 | sed 's#slot \([0-9]*\)/.*#\1#'); last=${last:-0}
    if [ "$last" -ge "$SLOTS" ]; then echo "done $name"; return; fi
    rng=$attempt
    if [ "$last" -gt 0 ]; then rng=$((attempt + 1))
      python3 -c "
import sys, pandas as pd
for f in sys.argv[1:3]:
    try:
        df = pd.read_csv(f); df[df.slot <= int(sys.argv[3])].to_csv(f, index=False)
    except FileNotFoundError: pass" "$node_csv" "$flow_csv" "$last"
    else rm -f $d/ns3.log; fi
    (cd $NS3 && LD_LIBRARY_PATH=$NS3/build/lib:$LD_LIBRARY_PATH "$BIN" $(cat $d/params.txt) \
      --slots=$SLOTS --mode=inslot --firstSlot=$((last + 1)) --run=$rng >> $d/ns3.log 2>&1) \
      || echo "retry $name from slot $((last + 1)) (attempt $attempt)"
  done
  echo "FAILED $name"
}
run_cont () {     # one simulation; on a crash, start again with a new random stream
  name=$1; d=$AI/runs/$name
  if [ -s $d/ns3_flow_continuous_servable_rts.csv ] && grep -q "^0," $d/ns3_flow_continuous_servable_rts.csv; then echo "done $name"; return; fi
  for attempt in 1 2 3 4 5; do
    (cd $NS3 && LD_LIBRARY_PATH=$NS3/build/lib:$LD_LIBRARY_PATH "$BIN" $(cat $d/params.txt) \
      --slots=$SLOTS --mode=continuous --queueMaxDelay=10 --maxMissedBeacons=1000000 --run=$attempt > $d/ns3_cont.log 2>&1) \
      && { echo "done $name (attempt $attempt)"; return; }
    echo "retry $name (attempt $attempt)"
  done
  echo "FAILED $name"
}
export -f run_inslot run_cont; export AI NS3 BIN SLOTS

if [ "$WHAT" = adm ]; then
echo "== Part A: admission rules"
python rev2_policy_export.py --policies all,snr$THR || exit 1
ls -d runs/*_tapra_adm-* | xargs -n1 basename | xargs -P $PAR -I{} bash -c 'run_inslot {}'
fi
if [ "$WHAT" = cont ]; then
echo "== Part B: continuous runs"
ls -d runs/*_tapra runs/*_vtopa runs/*_full | xargs -n1 basename | xargs -P $PAR -I{} bash -c 'run_cont {}'
fi
[ "$WHAT" = sum ] || exit 0

echo "== Summaries"
rm -f results/rev2_ns3_admission.csv results/rev2_ns3_continuous.csv
for d in runs/*_tapra runs/*_tapra_adm-*; do
  python compare_ns3.py $d inslot_servable_rts --summary results/rev2_ns3_admission.csv > /dev/null 2>&1
done
for d in runs/*_tapra runs/*_vtopa runs/*_full; do
  python compare_ns3.py $d continuous_servable_rts --summary results/rev2_ns3_continuous.csv > /dev/null 2>&1
done
tar czf rev2_ns3_results.tar.gz results/rev2_ns3_*.csv runs/*/ns3_*_servable_rts.csv
echo "send me rev2_ns3_results.tar.gz"
