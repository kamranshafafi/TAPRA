#!/bin/bash
# Rev. 2: two more random streams for every continuous ns-3 run (27 plans x 2 = 54 runs), so that each
# city and method is the mean of three independent simulations. Put it in ~/AI-DTOPA and run:
#   PAR=4 bash rev2_ns3_reps.sh          (resumable: finished runs are skipped)
# It writes runs/<name>_rep2 and runs/<name>_rep3 (same plan, other random stream) and packs
# rev2_ns3_reps.tar.gz to send back.
PAR=${PAR:-4}
SLOTS=50
AI=$HOME/AI-DTOPA
NS3=$HOME/workspace/ns-3-dev
BIN=$(ls $NS3/build/scratch/ns3.*-tapra_validate-default | head -1)
cd $AI

make_rep () {     # copy the plan of runs/$1 to runs/$1_rep$2
  src=$AI/runs/$1; dst=$AI/runs/$1_rep$2
  mkdir -p $dst
  for f in $src/*.csv $src/scenario.json; do
    case $(basename $f) in ns3_*) ;; *) cp $f $dst/ ;; esac
  done
  sed "s#$src#$dst#g" $src/params.txt > $dst/params.txt
}
run_rep () {      # one continuous simulation with random streams 10*rep+1, 10*rep+2, ...
  name=$1; rep=$2; d=$AI/runs/${name}_rep$rep
  if [ -s $d/ns3_flow_continuous_servable_rts.csv ] && grep -q "^0," $d/ns3_flow_continuous_servable_rts.csv; then
    echo "done ${name}_rep$rep"; return; fi
  for attempt in 1 2 3 4 5; do
    (cd $NS3 && LD_LIBRARY_PATH=$NS3/build/lib:$LD_LIBRARY_PATH "$BIN" $(cat $d/params.txt) \
      --slots=$SLOTS --mode=continuous --queueMaxDelay=10 --maxMissedBeacons=1000000 \
      --run=$((10 * rep + attempt)) > $d/ns3_cont.log 2>&1) \
      && { echo "done ${name}_rep$rep (attempt $attempt)"; return; }
    echo "retry ${name}_rep$rep (attempt $attempt)"
  done
  echo "FAILED ${name}_rep$rep"
}
export -f run_rep; export AI NS3 BIN SLOTS

BASES=$(ls -d runs/*_tapra runs/*_vtopa runs/*_full | xargs -n1 basename)
for b in $BASES; do for r in 2 3; do make_rep $b $r; done; done
echo "== $(echo $BASES | wc -w) plans x 2 random streams, $PAR in parallel"
for r in 2 3; do for b in $BASES; do echo "$b $r"; done; done | xargs -P $PAR -L1 bash -c 'run_rep $0 $1'

echo "== Summaries"
rm -f results/rev2_ns3_continuous_reps.csv
for b in $BASES; do for r in 2 3; do
  python compare_ns3.py runs/${b}_rep$r continuous_servable_rts --summary results/rev2_ns3_continuous_reps.csv > /dev/null 2>&1
done; done
tar czf rev2_ns3_reps.tar.gz results/rev2_ns3_continuous_reps.csv runs/*_rep[23]/ns3_*continuous_servable_rts.csv
grep -c "" results/rev2_ns3_continuous_reps.csv
echo "send me rev2_ns3_reps.tar.gz"
