"""Rev. 2, Reviewer 1 comment 3: separate UE motion prediction from UAV travel,
same search budgets as TAPRA. Appends to results/rev2_ablation.csv (restartable)."""
import csv, os, sys, time
from multiprocessing import Pool
import batch
batch.METHODS.update({
    "Ablation: no motion prediction": ("adaptive", {"predict_motion": False}),
    "Ablation: no UAV travel": ("adaptive", {"model_travel": False}),
})
OUT = "results/rev2_ablation.csv"
if __name__ == "__main__":
    names = list(sys.argv[1:]) or ["Ablation: no motion prediction", "Ablation: no UAV travel"]
    done = set()
    if os.path.exists(OUT):
        for r in csv.DictReader(open(OUT)):
            done.add((r["env"], int(r["area"]), int(r["nodes"]), r["traffic"], int(r["seed"]), r["method"]))
    jobs = [(e, a, n, t, s, m) for (e, a, n, t) in batch.SCENARIOS for s in range(1, 11)
            for m in names if (e, a, n, t, s, m) not in done]
    jobs.sort(key=lambda j: -j[1])          # big cities first
    new = not os.path.exists(OUT)
    t0 = time.time()
    with open(OUT, "a", newline="") as f, Pool(2) as pool:
        w = csv.DictWriter(f, fieldnames=batch.FIELDS)
        if new: w.writeheader()
        for i, row in enumerate(pool.imap_unordered(batch.run_one, jobs), 1):
            if row: w.writerow(row); f.flush()
            print(f"{i}/{len(jobs)} {(time.time()-t0)/60:.1f} min", flush=True)
