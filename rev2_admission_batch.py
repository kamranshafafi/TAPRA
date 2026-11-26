"""Run rev2_admission.analyse over all main scenarios and the small best-subset cases.
Per-job results cached in results/adm/ (resumable). Output: results/rev2_admission.csv"""
import os, pickle, sys
from multiprocessing import Pool
import pandas as pd
from rev2_admission import analyse

MAIN = [("suburban", 500, 20), ("urban", 500, 20), ("urban", 1000, 50)]
SMALL = [("urban", 500, 8), ("urban", 500, 10), ("urban", 500, 12)]
TRAF = ["homogeneous", "heterogeneous"]
OUT = "results/adm"

def job(a):
    env, area, n, tr, seed, best = a
    f = f"{OUT}/{env}_{area}_{n}_{tr}_s{seed}.pkl"
    if not os.path.exists(f):
        rows = analyse(env, area, n, tr, seed, best=best)
        pickle.dump(rows, open(f, "wb"))
    return f

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    jobs = []
    if which in ("all", "small"):
        jobs += [(e, a, n, t, s, True) for e, a, n in SMALL for t in TRAF for s in range(1, 11)]
    if which in ("all", "main"):
        jobs += [(e, a, n, t, s, False) for e, a, n in MAIN for t in TRAF for s in range(1, 11)]
    with Pool(2) as p:
        for i, f in enumerate(p.imap_unordered(job, jobs), 1):
            print(i, len(jobs), f, flush=True)
    rows = []
    for f in sorted(os.listdir(OUT)):
        rows += pickle.load(open(os.path.join(OUT, f), "rb"))
    pd.DataFrame(rows).to_csv("results/rev2_admission.csv", index=False)
    print("written results/rev2_admission.csv", len(rows))
