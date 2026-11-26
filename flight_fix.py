import pickle, csv, numpy as np, batch
from scenario_gen import generate
from optimizer import Config
from los_geometry import boxes_from_buildings, los_vector
from rev2_cache import path, METHODS
with open("results/flight_fix.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["env", "area", "traffic", "seed", "method", "moving_segments", "through_building", "within_clearance", "length_m", "max_move_m"])
    for (e, a, n, t) in batch.SCENARIOS:
        for s in range(1, 11):
            sc = generate(e, a, n, t, 50, 5.0, seed=s); B = boxes_from_buildings(sc["Buildings"]); H = B.copy(); H[:, 4] += Config().clearance
            for m in METHODS:
                q = np.array([r["q"] for r in pickle.load(open(path(e, a, n, t, s, m), "rb"))])
                seg = hit = hc = 0; L = 0.0; mx = 0.0
                for p0, p1 in zip(q[:-1], q[1:]):
                    d = np.linalg.norm(p1 - p0); mx = max(mx, d)
                    if d < 1e-6: continue
                    seg += 1; L += d
                    hit += int(not los_vector(p0, p1[None, :], B)[0]); hc += int(not los_vector(p0, p1[None, :], H)[0])
                w.writerow([e, a, t, s, m, seg, hit, hc, round(L, 1), round(mx, 2)])
