"""Rev. 2, Reviewer 1 comment 3: a learning baseline with the SAME reports and the SAME
objective as TAPRA.
  - Reports: UE positions at the slot start, velocities estimated from the previous report,
    demands (exactly what TAPRA receives).
  - Objective: TAPRA's slot-average served demand (Evaluator.evaluate: motion prediction,
    UAV travel, calibrated capacity model, admission). Reward of each move = served share.
  - Agent: double DQN, MLP 2 x 128 (numpy, Adam), 7 moves (+-x, +-y, +-z, stay) of 10 m,
    floor(v_max * tau / 10) = 7 moves per slot (same speed limit), no sphere constraint.
  - Trained on separate cities (seeds 1001+), one agent per scenario; tested on seeds 1-10.
  python rev2_dqn.py train <env> <area> <nodes> <traffic> [episodes]
  python rev2_dqn.py test                      -> results/rev2_dqn.csv"""
import os, pickle, random, sys, time, csv
import numpy as np
from link_model import LinkModel
from optimizer import Config, Evaluator, run_trajectory, realised_inslot
from scenario_gen import generate
from tapra_export import uav_track

MOVES = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [0, 0, 0]], float)
STEP = 10.0

class MLP:
    def __init__(self, n_in, n_out, h=128, seed=0):
        r = np.random.default_rng(seed)
        self.p = [r.normal(0, np.sqrt(2 / n_in), (n_in, h)), np.zeros(h),
                  r.normal(0, np.sqrt(2 / h), (h, h)), np.zeros(h),
                  r.normal(0, np.sqrt(1 / h), (h, n_out)), np.zeros(n_out)]
        self.m = [np.zeros_like(x) for x in self.p]; self.v = [np.zeros_like(x) for x in self.p]; self.t = 0
    def fwd(self, x, keep=False):
        W1, b1, W2, b2, W3, b3 = self.p
        h1 = np.maximum(0, x @ W1 + b1); h2 = np.maximum(0, h1 @ W2 + b2); y = h2 @ W3 + b3
        if keep: self.cache = (x, h1, h2)
        return y
    def step(self, dy, lr=1e-3):
        x, h1, h2 = self.cache; W1, b1, W2, b2, W3, b3 = self.p
        g3 = h2.T @ dy; gb3 = dy.sum(0); d2 = (dy @ W3.T) * (h2 > 0)
        g2 = h1.T @ d2; gb2 = d2.sum(0); d1 = (d2 @ W2.T) * (h1 > 0)
        g1 = x.T @ d1; gb1 = d1.sum(0)
        self.t += 1
        for i, g in enumerate([g1, gb1, g2, gb2, g3, gb3]):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g; self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            mh = self.m[i] / (1 - 0.9 ** self.t); vh = self.v[i] / (1 - 0.999 ** self.t)
            self.p[i] -= lr * mh / (np.sqrt(vh) + 1e-8)
    def copy_from(self, o): self.p = [x.copy() for x in o.p]

def observe(ev, q, area, vmax_ue=10.0):
    _, los = ev.lm.snr_db(q, ev.nodes_now, ev.boxes)
    P = ev.nodes_now
    vel = (ev.node_sets[-1] - ev.node_sets[0]) / max(ev.times[-1] - ev.times[0], 1e-9) if len(ev.times) > 1 else np.zeros_like(P)
    c = ev.cfg
    uav = [q[0] / area, q[1] / area, (q[2] - c.z_min) / (c.z_max - c.z_min)]
    return np.concatenate([uav, ((P[:, :2] - q[:2]) / area).ravel(), (vel[:, :2] / vmax_ue).ravel(),
                           ev.demands / 26.0, los.astype(float)])

class DQNPolicy:
    def __init__(self, net, area): self.net, self.area = net, area
    def act(self, ev, q_prev, nodes, demands, radius):
        c = ev.cfg
        q = np.array([0.0, 0.0, 65.0]) if q_prev is None else np.array(q_prev, float)
        while not ev.valid(q): q[2] += 5
        for _ in range(int(c.v_max * c.tau // STEP)):
            a = int(np.argmax(self.net.fwd(observe(ev, q, self.area)[None])[0]))
            cand = q + MOVES[a] * STEP
            if ev.valid(cand): q = cand
        ev.evaluate(q)       # counts the final evaluation
        return q

def train(env, area, n, traffic, episodes=200, slots=30, seed0=1001, max_seconds=None):
    random.seed(0); cfg = Config(); lm = LinkModel(env=env)
    n_in = 3 + 6 * n
    net, tgt = MLP(n_in, len(MOVES)), MLP(n_in, len(MOVES)); tgt.copy_from(net)
    buf, gamma, B = [], 0.9, 64
    n_moves = int(cfg.v_max * cfg.tau // STEP)
    total, decay = 0, episodes * slots * n_moves * 0.6
    ck = f"results/dqn_{env}_{area}_{n}_{traffic}.ckpt"; ep0 = 0
    if os.path.exists(ck):
        st = pickle.load(open(ck, "rb"))
        net.p, net.m, net.v, net.t = st["p"], st["m"], st["v"], st["t"]; tgt.p = st["tp"]
        buf, total, ep0 = st["buf"], st["total"], st["ep"]; random.setstate(st["rs"])
    log = open(f"results/dqn_{env}{area}_{n}_{traffic}.log", "a" if ep0 else "w"); t0 = time.time()
    for ep in range(ep0, episodes):
        if max_seconds and time.time() - t0 > max_seconds:
            pickle.dump(dict(p=net.p, m=net.m, v=net.v, t=net.t, tp=tgt.p, buf=buf, total=total, ep=ep,
                             rs=random.getstate()), open(ck, "wb"))
            return None
        sc = generate(env, area, n, traffic, slots, cfg.tau, seed=seed0 + ep % 50)
        ev = Evaluator(lm, sc["Buildings"], area, cfg)
        T = np.array([x["traffic_demand_mbps"] for x in sc["UENodes"]], float)
        P = [np.array(x["position"], float) for x in sc["UENodes"]]
        q = np.array([random.uniform(-area / 4, area / 4), random.uniform(-area / 4, area / 4), 65.0])
        while not ev.valid(q): q[2] += 5
        ep_r = 0.0
        for k in range(slots):
            nodes = np.array([p[k] for p in P]); prev = np.array([p[k - 1] for p in P]) if k else None
            ev.set_slot(nodes, prev, q.copy(), T)
            for i in range(n_moves):
                eps = max(0.05, 1 - total / decay)
                o = observe(ev, q, area)
                a = random.randrange(len(MOVES)) if random.random() < eps else int(np.argmax(net.fwd(o[None])[0]))
                cand = q + MOVES[a] * STEP
                q2 = cand if ev.valid(cand) else q
                r = ev.evaluate(q2)[0] / T.sum()
                o2 = observe(ev, q2, area)
                buf.append((o, a, r, o2)); buf = buf[-50000:] if len(buf) > 60000 else buf
                q, ep_r, total = q2, ep_r + r, total + 1
                if len(buf) >= 1000:
                    s, a_, r_, s2 = map(np.array, zip(*random.sample(buf, B)))
                    a2 = np.argmax(net.fwd(s2), 1)
                    y = r_ + gamma * tgt.fwd(s2)[np.arange(B), a2]
                    qv = net.fwd(s, keep=True)
                    err = qv[np.arange(B), a_] - y
                    dy = np.zeros_like(qv); dy[np.arange(B), a_] = np.clip(err, -1, 1) / B   # Huber
                    net.step(dy)
                if total % 1000 == 0: tgt.copy_from(net)
        print(f"ep {ep} reward {ep_r / (slots * n_moves):.3f} eps {eps:.2f} {time.time() - t0:.0f}s", file=log, flush=True)
    f = f"results/dqn_{env}_{area}_{n}_{traffic}.pkl"
    pickle.dump({"p": net.p, "n_in": n_in, "area": area}, open(f, "wb"))
    return f

def load(f):
    d = pickle.load(open(f, "rb")); net = MLP(d["n_in"], len(MOVES)); net.p = d["p"]
    return DQNPolicy(net, d["area"])

def test_one(job):
    env, area, n, tr, seed = job
    cfg = Config(); lm = LinkModel(env=env)
    sc = generate(env, area, n, tr, 50, cfg.tau, seed=seed)
    pol = load(f"results/dqn_{env}_{area}_{n}_{tr}.pkl")
    res = run_trajectory(sc, lm, cfg, "policy", policy=pol)
    m = realised_inslot(sc, res, lm, cfg, uav_track, details=True)
    out = [dict(k=r.k, q=r.q, admitted=r.admitted, served=r.served_mbps, demand=r.demand_mbps,
                seconds=r.seconds, evals=r.evals, mode=r.mode) for r in res]
    pickle.dump(out, open(f"results/traj/{env}_{area}_{n}_{tr}_s{seed}_DQN_TAPRA_objective.pkl", "wb"))
    return dict(env=env, area=area, nodes=n, traffic=tr, seed=seed, method="DQN (TAPRA objective)",
                planned_share=np.mean([r.served_mbps / r.demand_mbps for r in res]),
                realised_share=m["realised_share"], node_outage=m["node_outage"], jain=m["jain"],
                sec_per_slot=np.mean([r.seconds for r in res[1:]]), evals_per_slot=np.mean([r.evals for r in res[1:]]))

if __name__ == "__main__":
    if sys.argv[1] == "train":
        e, a, n, t = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
        ms = float(os.environ["MAXSEC"]) if os.environ.get("MAXSEC") else None
        print(train(e, a, n, t, int(sys.argv[6]) if len(sys.argv) > 6 else 200, max_seconds=ms))
    else:
        import batch
        from multiprocessing import Pool
        jobs = [(e, a, n, t, s) for (e, a, n, t) in batch.SCENARIOS for s in range(1, 11)
                if os.path.exists(f"results/dqn_{e}_{a}_{n}_{t}.pkl")]
        with Pool(2) as p, open("results/rev2_dqn.csv", "w", newline="") as f:
            w = None
            for r in p.imap_unordered(test_one, jobs):
                if w is None: w = csv.DictWriter(f, list(r)); w.writeheader()
                w.writerow(r); f.flush()
