"""
TAPRA optimiser, version with a slot-average objective.

Objective for slot k (lexicographic):
  1. maximise the demand served on average over the coming slot,
  2. then maximise the airtime margin.
"On average over the slot" (cfg.slot_average = True): a candidate position q is scored
at a few instants s in the slot (default 0.25, 0.5, 0.75 of tau). At each instant
  - the UAV is on its straight flight from the previous position towards q at v_max,
  - each node is at its position predicted from its last two reports
    (constant velocity; k = 1: no motion known, nodes assumed still).
The admitted set is fixed for the whole slot (it is announced at the slot start) and
chosen best-link-first to maximise the slot-average served demand.
With cfg.slot_average = False the objective is the old snapshot at the slot start,
with the UAV already at q (kept for the ablation study).

Search (no gradients; LoS and MCS change in jumps): coarse 3D grid + pattern search.
Adaptive update for k > 1: momentum/traffic prediction, a small screen (remembered
good spots + a local grid in the reachable disc), short pattern search; full search
when the served share drops by more than `trigger_drop` or every `period` slots.
"""
import time
from dataclasses import dataclass

import numpy as np

from link_model import C
from los_geometry import boxes_from_buildings


@dataclass
class Config:
    z_min: float = 30.0
    z_max: float = 100.0
    v_max: float = 15.0
    tau: float = 5.0
    clearance: float = 5.0
    grid_xy: int = 21
    grid_z: tuple = (30, 40, 50, 60, 70, 80, 90, 100)
    refine_top: int = 3
    step0_full: float = 20.0
    step0_adapt: float = 10.0
    step_min: float = 1.0
    max_evals_full: int = 400
    max_evals_adapt: int = 60
    momentum: float = 0.7
    traffic_weight: float = 0.3
    trigger_drop: float = 0.05
    period: int = 10
    tie_tol_mbps: float = 0.01
    cache_size: int = 15
    local_grid: int = 5
    slot_average: bool = True
    predict_motion: bool = True  # ablation (rev. 2): predict UE motion within the slot
    model_travel: bool = True    # ablation (rev. 2): model the UAV flight within the slot
    instants: tuple = (0.25, 0.5, 0.75)
    objective: str = "tapra"     # "tapra" or "vtopa" (max LoS count, then min mean distance)
    centroid_z: float = 60.0     # altitude of the traffic-centroid baseline
    use_cache: bool = True       # ablation: remembered good spots in the adaptive screen
    use_trigger: bool = True     # ablation: re-run the full search when service drops
    pos_sigma: float = 0.0       # assumed std of reported positions (m); > 0 makes the
                                 # objective average over sampled true positions
    pos_samples: int = 4         # samples per instant when pos_sigma > 0
    reach_fix: bool = True       # rev. 2: project the local-grid candidates onto the reachable
                                 # region (False reproduces the R1 results, where 1% of the slots
                                 # moved up to 100 m)
    map_match: bool = False      # push reported node positions that fall inside a
                                 # building out to the nearest street (GPS error)


@dataclass
class SlotResult:
    k: int
    q: np.ndarray
    served_mbps: float          # objective value (slot average or snapshot)
    demand_mbps: float
    margin: float
    per_node_served: np.ndarray  # slot-average served per node (admitted set)
    per_node_static: np.ndarray  # served per node at the slot start with UAV at q
    los: np.ndarray              # LoS at the slot start with UAV at q
    admitted: np.ndarray
    evals: int
    seconds: float
    mode: str


class Evaluator:
    def __init__(self, link_model, buildings, area, cfg: Config):
        self.lm, self.cfg = link_model, cfg
        self.boxes = boxes_from_buildings(buildings)
        self.half = area / 2.0
        self.n_evals = 0
        self.cache = []

    # ------------------------------------------------------------ slot context
    def set_slot(self, nodes_now, nodes_prev, q_prev, demands):
        c = self.cfg
        self.demands = np.asarray(demands, float)
        self.q_prev = None if q_prev is None else np.asarray(q_prev, float)
        self.nodes_now = np.asarray(nodes_now, float)
        if c.slot_average:
            vel = np.zeros_like(self.nodes_now) if nodes_prev is None else \
                (self.nodes_now - np.asarray(nodes_prev, float)) / c.tau
            vel[:, 2] = 0.0
            self.times = [f * c.tau for f in c.instants]
            if not c.predict_motion:
                vel = np.zeros_like(vel)
            self.node_sets = [self.nodes_now + vel * s for s in self.times]
        else:
            self.times, self.node_sets = [0.0], [self.nodes_now]
        if c.pos_sigma > 0:
            rng = np.random.default_rng(int(abs(self.nodes_now[:, :2]).sum() * 1000) % (2**32))
            times, sets = [], []
            for s, P in zip(self.times, self.node_sets):
                for _ in range(c.pos_samples):
                    J = P.copy()
                    J[:, :2] += rng.normal(0.0, c.pos_sigma, J[:, :2].shape)
                    times.append(s)
                    sets.append(J)
            self.times, self.node_sets = times, sets

    def uav_at(self, q, s):
        if self.q_prev is None or not self.cfg.slot_average or not self.cfg.model_travel:
            return q
        d = np.linalg.norm(q - self.q_prev)
        if d < 1e-9:
            return q
        return self.q_prev + (q - self.q_prev) * min(1.0, self.cfg.v_max * s / d)

    def valid(self, q):
        c = self.cfg
        if not (-self.half <= q[0] <= self.half and -self.half <= q[1] <= self.half
                and c.z_min <= q[2] <= c.z_max):
            return False
        b = self.boxes
        over = (q[0] >= b[:, 0]) & (q[0] <= b[:, 1]) & (q[1] >= b[:, 2]) & (q[1] <= b[:, 3])
        return not np.any(over & (q[2] < b[:, 4] + c.clearance))

    # ------------------------------------------------------------ objective
    def _goodputs(self, q):
        return [self.lm.goodput(self.lm.snr_db(self.uav_at(q, s), P, self.boxes)[0])
                for s, P in zip(self.times, self.node_sets)]

    def evaluate(self, q):
        """Returns (served, margin, per_node_served, admitted mask)."""
        self.n_evals += 1
        T, lm = self.demands, self.lm
        G = self._goodputs(q)                       # list over instants, each (N,)
        J = len(G)
        ok_any = np.any([g > 0 for g in G], axis=0)
        per = np.zeros(len(T))
        adm_mask = np.zeros(len(T), bool)
        if not ok_any.any():
            return 0.0, 0.0, per, adm_mask
        # order: least average airtime per Mbit/s first (inf where no link at an instant)
        with np.errstate(divide="ignore"):
            cost = np.mean([np.where(g > 0, 1.0 / np.maximum(g, 1e-12), 1e3) for g in G], axis=0)
        cand = np.where(ok_any)[0]
        order = cand[np.argsort(cost[cand])]
        best_val, best_m = -1.0, 0
        for m in range(1, len(order) + 1):
            adm = order[:m]
            val = 0.0
            for g in G:
                ok = adm[g[adm] > 0]
                if len(ok):
                    val += T[ok].sum() * lm.fraction(len(ok), (T[ok] / g[ok]).sum())
            val /= J
            if val > best_val + 1e-9:
                best_val, best_m = val, m
        adm = order[:best_m]
        adm_mask[adm] = True
        margins = []
        for g in G:
            ok = adm[g[adm] > 0]
            if len(ok):
                L = (T[ok] / g[ok]).sum()
                per[ok] += T[ok] * lm.fraction(len(ok), L) / J
                margins.append(lm.usable(len(ok)) / L)
        return float(best_val), float(np.mean(margins)) if margins else 0.0, per, adm_mask

    def static(self, q, admitted):
        """Served per node at the slot start with the UAV at q, for the admitted set."""
        snr, los = self.lm.snr_db(q, self.nodes_now, self.boxes)
        g = self.lm.goodput(snr)
        per = np.zeros(len(self.demands))
        ok = np.where(admitted & (g > 0))[0]
        if len(ok):
            per[ok] = self.demands[ok] * self.lm.fraction(len(ok), (self.demands[ok] / g[ok]).sum())
        return per, los

    def score(self, q):
        """Value the search maximises (lexicographic pair)."""
        if self.cfg.objective == "vtopa":
            # VTOPA objective [OJVT 2026], adapted to the calibrated link model:
            # 1) each UE_i has a coverage sphere: the free-space LoS distance within which the
            #    SNR reaches the minimum SNR of the MCS that carries its demand T_i;
            #    VTOPA searches the sphere intersection that contains the most UEs;
            # 2) inside it, VTOPA maximises the aggregate throughput sum_i min(T_i, C_i), where
            #    C_i is the rate of UE_i's link alone (each UE has the full bandwidth); here
            #    C_i is the calibrated single-station goodput g_i, with the actual LoS/NLoS;
            # 3) among equal positions, the one closest to the UEs (smallest mean distance).
            # Keys 1 and 2 are combined lexicographically into one value.
            self.n_evals += 1
            lm, nodes, T = self.lm, self.nodes_now, self.demands
            d = np.linalg.norm(nodes - q, axis=1)
            snr_los = lm.tx + 2 * lm.g - (20 * np.log10(4 * np.pi * d * lm.f / C) + lm.eta_los) - lm.noise_dbm
            in_sphere = lm.goodput(snr_los) >= T
            snr, _ = lm.snr_db(q, nodes, self.boxes)
            thr = np.minimum(T, lm.goodput(snr)).sum()
            return float(in_sphere.sum() * 1e4 + thr), -float(d.mean())
        return self.evaluate(q)[:2]

    @staticmethod
    def better(a, b, tol):
        if a[0] > b[0] + tol:
            return True
        if a[0] < b[0] - tol:
            return False
        return a[1] > b[1]


def project_reachable(q, center, radius):
    if center is None:
        return q
    v = q - center
    n = np.linalg.norm(v)
    return q if n <= radius else center + v * (radius / n)


def pattern_search(ev, q0, step0, max_evals, center=None, radius=None):
    c = ev.cfg
    q = np.array(q0, float)
    best = ev.score(q) if ev.valid(q) else (-np.inf, -np.inf)
    step, used = step0, 1
    dirs = np.vstack([np.eye(3), -np.eye(3)])
    while step >= c.step_min and used < max_evals:
        improved = False
        for d in dirs:
            cand = project_reachable(q + step * d, center, radius)
            if not ev.valid(cand):
                continue
            val = ev.score(cand)
            used += 1
            if Evaluator.better(val, best, c.tie_tol_mbps):
                q, best, improved = cand, val, True
                break
            if used >= max_evals:
                break
        if not improved:
            step /= 2.0
    return q, best


def full_search(ev, center=None, radius=None):
    c = ev.cfg
    xs = np.linspace(-ev.half, ev.half, c.grid_xy)
    scored = []
    for x in xs:
        for y in xs:
            for z in c.grid_z:
                q = np.array([x, y, float(z)])
                if center is not None and np.linalg.norm(q - center) > radius:
                    continue
                if ev.valid(q):
                    scored.append((ev.score(q), q))
    if center is not None:
        scored.append((ev.score(center), center.copy()))
    scored.sort(key=lambda s: (round(s[0][0] / c.tie_tol_mbps), s[0][1]), reverse=True)
    ev.cache = [q for _, q in scored[: c.cache_size]]
    best_q, best = None, (-np.inf, -np.inf)
    for _, q in scored[: c.refine_top]:
        q2, v2 = pattern_search(ev, q, c.step0_full, c.max_evals_full, center, radius)
        if Evaluator.better(v2, best, c.tie_tol_mbps):
            best_q, best = q2, v2
    return best_q, best


def traffic_centroid(nodes, demands):
    w = demands / demands.sum()
    return (nodes * w[:, None]).sum(axis=0)


def map_match(points, boxes, margin=1.0):
    """Move points that lie inside a building footprint to the nearest outside point."""
    P = np.array(points, float)
    for i, p in enumerate(P):
        inside = (p[0] > boxes[:, 0]) & (p[0] < boxes[:, 1]) & (p[1] > boxes[:, 2]) & \
                 (p[1] < boxes[:, 3])
        for b in boxes[inside]:
            d = [p[0] - b[0], b[1] - p[0], p[1] - b[2], b[3] - p[1]]
            j = int(np.argmin(d))
            if j == 0:
                P[i, 0] = b[0] - margin
            elif j == 1:
                P[i, 0] = b[1] + margin
            elif j == 2:
                P[i, 1] = b[2] - margin
            else:
                P[i, 1] = b[3] + margin
    return P


def centroid_position(ev, nodes, demands, q_prev, radius):
    c = ev.cfg
    q = traffic_centroid(nodes, demands)
    q[2] = c.centroid_z
    q = project_reachable(q, q_prev, radius)
    while not ev.valid(q) and q[2] < c.z_max:
        q[2] = min(c.z_max, q[2] + 5.0)
    return q if ev.valid(q) else (q_prev if q_prev is not None else q)


def run_trajectory(scenario, link_model, cfg=Config(), method="adaptive", fixed_q=None,
                   policy=None):
    """method: 'adaptive' (TAPRA), 'full' (search every slot), 'fixed' (static AP),
    'centroid' (traffic-weighted centroid at fixed altitude), 'policy' (a trained
    agent with .act(ev, q_prev, nodes, demands, radius) -> q, e.g. RLTOPA).
    cfg.objective selects what 'full'/'adaptive' maximise ('vtopa' for the VTOPA rule)."""
    area = scenario["simulationParams"]["area_size_m"]
    ev = Evaluator(link_model, scenario["Buildings"], area, cfg)
    nodes_all = [np.array(n["position"], float) for n in scenario["UENodes"]]
    if cfg.map_match:
        nodes_all = [map_match(p, ev.boxes) for p in nodes_all]
    demands = np.array([n["traffic_demand_mbps"] for n in scenario["UENodes"]], float)
    K = len(nodes_all[0])
    radius = cfg.v_max * cfg.tau
    out, q_prev, q_prev2, share_prev, last_full = [], None, None, None, 0
    for k in range(K):
        nodes = np.array([p[k] for p in nodes_all])
        prev_nodes = np.array([p[k - 1] for p in nodes_all]) if k > 0 else None
        ev.set_slot(nodes, prev_nodes, q_prev if method != "fixed" else fixed_q, demands)
        ev.n_evals = 0
        t0 = time.perf_counter()
        if method == "fixed":
            q, mode = np.asarray(fixed_q, float), "fixed"
        elif method == "centroid":
            q, mode = centroid_position(ev, nodes, demands, q_prev, None if k == 0 else radius), "centroid"
        elif method == "policy":
            q, mode = policy.act(ev, q_prev, nodes, demands, None if k == 0 else radius), "policy"
        elif k == 0 or method == "full":
            q, _ = full_search(ev, None if k == 0 else q_prev, None if k == 0 else radius)
            mode = "full"
        else:
            mom = cfg.momentum * (q_prev - q_prev2) if q_prev2 is not None else 0.0
            shift = traffic_centroid(nodes, demands) - traffic_centroid(prev_nodes, demands)
            shift[2] = 0.0
            q0 = project_reachable(q_prev + mom + cfg.traffic_weight * shift, q_prev, radius)
            q0[2] = np.clip(q0[2], cfg.z_min, cfg.z_max)
            if not ev.valid(q0):
                q0 = q_prev.copy()
            cands = [q0, q_prev.copy()] + ([c_ for c_ in ev.cache
                                             if np.linalg.norm(c_ - q_prev) <= radius]
                                            if cfg.use_cache else [])
            off = np.linspace(-radius, radius, cfg.local_grid) / np.sqrt(2)
            for dx in off:
                for dy in off:
                    for z in (cfg.z_min + 25, cfg.z_max - 20):
                        cq = np.array([q0[0] + dx, q0[1] + dy, z])
                        cands.append(project_reachable(cq, q_prev, radius) if cfg.reach_fix else cq)
            best_c, best_v = q0, (-np.inf, -np.inf)
            for cq in cands:
                if ev.valid(cq):
                    v_ = ev.score(cq)
                    if Evaluator.better(v_, best_v, cfg.tie_tol_mbps):
                        best_c, best_v = cq, v_
            q, val = pattern_search(ev, best_c, cfg.step0_adapt, cfg.max_evals_adapt, q_prev, radius)
            mode = "adaptive"
            share = val[0] / demands.sum()
            if cfg.use_trigger and ((share_prev is not None and
                                     share < share_prev - cfg.trigger_drop) or
                                    (k - last_full >= cfg.period)):
                qf, vf = full_search(ev, q_prev, radius)
                if Evaluator.better(vf, val, cfg.tie_tol_mbps):
                    q = qf
                mode, last_full = "adaptive+full", k
        secs = time.perf_counter() - t0
        n_ev = ev.n_evals
        served, margin, per_node, adm = ev.evaluate(q)
        per_static, los = ev.static(q, adm)
        out.append(SlotResult(k + 1, q.copy(), served, float(demands.sum()), margin, per_node,
                              per_static, los, adm, n_ev, secs, mode))
        q_prev2, q_prev = q_prev, q.copy()
        share_prev = served / demands.sum()
        if mode == "full":
            last_full = k
    return out


def realised_inslot(scenario, results, link_model, cfg, uav_track_fn, details=False):
    """Python check of what the plan delivers when the geometry is updated every second:
    nodes on their real 1-second tracks, UAV on its real flight path, admitted set fixed
    per slot. Returns the average served share of demand."""
    B = boxes_from_buildings(scenario["Buildings"])
    T = np.array([n["traffic_demand_mbps"] for n in scenario["UENodes"]], float)
    tracks = [np.array(n["track_1s"]) for n in scenario["UENodes"]]
    ut = np.array(uav_track_fn([r.q.tolist() for r in results], cfg.tau, cfg.v_max))
    lm = link_model
    vals, node_sat = [], []
    for r in results:
        adm = np.where(r.admitted)[0]
        for s in np.arange(0.5, cfg.tau, 1.0):
            t = (r.k - 1) * cfg.tau + s
            q = np.array([np.interp(t, ut[:, 0], ut[:, c]) for c in (1, 2, 3)])
            P = np.array([[np.interp(t, tr[:, 0], tr[:, c]) for c in (1, 2, 3)] for tr in tracks])
            g = lm.goodput(lm.snr_db(q, P, B)[0])
            ok = adm[g[adm] > 0]
            f = lm.fraction(len(ok), (T[ok] / g[ok]).sum()) if len(ok) else 0.0
            vals.append(T[ok].sum() * f / T.sum() if len(ok) else 0.0)
            sat = np.zeros(len(T))
            sat[ok] = f
            node_sat.append(sat)
    if not details:
        return float(np.mean(vals))
    S = np.array(node_sat)
    return {"realised_share": float(np.mean(vals)),
            "node_outage": float((S < 0.95).mean()),          # node-seconds below 95% of demand
            "jain": float(np.mean([(s.sum() ** 2) / (len(s) * (s ** 2).sum()) if s.sum() > 0 else 0
                                   for s in S]))}
