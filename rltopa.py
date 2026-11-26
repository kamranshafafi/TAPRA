"""
RLTOPA baseline, rebuilt for the TAPRA evaluation.

Faithful to the published RLTOPA (IEEE Access 2024):
  - deep Q-network (the original used tf-agents DQN through ns3-gym),
  - discrete moves of the UAV: +-x, +-y, +-z, stay, one grid step each,
  - reward = nLoS / N, the share of UEs in LoS (Algorithm 1 of the paper),
  - search space restricted to the demand-based coverage spheres: each UE_i has a sphere
    whose radius is the free-space LoS distance at which the SNR reaches the minimum SNR of
    the MCS that carries T_i (same spheres as the VTOPA rule). A move that would reduce the
    number of spheres containing the UAV is rejected, so the UAV moves into, and stays in,
    the intersection of the spheres of the most UEs. (The original assumes this intersection
    contains all UEs; with 20-50 spread UEs it usually does not, hence the count.)
Adapted so the comparison is fair:
  - trained here, on separate training cities (seeds 1001+), never on test cities,
  - throughput share from the same calibrated link/capacity model as TAPRA,
  - same speed limit: at most floor(v_max * tau / step) moves per slot.
Observation: UAV position (normalised), and for every node its offset from the UAV
(normalised) and its LoS flag. One agent per number of nodes.

  python rltopa.py --nodes 20 --area 500 --env urban --episodes 300   # trains, saves .pt
"""
import argparse
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from link_model import LinkModel, C
from optimizer import Config, Evaluator, project_reachable
from scenario_gen import generate

MOVES = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [0, 0, 0]],
                 float)


def observe(ev, q, nodes, area):
    _, los = ev.lm.snr_db(q, nodes, ev.boxes)
    rel = (nodes[:, :2] - q[:2]) / area
    uav = np.array([q[0] / area, q[1] / area, (q[2] - ev.cfg.z_min) / (ev.cfg.z_max - ev.cfg.z_min)])
    return np.concatenate([uav, rel.ravel(), los.astype(float)]).astype(np.float32), los


def reward(ev, q, nodes, demands):
    """Original RLTOPA reward: share of UEs in LoS."""
    _, los = ev.lm.snr_db(q, nodes, ev.boxes)
    return float(los.mean())


def sphere_count(ev, q, nodes, demands):
    """Number of demand-based coverage spheres that contain q."""
    lm = ev.lm
    d = np.maximum(np.linalg.norm(nodes - q, axis=1), 1.0)
    snr_los = lm.tx + 2 * lm.g - (20 * np.log10(4 * np.pi * d * lm.f / C) + lm.eta_los) - lm.noise_dbm
    return int((lm.goodput(snr_los) >= demands).sum())


def allowed(ev, q, cand, nodes, demands):
    """Valid airspace and no loss of coverage spheres (the original sphere constraint)."""
    return ev.valid(cand) and sphere_count(ev, cand, nodes, demands) >= sphere_count(ev, q, nodes, demands)


class QNet(nn.Module):
    def __init__(self, n_in, n_out=len(MOVES), h=128):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(n_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(),
                               nn.Linear(h, n_out))

    def forward(self, x):
        return self.f(x)


class RltopaPolicy:
    """Greedy policy used by optimizer.run_trajectory(method='policy')."""

    def __init__(self, net, area, step, start=None):
        self.net, self.area, self.step, self.start = net, area, step, start

    def act(self, ev, q_prev, nodes, demands, radius):
        c = ev.cfg
        q = np.array(self.start if q_prev is None else q_prev, float)
        if q_prev is None and self.start is None:
            q = np.array([0.0, 0.0, (c.z_min + c.z_max) / 2])
        n_moves = int(c.v_max * c.tau // self.step)
        for _ in range(n_moves):
            obs, _ = observe(ev, q, nodes, self.area)
            with torch.no_grad():
                a = int(self.net(torch.from_numpy(obs)).argmax())
            cand = q + MOVES[a] * self.step
            if allowed(ev, q, cand, nodes, demands):
                q = cand
        return q


def train(env="urban", area=500, n_nodes=20, traffic="homogeneous", episodes=300, slots=30,
          step=10.0, seed0=1001, out=None, verbose=True):
    torch.manual_seed(0)
    random.seed(0)
    lm = LinkModel(env=env)
    cfg = Config()
    n_in = 3 + 3 * n_nodes
    net, tgt = QNet(n_in), QNet(n_in)
    tgt.load_state_dict(net.state_dict())
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    buf = deque(maxlen=50000)
    gamma, batch = 0.95, 64
    n_moves = int(cfg.v_max * cfg.tau // step)
    total_steps, eps_end, eps_decay = 0, 0.05, episodes * slots * n_moves * 0.6
    for ep in range(episodes):
        sc = generate(env, area, n_nodes, traffic, slots, cfg.tau, seed=seed0 + ep % 50)
        ev = Evaluator(lm, sc["Buildings"], area, cfg)
        T = np.array([n["traffic_demand_mbps"] for n in sc["UENodes"]], float)
        P = [np.array(n["position"], float) for n in sc["UENodes"]]
        q = np.array([random.uniform(-area / 4, area / 4), random.uniform(-area / 4, area / 4), 65.0])
        while not ev.valid(q):
            q[2] += 5
        ep_r = 0.0
        for k in range(slots):
            nodes = np.array([p[k] for p in P])
            ev.set_slot(nodes, None, None, T)
            for _ in range(n_moves):
                eps = max(eps_end, 1.0 - total_steps / eps_decay)
                obs, _ = observe(ev, q, nodes, area)
                if random.random() < eps:
                    a = random.randrange(len(MOVES))
                else:
                    with torch.no_grad():
                        a = int(net(torch.from_numpy(obs)).argmax())
                cand = q + MOVES[a] * step
                q2 = cand if allowed(ev, q, cand, nodes, T) else q
                r = reward(ev, q2, nodes, T)
                obs2, _ = observe(ev, q2, nodes, area)
                buf.append((obs, a, r, obs2))
                q, ep_r, total_steps = q2, ep_r + r, total_steps + 1
                if len(buf) >= 1000:
                    s, a_, r_, s2 = zip(*random.sample(buf, batch))
                    s, s2 = torch.tensor(np.array(s)), torch.tensor(np.array(s2))
                    a_, r_ = torch.tensor(a_), torch.tensor(r_, dtype=torch.float32)
                    qv = net(s).gather(1, a_.view(-1, 1)).squeeze(1)
                    with torch.no_grad():
                        y = r_ + gamma * tgt(s2).max(1).values
                    loss = nn.functional.smooth_l1_loss(qv, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                if total_steps % 1000 == 0:
                    tgt.load_state_dict(net.state_dict())
        if verbose and (ep % 25 == 0 or ep == episodes - 1):
            print(f"episode {ep}: mean reward {ep_r / (slots * n_moves):.3f}, epsilon {eps:.2f}")
    out = out or f"rltopa_{env}_{int(area)}m_{n_nodes}n_{traffic}.pt"
    torch.save({"state": net.state_dict(), "n_in": n_in, "step": step, "area": area}, out)
    if verbose:
        print("saved", out)
    return out


def load_policy(path):
    d = torch.load(path, weights_only=False)
    net = QNet(d["n_in"])
    net.load_state_dict(d["state"])
    net.eval()
    return RltopaPolicy(net, d["area"], d["step"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="urban")
    ap.add_argument("--area", type=float, default=500)
    ap.add_argument("--nodes", type=int, default=20)
    ap.add_argument("--traffic", default="homogeneous")
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--slots", type=int, default=30)
    a = ap.parse_args()
    train(a.env, a.area, a.nodes, a.traffic, a.episodes, a.slots)
