"""
Seeded scenario generator for TAPRA.

Buildings follow the ITU-R P.1410 statistical built-up model (Manhattan grid):
  alpha = fraction of land covered by buildings
  beta  = buildings per km^2
  gamma = scale of the Rayleigh distribution of building heights (m)
Square buildings of width W = 1000*sqrt(alpha/beta) on a grid of pitch
P = 1000/sqrt(beta), so streets are S = P - W wide. Each seed shifts the grid and
redraws all heights, so every seed is a different city with the same statistics.
Buildings are axis-aligned boxes, which is what ns-3 Building objects support.

Ground nodes move along street centre lines (Manhattan mobility): straight on,
left or right at each crossing, with a small fixed lateral offset inside the
street, so no node is ever inside a building.

Output uses the same JSON format as the existing ScenarioX.json files, plus a
"generator" block with every parameter and the seed, so any scenario can be rebuilt.
"""

import argparse
import json

import numpy as np

from los_geometry import boxes_from_buildings, los_vector

# ITU-R P.1410 built-up parameters (alpha, beta [1/km^2], gamma [m])
ENVIRONMENTS = {
    "suburban": (0.1, 750, 8),
    "urban": (0.3, 500, 15),
    "dense_urban": (0.5, 300, 20),
    "highrise_urban": (0.5, 300, 50),
}

TRAFFIC_CLASSES_MBPS = [6.5, 13.0, 19.5, 26.0]  # telemetry, situational data, video, HD video

NODE_TYPES = {  # speed range in m/s
    "UE": (1.0, 2.0),    # pedestrian responders
    "UGV": (3.0, 8.0),   # ground vehicles
}


def make_buildings(env, area, rng, min_height=3.0):
    alpha, beta, gamma = ENVIRONMENTS[env]
    W = 1000.0 * np.sqrt(alpha / beta)
    P = 1000.0 / np.sqrt(beta)
    S = P - W
    half = area / 2.0
    x0 = -half - rng.uniform(0, P)            # random grid phase
    y0 = -half - rng.uniform(0, P)
    xs = x0 + P * np.arange(int(np.ceil(area / P)) + 2)
    ys = y0 + P * np.arange(int(np.ceil(area / P)) + 2)
    buildings = []
    for xa in xs:
        for ya in ys:
            xb, yb = xa + W, ya + W
            # keep buildings fully inside the area
            if xa < -half or ya < -half or xb > half or yb > half:
                continue
            h = max(min_height, rng.rayleigh(gamma))
            buildings.append({"id": len(buildings) + 1, "height": round(float(h), 2),
                              "corners": [[xa, ya], [xb, ya], [xb, yb], [xa, yb]]})
    grid = {"W": W, "S": S, "P": P, "x0": x0, "y0": y0}
    return buildings, grid


def street_lines(grid, area):
    """Centre-line coordinates of the vertical (x = const) and horizontal streets."""
    half = area / 2.0
    P, W = grid["P"], grid["W"]
    cx = grid["x0"] + W + grid["S"] / 2 + P * np.arange(-1, int(area / P) + 3)
    cy = grid["y0"] + W + grid["S"] / 2 + P * np.arange(-1, int(area / P) + 3)
    cx = cx[(cx > -half) & (cx < half)]
    cy = cy[(cy > -half) & (cy < half)]
    return cx, cy


def manhattan_track(cx, cy, speed, tau, n_slots, lateral, z, rng, dt=1.0):
    """Positions at slot starts, plus a fine track every `dt` seconds (for ns-3).
    The node moves along street centre lines; at each crossing it goes straight
    (50%), left (25%) or right (25%); it turns back at the edge of the area."""
    i, j = rng.integers(len(cx)), rng.integers(len(cy))
    st = {"x": cx[i], "y": cy[j]}
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    st["dx"], st["dy"] = dirs[rng.integers(4)]

    def pos():
        off_x = lateral if st["dy"] != 0 else 0.0
        off_y = lateral if st["dx"] != 0 else 0.0
        return [round(float(st["x"] + off_x), 2), round(float(st["y"] + off_y), 2), z]

    def advance(dist):
        remaining = dist
        while remaining > 1e-9:
            x, y, dx, dy = st["x"], st["y"], st["dx"], st["dy"]
            if dx != 0:
                nxt = cx[cx > x + 1e-9] if dx > 0 else cx[cx < x - 1e-9]
                nxt = (nxt.min() if dx > 0 else nxt.max()) if len(nxt) else None
                gap = abs(nxt - x) if nxt is not None else 0.0
            else:
                nxt = cy[cy > y + 1e-9] if dy > 0 else cy[cy < y - 1e-9]
                nxt = (nxt.min() if dy > 0 else nxt.max()) if len(nxt) else None
                gap = abs(nxt - y) if nxt is not None else 0.0
            if nxt is None:                      # edge of the area: turn back
                st["dx"], st["dy"] = -dx, -dy
                continue
            step = min(remaining, gap)
            st["x"], st["y"] = x + dx * step, y + dy * step
            remaining -= step
            if step == gap:                      # at a crossing
                r = rng.random()
                if r >= 0.5:
                    st["dx"], st["dy"] = (-dy, dx) if r < 0.75 else (dy, -dx)

    slots, fine = [], []
    n_sub = int(round(tau / dt))
    for k in range(n_slots):
        slots.append(pos())
        for s in range(n_sub):
            fine.append([round(k * tau + s * dt, 3)] + pos())
            advance(speed * dt)
    fine.append([round(n_slots * tau, 3)] + pos())
    return slots, fine


def generate(env="urban", area=500.0, n_nodes=20, traffic="homogeneous", n_slots=50,
             tau=5.0, ugv_share=0.3, z_node=1.5, seed=1, homo_demand=6.5):
    rng = np.random.default_rng(seed)
    buildings, grid = make_buildings(env, area, rng)
    cx, cy = street_lines(grid, area)
    lat_max = max(0.0, grid["S"] / 2 - 2.0)  # stay at least 2 m from walls
    nodes = []
    for k in range(n_nodes):
        ntype = "UGV" if rng.random() < ugv_share else "UE"
        speed = rng.uniform(*NODE_TYPES[ntype])
        demand = homo_demand if traffic == "homogeneous" else float(rng.choice(TRAFFIC_CLASSES_MBPS))
        track, fine = manhattan_track(cx, cy, speed, tau, n_slots,
                                      rng.uniform(-lat_max, lat_max), z_node, rng)
        nodes.append({"id": k + 1, "type": ntype, "speed_mps": round(float(speed), 2),
                      "traffic_demand_mbps": demand, "position": track,
                      "track_1s": fine})
    alpha, beta, gamma = ENVIRONMENTS[env]
    return {
        "scenario_name": f"{env}_{int(area)}m_{n_nodes}n_{traffic}_s{seed}",
        "wifiConfig": {"wifiStandard": "WIFI_STANDARD_80211ac", "wifiChannelNumber": 50},
        "systemParams": {"bandwidth_total_mhz": 160, "tx_power_dbm": 20,
                         "tx_antenna_gain_dbi": 0, "rx_antenna_gain_dbi": 0,
                         "noise_density_dbm_hz": -174, "altitude_min_m": 30,
                         "altitude_max_m": 100},
        "simulationParams": {"area_size_m": area, "num_ues": n_nodes,
                             "num_buildings": len(buildings), "k_time_slots": n_slots,
                             "time_slot_duration_s": tau, "traffic_type": traffic},
        "generator": {"seed": seed, "environment": env, "itu_p1410": {
            "alpha": alpha, "beta_per_km2": beta, "gamma_m": gamma},
            "building_width_m": round(grid["W"], 2), "street_width_m": round(grid["S"], 2),
            "grid_pitch_m": round(grid["P"], 2), "ugv_share": ugv_share,
            "node_speeds_mps": NODE_TYPES, "traffic_classes_mbps": TRAFFIC_CLASSES_MBPS},
        "UENodes": nodes,
        "Buildings": buildings,
    }


def summary(sc, altitudes=(30, 50, 75, 100)):
    B = sc["Buildings"]
    boxes = boxes_from_buildings(B)
    area = sc["simulationParams"]["area_size_m"]
    built = sum((b[1] - b[0]) * (b[3] - b[2]) for b in boxes) / area ** 2
    H = boxes[:, 4]
    pos = np.array([p for n in sc["UENodes"] for p in n["position"]])
    # no node inside a building footprint
    inside = ((pos[:, None, 0] > boxes[None, :, 0]) & (pos[:, None, 0] < boxes[None, :, 1]) &
              (pos[:, None, 1] > boxes[None, :, 2]) & (pos[:, None, 1] < boxes[None, :, 3])).any()
    print(f"{sc['scenario_name']}: {len(B)} buildings, built-up {built:.1%}, "
          f"height mean {H.mean():.1f} m, max {H.max():.1f} m, "
          f"street {sc['generator']['street_width_m']} m, nodes inside buildings: {bool(inside)}")
    k1 = np.array([n["position"][0] for n in sc["UENodes"]])
    for z in altitudes:
        los = los_vector([0.0, 0.0, z], k1, boxes).mean()
        print(f"   UAV at area centre, z = {z:3d} m: {los:.0%} of nodes in LoS at k=1")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="urban", choices=list(ENVIRONMENTS))
    ap.add_argument("--area", type=float, default=500)
    ap.add_argument("--nodes", type=int, default=20)
    ap.add_argument("--traffic", default="homogeneous", choices=["homogeneous", "heterogeneous"])
    ap.add_argument("--slots", type=int, default=50)
    ap.add_argument("--tau", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    sc = generate(a.env, a.area, a.nodes, a.traffic, a.slots, a.tau, seed=a.seed)
    summary(sc)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(sc, f)
        print("written", a.out)
