"""
Exact 3D line-of-sight test for TAPRA.

A building is a vertical prism: a convex footprint polygon from z = 0 up to z = height.
A link is blocked when the straight segment between the two antennas passes through
the inside of any prism. Unlike the old ray_intersects_building in gradient.py, this
uses the height of the ray where it crosses the footprint, so a ray that passes above
a roof counts as LoS.

Method: Cyrus-Beck clipping. The segment p(t) = p1 + t (p2 - p1), t in [0, 1], is
clipped against every side of the footprint and against the planes z = 0 and
z = height. If an interval of positive length survives, the segment enters the prism.
Touching a wall or grazing the roof edge (zero-length overlap) counts as LoS.
"""

import numpy as np

EPS = 1e-9


def _is_convex(corners):
    c = np.asarray(corners, dtype=float)
    n = len(c)
    signs = set()
    for i in range(n):
        a, b, d = c[i], c[(i + 1) % n], c[(i + 2) % n]
        cross = (b[0] - a[0]) * (d[1] - b[1]) - (b[1] - a[1]) * (d[0] - b[0])
        if abs(cross) > EPS:
            signs.add(cross > 0)
    return len(signs) <= 1


def segment_blocked_by_prism(p1, p2, corners, height):
    """True if the open segment p1-p2 passes through the inside of the prism."""
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    d = p2 - p1
    c = np.asarray(corners, dtype=float)
    if not _is_convex(c):
        raise ValueError("Footprint must be convex; split non-convex buildings first.")

    t_in, t_out = 0.0, 1.0

    def clip(num, den):
        # Keep the part of the segment where num + den * t < 0 (strictly inside).
        nonlocal t_in, t_out
        if abs(den) < EPS:
            return num < -EPS  # parallel: inside only if strictly inside
        t = -num / den
        if den < 0:
            t_in = max(t_in, t)
        else:
            t_out = min(t_out, t)
        return t_in < t_out - EPS

    # Vertical extent: 0 < z < height  ->  -z < 0  and  z - height < 0
    if not clip(-p1[2], -d[2]):
        return False
    if not clip(p1[2] - height, d[2]):
        return False

    # Footprint sides, with outward normals regardless of corner order.
    area2 = sum(c[i, 0] * c[(i + 1) % len(c), 1] - c[(i + 1) % len(c), 0] * c[i, 1]
                for i in range(len(c)))
    ccw = area2 > 0
    for i in range(len(c)):
        a, b = c[i], c[(i + 1) % len(c)]
        edge = b - a
        normal = np.array([edge[1], -edge[0]]) if ccw else np.array([-edge[1], edge[0]])
        num = normal @ (p1[:2] - a)
        den = normal @ d[:2]
        if not clip(num, den):
            return False
    return True


def is_los(uav_pos, node_pos, buildings):
    """True if no building blocks the UAV-node segment."""
    for b in buildings:
        if segment_blocked_by_prism(uav_pos, node_pos, b["corners"], b["height"]):
            return False
    return True


# ---------------------------------------------------------------- tests
def _tests():
    bldg = {"corners": [[-105.8, 135.6], [-74.6, 135.6], [-74.6, 184.1], [-105.8, 184.1]],
            "height": 16.6}
    B = [bldg]
    # 1. Ray passes over the roof at 40-68 m: LoS (the old code said NLoS).
    assert is_los([-90, 100, 89], [-90, 250, 1.5], B)
    # 2. Low UAV, ray crosses the footprint below the roof: NLoS.
    assert not is_los([-90, 100, 10], [-90, 250, 1.5], B)
    # 3. Ray misses the footprint in 2D: LoS.
    assert is_los([0, 100, 10], [0, 250, 1.5], B)
    # 4. Ray clips the roof corner region from above: blocked only if below roof there.
    #    UAV right above the far wall, low enough that the ray dips under 16.6 m inside.
    assert not is_los([-90, 190, 20], [-90, 120, 1.5], B)
    # 5. Ray grazes along a wall (on the boundary line, not inside): LoS.
    assert is_los([-74.6, 100, 10], [-74.6, 250, 1.5], B)
    # 6. Corner order does not matter (clockwise footprint).
    cw = {"corners": bldg["corners"][::-1], "height": 16.6}
    assert is_los([-90, 100, 89], [-90, 250, 1.5], [cw])
    assert not is_los([-90, 100, 10], [-90, 250, 1.5], [cw])
    # 7. Endpoint order does not matter.
    assert not is_los([-90, 250, 1.5], [-90, 100, 10], B)
    # 8. Node standing next to a tall building, UAV straight above the node: LoS.
    assert is_los([-70, 160, 60], [-70, 160, 1.5], B)
    print("All LoS geometry tests passed.")


if __name__ == "__main__":
    _tests()


# ------------------------------------------------- fast version for many buildings
def boxes_from_buildings(buildings):
    """Axis-aligned boxes as an (M, 5) array: xmin, xmax, ymin, ymax, height.
    ns-3 Building objects are axis-aligned boxes, so this matches the simulator."""
    out = []
    for b in buildings:
        c = np.asarray(b["corners"], dtype=float)
        out.append([c[:, 0].min(), c[:, 0].max(), c[:, 1].min(), c[:, 1].max(), b["height"]])
    return np.asarray(out, dtype=float)


def los_vector(uav, nodes, boxes):
    """LoS flag for every node (N,) from one UAV position, vectorised over all
    buildings (slab method). Same rule as segment_blocked_by_prism: a link is
    blocked only if the segment has an overlap of positive length with a box
    interior (z from 0 to height)."""
    uav = np.asarray(uav, dtype=float)
    nodes = np.atleast_2d(np.asarray(nodes, dtype=float))
    d = nodes - uav                                      # (N, 3)
    lo = np.stack([boxes[:, 0], boxes[:, 2], np.zeros(len(boxes))], 1)  # (M, 3)
    hi = np.stack([boxes[:, 1], boxes[:, 3], boxes[:, 4]], 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d[:, None, :]                        # (N, 1, 3)
        t1 = (lo[None] - uav) * inv
        t2 = (hi[None] - uav) * inv
    tmin = np.minimum(t1, t2)
    tmax = np.maximum(t1, t2)
    # parallel axes: inside the open slab -> no constraint; outside or on it -> empty
    par = np.abs(d)[:, None, :] < EPS
    inside = (uav > lo[None] + EPS) & (uav < hi[None] - EPS)
    tmin = np.where(par, np.where(inside, -np.inf, np.inf), tmin)
    tmax = np.where(par, np.where(inside, np.inf, -np.inf), tmax)
    t_in = np.maximum(tmin.max(axis=2), 0.0)
    t_out = np.minimum(tmax.min(axis=2), 1.0)
    blocked = (t_out - t_in) > EPS                       # (N, M)
    return ~blocked.any(axis=1)


def _test_fast_matches_exact(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    bldgs = []
    for _ in range(30):
        x, y = rng.uniform(-200, 200, 2)
        w, h = rng.uniform(10, 40, 2)
        bldgs.append({"corners": [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                      "height": rng.uniform(3, 60)})
    boxes = boxes_from_buildings(bldgs)
    bad = 0
    for _ in range(n // 100):
        uav = np.r_[rng.uniform(-250, 250, 2), rng.uniform(25, 100)]
        nodes = np.c_[rng.uniform(-250, 250, (100, 2)), np.full(100, 1.5)]
        fast = los_vector(uav, nodes, boxes)
        exact = np.array([is_los(uav, p, bldgs) for p in nodes])
        bad += int((fast != exact).sum())
    print(f"Fast vs exact LoS on {n} random links: {bad} disagreements")
    assert bad == 0


if __name__ == "__main__":
    _test_fast_matches_exact()
