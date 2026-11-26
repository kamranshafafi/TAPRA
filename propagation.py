"""
Propagation models for TAPRA, written line by line from the ns-3 sources
(src/propagation/model/itu-r-1411-los-propagation-loss-model.cc and
 itu-r-1411-nlos-over-rooftop-propagation-loss-model.cc, ns-3-dev, Nov 2023 copy).

Every formula, constant and branch below follows the ns-3 code, including its
choices that differ from the ITU-R P.1411 text (see NOTES at the end), so that
the optimiser and the ns-3 validation use exactly the same loss values.

Argument order matters for the NLoS model: ns-3 uses a->GetPosition().z in one
term (kd), and the framework calls GetLoss(ap, ue). So `a` is the UAV here.
"""

import math
from dataclasses import dataclass

import numpy as np

from los_geometry import is_los

C_NS3 = 299792458.0  # speed of light used by both ns-3 models


@dataclass
class NlosParams:
    """Attribute values set in fw-phd/rl/utilgym.cc, lossModelNLoS()."""
    rooftop_height: float = 20.0      # RooftopLevel
    streets_orientation: float = 45.0  # StreetsOrientation [deg]
    streets_width: float = 200.0       # StreetsWidth [m]
    buildings_extend: float = 10.0     # BuildingsExtend [m]
    building_separation: float = 100.0  # BuildingSeparation [m]
    environment: str = "SubUrban"      # Urban | SubUrban | OpenAreas
    city_size: str = "Small"           # Small | Medium | Large


def _dist(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b))


def itu1411_los_loss(a, b, freq_hz):
    """ns-3 ItuR1411LosPropagationLossModel::GetLoss. Returns loss in dB (positive).

    ns-3 returns the MEAN of the P.1411 lower and upper bounds, not the lower bound.
    """
    lam = C_NS3 / freq_hz
    za, zb = float(a[2]), float(b[2])
    if not (za > 0 and zb > 0):
        raise ValueError("nodes' height must be greater than 0")
    dist = _dist(a, b)
    lbp = abs(20 * math.log10((lam * lam) / (8 * math.pi * za * zb)))
    rbp = (4 * za * zb) / lam
    if dist <= rbp:
        loss_low = lbp + 20 * math.log10(dist / rbp)
        loss_up = lbp + 20 + 25 * math.log10(dist / rbp)
    else:
        loss_low = lbp + 40 * math.log10(dist / rbp)
        loss_up = lbp + 20 + 40 * math.log10(dist / rbp)
    return (loss_up + loss_low) / 2


def itu1411_nlos_rooftop_loss(a, b, freq_hz, p: NlosParams = NlosParams()):
    """ns-3 ItuR1411NlosOverRooftopPropagationLossModel::GetLoss. Loss in dB."""
    lam = C_NS3 / freq_hz
    fmhz = freq_hz / 1e6
    phi = p.streets_orientation
    if not (0 <= phi <= 90):
        raise ValueError("Street Orientation must be in [0,90]")
    if phi < 35:
        lori = -10.0 + 0.354 * phi
    elif phi < 55:
        lori = 2.5 + 0.075 * (phi - 35)
    else:
        lori = 2.5 + 0.075 * (phi - 55)  # as in ns-3 (P.1411 text: 4.0 - 0.114*(phi-55))

    distance = _dist(a, b)
    za, zb = float(a[2]), float(b[2])
    hb = za if za > zb else zb
    hm = za if za < zb else zb
    if not (hm > 0 and hb > 0):
        raise ValueError("nodes' height must be greater then 0")
    dhb = hb - p.rooftop_height
    if dhb == 0:
        raise ValueError("ns-3 divides by (hb - rooftop); hb equal to rooftop is undefined")
    ds = (lam * distance * distance) / (dhb * dhb)

    if ds < p.buildings_extend:
        if hb > p.rooftop_height:
            lbsh = -18 * math.log10(1 + dhb)
            ka = 71.4 if fmhz > 2000 else 54.0
            kd = 18.0
        else:
            lbsh = 0.0
            kd = 18.0 - 15 * dhb / za  # ns-3 uses a's height here
            ka = 54.0 - 1.6 * dhb * distance / 1000 if distance < 500 else 54.0 - 0.8 * dhb
        if fmhz > 2000:
            kf = -8.0
        elif p.environment == "Urban" and p.city_size == "Large":
            kf = -4 + 0.7 * (fmhz / 925.0 - 1)
        else:
            kf = -4 + 1.5 * (fmhz / 925.0 - 1)
        lmsd = (lbsh + ka + kd * math.log10(distance / 1000.0) + kf * math.log10(fmhz)
                - 9.0 * math.log10(p.building_separation))
    else:
        theta = math.atan(dhb / p.building_separation)
        rho = math.sqrt(dhb * dhb + p.building_separation * p.building_separation)
        if (p.rooftop_height - 1.0) < hb < (p.rooftop_height + 1.0):
            qm = p.building_separation / distance
        elif hb > p.rooftop_height:
            qm = 2.35 * (dhb / distance * math.sqrt(p.building_separation / lam)) ** 0.9
        else:
            qm = (p.building_separation / (2 * math.pi * distance) * math.sqrt(lam / rho)
                  * (1 / theta - (1 / (2 * math.pi + theta))))
        lmsd = -10 * math.log10(qm * qm)

    lbf = 32.4 + 20 * math.log10(distance / 1000) + 20 * math.log10(fmhz)
    dhm = p.rooftop_height - hm
    lrts = (-8.2 - 10 * math.log10(p.streets_width) + 10 * math.log10(fmhz)
            + 20 * math.log10(dhm) + lori)
    return lbf + lrts + lmsd if (lrts + lmsd) > 0 else lbf


def friis_loss(a, b, freq_hz):
    """Free-space loss in dB, with the same speed of light."""
    lam = C_NS3 / freq_hz
    return 20 * math.log10(4 * math.pi * _dist(a, b) / lam)


def link_loss(uav, node, buildings, freq_hz, p: NlosParams = NlosParams()):
    """Loss for one UAV-node link: geometric LoS test, then the matching ns-3 model.

    Returns (loss_db, los_flag). The UAV is passed first, as in the framework.
    """
    los = is_los(uav, node, buildings)
    if los:
        return itu1411_los_loss(uav, node, freq_hz), True
    return itu1411_nlos_rooftop_loss(uav, node, freq_hz, p), False


# ------------------------------------------------------------------ NOTES
# 1. ns-3 LoS returns the mean of the P.1411 lower and upper bounds. Below the
#    breakpoint this is about free space + 4 + 2.5*log10(d/Rbp) dB, so it falls
#    a few dB BELOW free space at short distance. "P.1411 loss always exceeds
#    Friis" is therefore false for the ns-3 model too.
# 2. ns-3 orientation term for phi >= 55 deg differs from the P.1411 text; it is
#    unused at the 45 deg setting.
# 3. The NLoS attribute values above (street width 200 m, separation 100 m,
#    extend 10 m) come from the framework; the paper must state them.

if __name__ == "__main__":
    f = 5250e6
    print(" z_uav  d_horiz |  Friis   ns3-LoS  LoS-Friis |  ns3-NLoS")
    for z in (30, 60, 100):
        for dh in (10, 50, 100, 200, 400, 700):
            a, b = (0, 0, z), (dh, 0, 1.5)
            fs, lo, nl = friis_loss(a, b, f), itu1411_los_loss(a, b, f), itu1411_nlos_rooftop_loss(a, b, f)
            print(f"{z:6.0f} {dh:8.0f} | {fs:7.1f} {lo:8.1f} {lo - fs:+9.1f} | {nl:8.1f}")
