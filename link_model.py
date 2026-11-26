"""
Link and capacity model for TAPRA, calibrated on ns-3.38 for the chosen network:
802.11ax, UAV as access point, contention access with RTS/CTS, Yans PHY with
the Nist error model (calibration: wifi_calib_ax.cc, calib/calib_ax.csv).

Loss: free space + mean extra loss for LoS or NLoS (Al-Hourani et al., Globecom 2014,
Table II, urban, 5.8 GHz: 1.2 dB LoS, 23 dB NLoS), with geometric LoS from the
building map. SNR uses the ns-3 noise floor (thermal + 7 dB noise figure).
Goodput per link: measured single-station saturated goodput versus SNR (part A).
Sharing: airtime load L = sum_i T_i / g(SNR_i). Fitted on part C (RTS/CTS on,
6.5 Mbit/s per station, 1 to 50 stations, calib_ax.csv + calib_ax_rts50.csv;
RMSE 3.1 points, max error 10.1 points):
    delivered fraction f(N, L) = min(1 - KAPPA * max(0, N - N0), (ETA0 - BETA * N) / L)
applied equally to all admitted stations.
Admission: stations are admitted best link first; the planner keeps the number of
admitted stations that maximises the delivered demand. Others stay silent.
Margin: (ETA0 - BETA * N) / L of the admitted set (> 1 means spare airtime).
"""
import math
import numpy as np
import pandas as pd

from los_geometry import boxes_from_buildings, los_vector

C = 299792458.0
ATG_EXCESS_DB = {  # Al-Hourani et al. 2014, Table II, 5800 MHz: (LoS, NLoS)
    "suburban": (0.2, 24.0), "urban": (1.2, 23.0),
    "dense_urban": (1.8, 26.0), "highrise_urban": (2.5, 41.0)}


class LinkModel:
    ETA0, BETA, KAPPA, N0 = 0.99, 0.003, 0.0036, 15   # fitted, part C, RTS/CTS on

    def __init__(self, calib_csv="calib/calib_ax.csv", env="urban", freq_hz=5250e6,
                 bw_hz=160e6, tx_dbm=20.0, gains_db=0.0, noise_figure_db=7.0):
        self.f, self.tx, self.g = freq_hz, tx_dbm, gains_db
        self.eta_los, self.eta_nlos = ATG_EXCESS_DB[env]
        self.noise_dbm = -174 + 10 * math.log10(bw_hz) + noise_figure_db
        df = pd.read_csv(calib_csv).dropna()
        a = df[(df.part == "A") & (df.associated >= 1)].groupby("rss_dbm").goodput_mbps.mean()
        a = a[a > 0]
        self.snr_grid = (a.index.values - self.noise_dbm).astype(float)
        self.gp_grid = a.values.astype(float)            # Mbit/s, one step per MCS

    def loss_db(self, uav, nodes, boxes):
        nodes = np.atleast_2d(nodes)
        d = np.linalg.norm(nodes - np.asarray(uav, float), axis=1)
        fspl = 20 * np.log10(4 * np.pi * d * self.f / C)
        los = los_vector(uav, nodes, boxes)
        return fspl + np.where(los, self.eta_los, self.eta_nlos), los

    def snr_db(self, uav, nodes, boxes):
        L, los = self.loss_db(uav, nodes, boxes)
        return self.tx + 2 * self.g - L - self.noise_dbm, los

    def goodput(self, snr_db):
        # step lookup: the measured value at the highest grid SNR not above snr
        idx = np.searchsorted(self.snr_grid, snr_db, side="right") - 1
        return np.where(idx < 0, 0.0, self.gp_grid[np.clip(idx, 0, len(self.gp_grid) - 1)])

    def fraction(self, n, load):
        """Delivered share of demand for n admitted stations with airtime load `load`."""
        if n == 0:
            return 1.0
        return float(min(1.0 - self.KAPPA * max(0, n - self.N0),
                         self.usable(n) / max(load, 1e-12)))

    def usable(self, n):
        """Usable share of airtime with n admitted stations."""
        return self.ETA0 - self.BETA * n

    def allocate(self, g, demands):
        """Admission and delivered rate per node. g: link goodput (0 = no link).
        Returns (served per node, margin)."""
        demands = np.asarray(demands, float)
        served = np.zeros(len(demands))
        ok = np.where(g > 0)[0]
        if len(ok) == 0:
            return served, 0.0
        cost = demands[ok] / g[ok]                   # airtime needed by each node
        order = ok[np.argsort(cost / demands[ok])]   # best link (least airtime per Mbit/s) first
        best_val, best_m, load = -1.0, 0, 0.0
        loads = np.cumsum(demands[order] / g[order])
        dems = np.cumsum(demands[order])
        for m in range(1, len(order) + 1):
            val = dems[m - 1] * self.fraction(m, loads[m - 1])
            if val > best_val + 1e-9:
                best_val, best_m = val, m
        adm = order[:best_m]
        f = self.fraction(best_m, loads[best_m - 1])
        served[adm] = demands[adm] * f
        return served, self.usable(best_m) / loads[best_m - 1]

    def margin(self, uav, nodes, demands_mbps, boxes):
        snr, los = self.snr_db(uav, nodes, boxes)
        g = self.goodput(snr)
        served, m = self.allocate(g, demands_mbps)
        return m, served, los
