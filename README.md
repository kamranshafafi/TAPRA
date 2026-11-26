# TAPRA: Traffic-Aware UAV Positioning and Radio Resource Allocation

Code, simulator settings, calibration data, and per-seed results for

> K. Shafafi *et al.*, "Traffic-Aware UAV Positioning and Radio Resource Allocation for Mobile Users
> in Urban Networks," *IEEE Open Journal of the Communications Society*, 2026 (OJCOMS-07829-2026).

Release `v1.0-ojcoms-r2` reproduces every table and figure of the second revision.

## Contents

| Folder / file | What it is |
|---|---|
| `scenario_gen.py` | City generator (ITU-R P.1410 statistics), UE tracks, demands |
| `link_model.py`, `los_geometry.py`, `propagation.py` | A2G channel, LoS from building geometry, 802.11ax MCS and capacity model |
| `optimizer.py` | TAPRA (slot-average objective, admission, grid + pattern search, adaptive update) and the baselines |
| `rltopa.py`, `rev2_dqn.py` | RLTOPA agent and the DQN baseline trained on the TAPRA objective |
| `batch.py`, `summarise.py`, `rev2_tables.py` | Main experiment (6 scenarios x 10 cities) and the tables |
| `rev2_*.py` | Rev. 2 studies: admission, best subset, optimality gap, channel and elevation, sampling, sensitivity, latency, paired differences |
| `robustness.py`, `energy.py`, `runtime_scaling.py`, `flight_fix.py` | Imperfect information, propulsion energy, runtime, flight-path clearance |
| `tapra_export.py`, `compare_ns3.py`, `rev2_policy_export.py` | Export of plans to ns-3 and comparison with the model |
| `ns3/tapra_validate.cc` | ns-3.38 scratch program (IEEE 802.11ax, RTS/CTS, buildings) |
| `ns3_validation_batch.sh`, `rev2_ns3.sh`, `rev2_ns3_reps.sh`, `rev2_ns3_stats.py` | ns-3 runs: in-slot snapshots, continuous 250 s runs (three random streams), admission rules; packet statistics |
| `calib/calib_ax.csv` | ns-3 calibration of the capacity model |
| `results/` | Per-seed results (CSV) behind every table and figure |

## Requirements

- Python 3.10+, `numpy`, `scipy`, `pandas`, `matplotlib`, `torch` (RLTOPA only)
- ns-3.38 for the packet-level validation (copy `ns3/tapra_validate.cc` to `ns-3-dev/scratch/`)

## Reproducing

```bash
python batch.py --workers 4 --out results/batch.csv        # main comparison, Fig. 4
python rev2_tables.py                                      # served-share and paired tables
python rev2_admission_batch.py all                         # admission rules and best subset
python rev2_optgap.py 2                                    # dense-search optimality gap
python rev2_channel.py 2                                   # independent channel, elevation bins
python rev2_sampling.py 2                                  # evaluation sampling density
python rev2_sens.py                                        # speed, slot duration, parameters
python robustness.py --workers 4                           # imperfect information
bash rev2_timing.sh                                        # runtime and latency (idle machine)
```

ns-3 (after `./ns3 build` with the scratch program):

```bash
SLOTS=50 PAR=4 bash ns3_validation_batch.sh                # export plans, in-slot snapshots
PAR=4 bash rev2_ns3.sh cont                                # continuous runs
THR=8 PAR=4 bash rev2_ns3.sh adm                           # admission rules
bash rev2_ns3.sh sum                                       # summaries
PAR=4 bash rev2_ns3_reps.sh                                # two more random streams per continuous run
```

Every method is deterministic for a given seed. A seed fixes the city layout; the homogeneous and
heterogeneous scenarios of the same environment and area share the layouts but not the UEs.
Confidence intervals are Student-t over the ten cities (9 degrees of freedom).

`Config.reach_fix = True` (default) projects every candidate of the adaptive update onto the
reachable region; `False` reproduces the first-revision results.

## Figures

`python fig_served_gain.py`, `fig_pr_rev2.py`, `fig_rob_heat.py`, and `fig_energy_dot.py` redraw Figs. 4, 5, 6, and 9 from `results/` into `figures/`; `plots.py` draws Figs. 7 and 8.

## Result files

One row per city (seed) and method, unless stated otherwise. Shares are fractions of the offered demand.

| File | Paper |
|---|---|
| `results/main_results.csv` | Served-share and paired-difference tables, Figs. 4 and 5 |
| `results/rev2_paired.csv` | Paired differences TAPRA minus every method, by scenario |
| `results/rev2_ablation.csv` | Ablations: no motion prediction, no UAV travel |
| `results/rev2_dqn.csv`, `results/dqn_agents/` | DQN baseline with the TAPRA objective, and the trained agents |
| `results/rev2_admission.csv` | Admission rules, best subset (`_gap` rows), fairness |
| `results/rev2_optgap.csv` | Dense-search reference, per slot |
| `results/rev2_sens.csv`, `rev2_sens_summary.csv` | Sensitivity table |
| `results/robustness.csv` | Imperfect information (Fig. 6) |
| `results/rev2_channel.csv`, `rev2_elevation.csv` | Independent channels; elevation bins |
| `results/rev2_sampling.csv` | Evaluation every 3 instants / 1 s / 0.25 s |
| `results/energy.csv`, `results/flight_path.csv` | Propulsion energy (Fig. 9); reachable distance and clearance |
| `results/runtime.csv`, `results/rev2_latency*.csv` | Runtime (Fig. 8); time from reports to commands, deadline misses |
| `results/ns3/` | ns-3 per-flow and per-slot outputs of every run (`ns3_runs.tar.gz`, unpack with `tar xzf`), and the summaries: `snapshot_summary.csv`, `continuous_summary_3streams.csv`, `admission_summary.csv` |
| `results/tables/` | LaTeX rows of the tables, as generated |

## License

MIT
