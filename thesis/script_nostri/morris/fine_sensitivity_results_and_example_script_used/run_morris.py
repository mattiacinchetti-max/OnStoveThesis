"""
run_morris.py
====================
Master script for Morris sensitivity analysis of the LPG
supply-chain model.

Architecture
------------
For each Morris sample row the script:
  1. Reads parameters_base.py  (a clean backup – never touched)
  2. Injects the current parameter values via regex string-replacement
  3. Overwrites parameters.py  (the file supply.py imports)
  4. Launches  subprocess.run(["python", "supply.py"])
  5. Reads 5 metrics from temp_output.txt
  6. Appends one row to morris_results.csv
  7. Cleans temp_output.txt

Resumability
------------
On every start the script reads morris_results.csv (if it exists) and
skips every sample index already present.  Interrupt with Ctrl-C at any
time; restart and it continues from where it stopped.

No Parallelism note
All runs are strictly sequential so that a single parameters.py can be
overwritten safely.  

"""

from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── SALib import ───────────────────────────────────────────────────────────────
try:
    from SALib.sample import morris as morris_sample
    from SALib.analyze import morris as morris_analyze
except ImportError:
    sys.exit(
        "SALib is not installed.  Run:  pip install SALib\n"
        "Then restart this script."
    )
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# ══════════════════════════════════════════════════════════════════════════════
# 1.  PROBLEM DEFINITION  –  edit this section to match your parameters
# ══════════════════════════════════════════════════════════════════════════════
#
# Rules for 'bounds':
#   • Continuous parameters  → [min, max]
#   • The centre of the range should be the value in parameters_base.py : simmetry give us information about how results change if the correct value is higher or lower than the base value.
#
# Rules for 'param_type':
#   'float'  – written as   param_name = <value>
#   'int'    – written as   param_name = <int(value)>
#   The regex injector handles both; see _inject_param() below.
#
# ──────────────────────────────────────────────────────────────────────────────

# ----- Default values (parameters_base.py) -----
DEFAULTS = {
    
    

    #disharged after morris with N=10, we keep only parameters inside top 7 of each obj value (excluding now the cost component), we infact use only as obj :
    # "pw_mean", "pw_p10", "pw_p90", "price_min", "price_max",
    #"city_abuja", "city_benincity", "city_kano", "city_lagos", "port_harcourt", 
    # + we adjust the lagos problem + reduced to only social the discount rate range
    
    #KEEP PARAMETERS
    "cost_source_nigeria": 0.924986,
    "factor_import": 1.0,
    "user_threshold_minutes": 30,
    "discount_rate": 0.03,
    "tanker_capacity_kg": 24000,
    "vom_usd_per_kg_fill": 0.035, # Gurkan et al. (2026)
    "vom_usd_per_kg_res": 0.01,
    "car_variable_cost_per_km": 0.136, #source dont remember
    "storage_overnight_cost_usd_kg": 12.5,
    "tanker_variable_cost_per_km": 0.635,
    "INCOME_GINI_VALUE": 0.339,
    "RURAL_SHARE": 0.027905, 
    "filling_allocation_fraction": 0.333,
    "loss_factor_fill": 0.005, # 0.5% Based on Hydrocarbon Processing (2010)

    #REMOVED
    #"STS_cost": 0.1, # US$/kg
    #"driver_annual_salary_usd": 2604,
    #"hours_per_day": 8, # working hours per day
    #"tanker_overnight_cost_usd": 197673, 
    #"tanker_annual_km": 42000,
    #"tanker_fixed_loading_unloading_hours": 1.0, #generate difference also for others
    #"truck_capacity_kg": 9996,
    #"truck_overnight_cost_usd": 170200,
    #"truck_variable_cost_per_km": 0.54,
    #"truck_annual_km": 16000,
    #"van_capacity_kg": 3240,
    #"van_overnight_cost_usd": 57819,
    #"van_variable_cost_per_km": 0.437,
    #"van_annual_km": 16000, 
    #"license_cost_usd": 418,
    #"meals_per_day": 1.0,
    #"storage_license_cost_usd": 3952.0,
    #"WALK_THRESHOLD_MIN": 30, 
    #"CAR_THRESHOLD_MIN": 45,  
    #"INCOME_GDP_PC_VALUE": 2139,
    #"VEHICLE_TARGET_TOTAL": 11600000,
    #"VEHICLE_F_URBAN": 6.3,   # Users per vehicle (Urban)
    #"VEHICLE_F_RURAL": 7.3,   # Users per vehicle (Rural)
    #"fom_annual_usd_fill": 348400, # Gurkan et al. (2026)
    #"op_license_fee_usd_fill": 3952, # Gurkan et al. (2026)
    #"overnight_filling_usd": 7_000_000, # Gurkan et al. (2026) 
    #"rent_per_month_usd_res": 30.0,
    #"license_annual_usd_res": 100.0,
    #"retail_allocation_fraction": 0.5, 
    #"minimum_wage_usd_per_month": 73.0, # from onstove 
    #"work_hours_per_month": 240.0,  # 30 days * 8 hours


    #NEVER CONSIDERED for this anlysis
    #"wage_range": (0.2, 0.5), #onstrove choice
    #"wage_fraction_fallback": 0.3
    #"border_point_time_hours": 39,
    #"ferry_cost_per_km": 0.78,  # US$/ton/straight line km
    #"ferry_time_per_km": 0.071,  # hours/straight line km  
    # included like other wait"fixed_time_at_retailer_hours": 0.25,
    #?"cylinder_weight_kg": 12.5, #onstove choice
    #"reference_filling_capacity_kg": 35_000_000, # Gurkan et al. (2026)
    #"storage_cap": 0.25 #US$/kg, maximum price for storage cost, used to cap outliers in the data when USE_DIFFERENTIATED_STORAGE_COSTS is True
    #"energy_per_meal_mj": 3.64, in meals per day
    #"efficiency": 0.5, in meals per day
    #"energy_content_mj_per_kg": 45.5,
    #"density_kg_per_l": 0.5
    #"van_utilization_factor": 0.85, in capacity
    #"truck_utilization_factor": 0.85, in capacity
    #"tanker_utilization_factor": 0.85, in capacity
    #"licence_life_years": 5,
    #"tanker_life_years": 35,
    #"truck_life_years": 15,
    #"van_life_years": 10,
    #"storage_license_life_years": 3,
    #"storage_life_years": 50,
    #"storage_installation_fee": 760.0,
    #"op_license_validity_years_fill": 3, # Gurkan et al. (2026)
    #"lifetime_filling": 15, # Gurkan et al. (2026) 
    #"salary_multiplier": 1.18,
    #"days_per_year": 330,
    
}

# Parameters where ±40% would produce physically invalid values
MANUAL_BOUNDS = {
    "cost_source_nigeria": [0.24, 1.61],  #litt propano 10y trading economics
    "factor_import": [0.26, 1.74],  #litt propano 10y trading economics
    "user_threshold_minutes": [1e-6, 60],
    "discount_rate": [0.02, 0.05], #only social prospective
    "loss_factor_fill": [1e-6, 0.05],
    "filling_allocation_fraction": [1e-6, 1.0],
    "vom_usd_per_kg_res": [1e-6, 0.035],
    
    #REMOVED
    #"hours_per_day": [6, 10],
    #"WALK_THRESHOLD_MIN": [5, 500],
    #"CAR_THRESHOLD_MIN": [5, 500],
    #"retail_allocation_fraction": [1e-6, 1.0],
    #"work_hours_per_month": [160, 320],
    #"rent_per_month_usd_res": [1e-6, 100.0],
    #"license_annual_usd_res": [1e-6, 3000.0],
}
def make_bounds(name: str, default: float, pct: float = 0.40):
    if name in MANUAL_BOUNDS:
        return MANUAL_BOUNDS[name]
    return [default * (1 - pct), default * (1 + pct)]

PARAM_NAMES = list(DEFAULTS.keys())
PROBLEM_BOUNDS = [make_bounds(name, DEFAULTS[name]) for name in PARAM_NAMES]

PROBLEM: Dict = {
    "num_vars": len(PARAM_NAMES),
    "names": PARAM_NAMES,
    "bounds": PROBLEM_BOUNDS,
    "groups": None,
}

# ══════════════════════════════════════════════════════════════════════════════
# 2.  MORRIS SAMPLING OPTIONS
# ══════════════════════════════════════════════════════════════════════════════

MORRIS_N          = 20      # trajectories  
MORRIS_NUM_LEVELS = 4       # grid levels   (4 is standard)
OPTIMAL_TRAJECTORIES = None # None = use standard sampling; set to MORRIS_N//2
                            # for optimised (slower to generate, more info)
SEED              = 77      # reproducibility

# ══════════════════════════════════════════════════════════════════════════════
# 3.  FILE PATHS  –  adjust if your layout differs
# ══════════════════════════════════════════════════════════════════════════════

BASE_PARAMS_FILE  = Path("parameters_base.py")   # clean backup – never modified
PARAMS_FILE       = Path("parameters.py")         # overwritten each run
MODEL_SCRIPT      = Path("supply.py")             # the converted notebook
TEMP_OUTPUT_FILE  = Path("temp_output.txt")       # 5 metrics, CSV line
RESULTS_CSV       = Path("morris_results.csv")    # accumulated results
INDICES_CSV       = Path("morris_indices.csv")    # ranked Morris mu*/sigma per metric
PLOTS_DIR         = Path("morris_plots")          # folder for generated charts
TOP_N_PARAMS      = 14                            # parameters shown in per-metric bar charts

# Metric names (must match the order written in supply.py's output block)
METRIC_NAMES = [
    "pw_mean", "pw_p10", "pw_p90", "price_min", "price_max",
    "city_abuja", "city_benincity", "city_kano", "city_lagos", "port_harcourt",
]

PYTHON_EXE = sys.executable   # uses the same interpreter running this script

# Progress print every N completed runs
PROGRESS_EVERY = 5

# ══════════════════════════════════════════════════════════════════════════════
# 4.  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _check_prerequisites() -> None:
    """Fail fast if required files are missing."""
    missing = []
    for p in [BASE_PARAMS_FILE, MODEL_SCRIPT]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        sys.exit(
            f"\nERROR – Required file(s) not found:\n  "
            + "\n  ".join(missing)
            + "\n\nMake sure you have:\n"
            "  • parameters_base.py  (a clean copy of parameters.py)\n"
            "  • supply.py           (converted notebook, with the Morris output block)\n"
            "in the same directory as run_morris.py.\n"
        )


def _build_injector_pattern(name: str) -> re.Pattern:
    """
    Returns a compiled regex that matches lines like:
        param_name = <number>   # optional comment
    Captures the comment so we can preserve it.
    """
    return re.compile(
        r"^(?P<indent>[ \t]*)"        # leading whitespace
        r"(?P<name>" + re.escape(name) + r")"
        r"(?P<sep>[ \t]*=[ \t]*)"     # = with optional spaces
        r"(?P<value>[^\n#]+?)"         # current value (lazy, stops at # or EOL)
        r"(?P<comment>[ \t]*#[^\n]*)?" # optional inline comment
        r"$",
        re.MULTILINE,
    )


# Pre-compile all patterns once
_PATTERNS: Dict[str, re.Pattern] = {
    name: _build_injector_pattern(name) for name in PROBLEM["names"]
}


def _inject_param(text: str, name: str, value: float) -> str:
    """Replace the assignment of *name* in *text* with *value*."""
    pat = _PATTERNS[name]

    # Format value: integers stay integers to keep the file readable
    if isinstance(value, float) and value == int(value) and abs(value) < 1e9:
        # Check original type hint from bounds
        idx = PROBLEM["names"].index(name)
        lo, hi = PROBLEM["bounds"][idx]
        if isinstance(lo, int) and isinstance(hi, int):
            formatted = str(int(value))
        else:
            formatted = f"{value:.8g}"
    else:
        formatted = f"{value:.8g}"

    def replacer(m: re.Match) -> str:
        comment = m.group("comment") or ""
        return (
            m.group("indent")
            + m.group("name")
            + m.group("sep")
            + formatted
            + comment
        )

    new_text, n_subs = pat.subn(replacer, text)
    if n_subs == 0:
        raise ValueError(
            f"Parameter '{name}' not found in parameters_base.py. "
            "Check the spelling or add it to the file."
        )
    if n_subs > 1:
        print(
            f"  [WARNING] '{name}' matched {n_subs} times in parameters_base.py. "
            "Only the first occurrence will be updated (expected behaviour for "
            "simple assignments; check if the variable appears multiple times)."
        )
    return new_text


def _write_params(sample_row: np.ndarray) -> None:
    """
    Read parameters_base.py, inject all parameter values from *sample_row*,
    overwrite parameters.py.
    """
    text = BASE_PARAMS_FILE.read_text(encoding="utf-8")
    for name, value in zip(PROBLEM["names"], sample_row):
        text = _inject_param(text, name, float(value))
    PARAMS_FILE.write_text(text, encoding="utf-8")


def _run_model(run_index: int) -> Optional[List[float]]:
    """
    Execute supply.py in a fresh subprocess.
    Returns a list of 10 floats on success, None on failure.
    """
    if TEMP_OUTPUT_FILE.exists():
        TEMP_OUTPUT_FILE.unlink()

    result = subprocess.run(
        [PYTHON_EXE, str(MODEL_SCRIPT)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"\n  [ERROR] Run #{run_index} – model exited with code "
              f"{result.returncode}")
        # Print last 30 lines of stderr for diagnostics
        stderr_tail = result.stderr.strip().splitlines()[-30:]
        print("  --- stderr (last 30 lines) ---")
        print("\n".join("  " + l for l in stderr_tail))
        print("  --- end stderr ---")
        return None

    if not TEMP_OUTPUT_FILE.exists():
        print(f"\n  [ERROR] Run #{run_index} – temp_output.txt not created.")
        return None

    raw = TEMP_OUTPUT_FILE.read_text(encoding="utf-8").strip()
    if raw.upper().startswith("ERROR"):
        print(f"\n  [ERROR] Run #{run_index} – model wrote error sentinel: {raw}")
        return None

    parts = raw.split(",")
    if len(parts) != len(METRIC_NAMES):
        print(f"\n  [ERROR] Run #{run_index} – expected {len(METRIC_NAMES)} "
              f"values, got {len(parts)}: {raw!r}")
        return None

    try:
        return [float(p) for p in parts]
    except ValueError as exc:
        print(f"\n  [ERROR] Run #{run_index} – cannot parse metrics: {exc}")
        return None


def _load_done_indices() -> set:
    """Return the set of run indices already recorded in morris_results.csv."""
    if not RESULTS_CSV.exists():
        return set()
    done = set()
    with open(RESULTS_CSV, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                done.add(int(row["run_index"]))
            except (KeyError, ValueError):
                pass
    return done


def _append_result(run_index: int, sample_row: np.ndarray,
                   metrics: List[float]) -> None:
    """Append one row to morris_results.csv (creates with header if new)."""
    write_header = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(
                ["run_index"]
                + list(PROBLEM["names"])
                + METRIC_NAMES
            )
        writer.writerow(
            [run_index]
            + [f"{v:.8g}" for v in sample_row]
            + [f"{m:.6f}" for m in metrics]
        )


def _eta_string(elapsed_s: float, done: int, total: int) -> str:
    """Return a human-readable ETA string."""
    if done == 0:
        return "unknown"
    remaining = total - done
    avg_s = elapsed_s / done
    eta_s = avg_s * remaining
    finish = datetime.now() + timedelta(seconds=eta_s)
    return (
        f"{timedelta(seconds=int(eta_s))} remaining  "
        f"(~{finish.strftime('%Y-%m-%d %H:%M')})"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ANALYSIS PHASE
# ══════════════════════════════════════════════════════════════════════════════

def _run_analysis(param_values: np.ndarray) -> None:
    """Load results CSV and run morris.analyze for each output metric."""
    print("\n" + "═" * 70)
    print("MORRIS ANALYSIS")
    print("═" * 70)

    if not RESULTS_CSV.exists():
        print("morris_results.csv not found – nothing to analyse.")
        return

    # Load results
    rows: List[List[float]] = []
    with open(RESULTS_CSV, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        index_to_row: Dict[int, List[float]] = {}
        for row in reader:
            try:
                idx = int(row["run_index"])
                metrics = [float(row[m]) for m in METRIC_NAMES]
                index_to_row[idx] = metrics
            except (KeyError, ValueError):
                pass

    n_total = len(param_values)
    missing = [i for i in range(n_total) if i not in index_to_row]
    if missing:
        print(f"  [WARNING] {len(missing)} run(s) missing from CSV "
              f"(indices: {missing[:10]}{'…' if len(missing) > 10 else ''}).")
        print("  Analysis will use NaN for missing entries.")

    # Build Y arrays
    Y_all = np.full((n_total, len(METRIC_NAMES)), np.nan)
    for idx, metrics in index_to_row.items():
        if idx < n_total:
            Y_all[idx] = metrics
    all_rows = []
    per_metric_results = {}

    for m_idx, metric_name in enumerate(METRIC_NAMES):
        Y = Y_all[:, m_idx]
        n_nan = np.isnan(Y).sum()
        if n_nan > 0:
            print(f"\n  [WARNING] Metric '{metric_name}': {n_nan} NaN values. "
                  "Results may be unreliable.")
            # Replace NaN with mean for analysis (crude but allows continuation)
            Y = np.where(np.isnan(Y), np.nanmean(Y), Y)

        print(f"\n── Morris indices for: {metric_name} ──")
        try:
            Si = morris_analyze.analyze(
                PROBLEM,
                param_values,
                Y,
                conf_level=0.95,
                print_to_console=True,
                num_resamples=1000,
                seed=SEED,
            )
            names        = PROBLEM["names"]
            mu           = Si["mu"]
            mu_star      = Si["mu_star"]
            sigma        = Si["sigma"]
            mu_star_conf = Si["mu_star_conf"]
            
            per_metric_results[metric_name] = {
                "names": names, "mu": mu, "mu_star": mu_star,
                "sigma": sigma, "mu_star_conf": mu_star_conf,
            }

            ranked = sorted(zip(names, mu, mu_star, sigma, mu_star_conf),
                             key=lambda x: x[2], reverse=True)
            print(f"\n  Ranked by µ* (most influential first):")
            print(f"  {'Parameter':<35} {'µ*':>10} {'σ':>10}")
            print(f"  {'-'*55}")
            for n_, mu_, ms, sg, conf in ranked:
                print(f"  {n_:<35} {ms:>10.4f} {sg:>10.4f}")
            for rank, (n_, mu_, ms, sg, conf) in enumerate(ranked, start=1):
                all_rows.append({"metric": metric_name, "parameter": n_,
                                  "rank": rank, "mu": mu_, "mu_star": ms,
                                  "sigma": sg, "mu_star_conf": conf})
        except Exception as exc:
            print(f"  [ERROR] Could not analyse {metric_name}: {exc}")
    if all_rows:
        with open(INDICES_CSV, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["metric", "parameter", "rank",
                                                     "mu", "mu_star", "sigma", "mu_star_conf"])
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n  Morris indices saved to: {INDICES_CSV.resolve()}")

    if per_metric_results:
        _generate_plots(per_metric_results)
    print("\n" + "═" * 70)
    print(f"Full results saved to: {RESULTS_CSV.resolve()}")
    print("═" * 70)

def _generate_plots(per_metric_results):
    """Bar chart + scatter per metric, plus one combined heatmap."""
    PLOTS_DIR.mkdir(exist_ok=True)
    all_names = list(next(iter(per_metric_results.values()))["names"])
    metric_list = list(per_metric_results.keys())
    heatmap = np.zeros((len(all_names), len(metric_list)))

    for col, metric_name in enumerate(metric_list):
        data = per_metric_results[metric_name]
        names, mu_star, sigma, conf = (data["names"], np.asarray(data["mu_star"]),
                                        np.asarray(data["sigma"]), np.asarray(data["mu_star_conf"]))
        order = np.argsort(mu_star)[::-1]
        top = order[:TOP_N_PARAMS]

        fig, ax = plt.subplots(figsize=(8, 0.4 * len(top) + 1.5))
        y_pos = np.arange(len(top))
        ax.barh(y_pos, mu_star[top], xerr=conf[top], color="steelblue", ecolor="black", capsize=3)
        ax.set_yticks(y_pos); ax.set_yticklabels([names[i] for i in top]); ax.invert_yaxis()
        ax.set_xlabel("mu* (overall influence)"); ax.set_title(f"Morris ranking - {metric_name}")
        fig.tight_layout(); fig.savefig(PLOTS_DIR / f"bar_{metric_name}.png", dpi=150); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(mu_star, sigma, color="steelblue")
        for i in top:
            ax.annotate(names[i], (mu_star[i], sigma[i]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("mu* (overall influence)"); ax.set_ylabel("sigma (nonlinearity/interactions)")
        ax.set_title(f"mu* vs sigma - {metric_name}")
        fig.tight_layout(); fig.savefig(PLOTS_DIR / f"scatter_{metric_name}.png", dpi=150); plt.close(fig)

        max_mu_star = mu_star.max() if mu_star.max() > 0 else 1.0
        for name_idx, name in enumerate(all_names):
            heatmap[name_idx, col] = mu_star[names.index(name)] / max_mu_star

    fig, ax = plt.subplots(figsize=(0.5 * len(metric_list) + 3, 0.3 * len(all_names) + 2))
    im = ax.imshow(heatmap, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(metric_list))); ax.set_xticklabels(metric_list, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(all_names))); ax.set_yticklabels(all_names, fontsize=7)
    ax.set_title("Normalised mu* across all metrics")
    fig.colorbar(im, ax=ax, label="mu* / max(mu*) per metric")
    fig.tight_layout(); fig.savefig(PLOTS_DIR / "heatmap_all_metrics.png", dpi=150); plt.close(fig)

    print(f"\n  Plots saved to: {PLOTS_DIR.resolve()}")
# ══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    _check_prerequisites()

    # ── Generate Morris sample ─────────────────────────────────────────────
    print("Generating Morris sample …")
    param_values: np.ndarray = morris_sample.sample(
        PROBLEM,
        N=MORRIS_N,
        num_levels=MORRIS_NUM_LEVELS,
        optimal_trajectories=OPTIMAL_TRAJECTORIES,
        seed=SEED,
    )
    n_total = len(param_values)
    print(f"  {n_total} model evaluations required "
          f"({MORRIS_N} trajectories × {PROBLEM['num_vars'] + 1} levels).")

    # ── Resumability: find already-done indices ────────────────────────────
    done_set = _load_done_indices()
    n_done_at_start = len(done_set)
    if n_done_at_start:
        print(f"  Resuming: {n_done_at_start}/{n_total} already completed.")
    else:
        print(f"  Starting fresh.")

    # ── Run loop ───────────────────────────────────────────────────────────
    start_time    = time.time()
    completed_now = 0          # completed in THIS session
    failed_indices: List[int] = []

    for run_idx in range(n_total):
        if run_idx in done_set:
            continue   # already recorded – skip

        sample_row = param_values[run_idx]

        # Banner
        print(f"\n{'─'*60}")
        print(f"  Run {run_idx + 1}/{n_total}  |  {datetime.now().strftime('%H:%M:%S')}")
        print(f"  Parameters:")
        for name, val in zip(PROBLEM["names"], sample_row):
            print(f"    {name:<35} = {val:.6g}")

        # Inject parameters and overwrite parameters.py
        try:
            _write_params(sample_row)
        except ValueError as exc:
            print(f"  [FATAL] Cannot inject parameter: {exc}")
            print("  Skipping this run.  Fix the parameter name and restart.")
            failed_indices.append(run_idx)
            continue

        # Run the model
        t0 = time.time()
        metrics = _run_model(run_idx)
        elapsed_run = time.time() - t0

        if metrics is None:
            print(f"  [SKIPPED] Run #{run_idx} failed after {elapsed_run:.0f}s.  "
                  "It will be retried on the next run of this script.")
            failed_indices.append(run_idx)
            # Do NOT append to CSV so it will be retried
            continue

        # Record result
        _append_result(run_idx, sample_row, metrics)
        done_set.add(run_idx)
        completed_now += 1

        # Clean up temp file
        if TEMP_OUTPUT_FILE.exists():
            TEMP_OUTPUT_FILE.unlink()

        # Progress & ETA
        elapsed_total = time.time() - start_time
        n_done_total  = len(done_set)
        print(f"  ✓  {elapsed_run:.1f}s  |  metrics: "
              + ", ".join(f"{m:.4f}" for m in metrics))

        if completed_now % PROGRESS_EVERY == 0 or run_idx == n_total - 1:
            pct  = 100 * n_done_total / n_total
            eta  = _eta_string(elapsed_total, completed_now, n_total - n_done_at_start)
            print(f"\n  ── Progress: {n_done_total}/{n_total}  ({pct:.1f}%)  "
                  f"ETA: {eta}")

    # ── Post-loop summary ──────────────────────────────────────────────────
    elapsed_total = time.time() - start_time
    print(f"\n{'═'*60}")
    print(f"Loop finished in {timedelta(seconds=int(elapsed_total))}.")
    print(f"  Completed this session : {completed_now}")
    print(f"  Failed / skipped       : {len(failed_indices)}")
    if failed_indices:
        print(f"  Failed indices         : {failed_indices}")
        print("  Re-run this script to retry failed runs.")

    all_done = (len(done_set) == n_total)

    # Restore the original parameters.py from the clean backup
    shutil.copy(str(BASE_PARAMS_FILE), str(PARAMS_FILE))
    print(f"\n  parameters.py restored from parameters_base.py.")

    # ── Analysis ───────────────────────────────────────────────────────────
    if all_done:
        _run_analysis(param_values)
    else:
        remaining = n_total - len(done_set)
        print(f"\n  {remaining} run(s) still pending.  "
              "Re-run this script to complete them.  "
              "Analysis will execute automatically after all runs finish.")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    main()

