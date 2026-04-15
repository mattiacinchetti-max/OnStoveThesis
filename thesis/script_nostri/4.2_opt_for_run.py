"""
Huff-based preferred LPG distributor assignment – OPTIMIZED v2
===============================================================
Versione corretta con funzioni helper definite prima del codice principale.
"""

from __future__ import annotations

import json
import time
from typing import Iterable
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, connected_components

import rasterio
from rasterio.warp import reproject, Resampling

# =============================================================================
# USER PARAMETERS
# =============================================================================
DATA_DIR = "dataset_big"

# Inputs
RESELLER_GPKG = f"{DATA_DIR}/full_lpg_chain_nig_3857.gpkg"
RESELLER_LAYER = "resell_and_filling"
POPULATION_RASTER = f"{DATA_DIR}/Population.tif"
CAR_SHARE_RASTER = f"{DATA_DIR}/vehicles_allocation_share.tif"
WALK_FRICTION_RASTER = f"{DATA_DIR}/friction_walk.tif"
MOTO_FRICTION_RASTER = f"{DATA_DIR}/friction_moto.tif"

# Outputs
OUTPUT_PIXEL_RASTER = f"{DATA_DIR}/huff_preferred_distributor_per_pixel_optimized_v2.tif"
OUTPUT_LOOKUP_CSV = f"{DATA_DIR}/huff_reseller_lookup_optimized_v2.csv"

# Columns
RESELLER_ID_COLUMN = "id_res&fil"
ATTRACTIVENESS_COLUMN = "attractiveness"

# Huff parameters
BETA = 2.0
EPS = 1e-6
MIN_ATTRACTIVENESS = 1e-6

# Population rules
MIN_POP_PER_PIXEL = 0.0

# Car‑share interpretation
CAR_SHARE_IS_PERCENT = True
MIN_CAR_SHARE_FOR_CAR_MODE = 0.05
MIN_EXPECTED_CAR_USERS_PER_PIXEL = 1.0

# Candidate strategy
MIN_CANDIDATES = 4
PRIMARY_SEARCH_RADIUS_KM = 60
MAX_SEARCH_RADIUS_KM = 100
EXTRA_CANDIDATES_CAR = 6

# Dijkstra limit factors
INITIAL_LIMIT_FACTOR_WALK = 8.0
INITIAL_LIMIT_FACTOR_CAR = 10.0
FINAL_LIMIT_FACTOR_WALK = 14.0
FINAL_LIMIT_FACTOR_CAR = 16.0
LIMIT_MARGIN_MIN = 30.0
UNASSIGNED_TIME_MIN = 7000.0

# Graph assumptions
CELL_SIZE_METERS = 1000.0
USE_8_NEIGHBORS = False

# Progress & profiling
PROGRESS_EVERY = 2000
MAX_PIXELS_DEBUG = None
ENABLE_PROFILING = True
PROFILE_LOG_JSON = f"{DATA_DIR}/huff_run_profile_optimized_v2.json"

# Chunking parameters
CHUNK_SIZE = 4000
OUTLIER_PERCENTILE = 95

# Nodata conventions
NODATA_FLOAT = -9999.0
NODATA_INT = -1

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def _read_raster(path: str):
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32, copy=False)
        profile = src.profile.copy()
        nodata = src.nodata
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr).astype(np.float32)
    return arr, profile

def _align_to_reference(path: str, ref_profile: dict, resampling: Resampling) -> np.ndarray:
    with rasterio.open(path) as src:
        dst = np.full((ref_profile["height"], ref_profile["width"]), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return dst

def _safe_attractiveness(values: Iterable) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").astype(np.float64).to_numpy()
    arr = np.where(np.isfinite(arr), arr, MIN_ATTRACTIVENESS)
    arr = np.maximum(arr, MIN_ATTRACTIVENESS)
    return arr

def _to_fraction(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float32, copy=True)
    if CAR_SHARE_IS_PERCENT:
        out = out / 100.0
    out = np.where(np.isfinite(out), out, 0.0)
    out = np.clip(out, 0.0, 1.0)
    return out

def _eta(seconds: float) -> str:
    if (not np.isfinite(seconds)) or seconds < 0:
        return "n/a"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"

def _write_multiband_pixel_raster(path: str, ref_profile: dict, bands: list[np.ndarray], names: list[str]):
    if len(bands) != len(names):
        raise ValueError("Bands and names must have the same length.")
    profile = ref_profile.copy()
    profile.update(dtype="float32", count=len(bands), nodata=NODATA_FLOAT, compress="lzw")
    with rasterio.open(path, "w", **profile) as dst:
        for i, (arr, name) in enumerate(zip(bands, names), start=1):
            out = np.where(np.isfinite(arr), arr, NODATA_FLOAT).astype(np.float32)
            dst.write(out, i)
            dst.set_band_description(i, name)

def _read_reseller_ids(gdf: gpd.GeoDataFrame) -> np.ndarray:
    if RESELLER_ID_COLUMN not in gdf.columns:
        raise KeyError(f"Missing column '{RESELLER_ID_COLUMN}' in reseller layer.")
    rid = pd.to_numeric(gdf[RESELLER_ID_COLUMN], errors="coerce")
    if rid.isna().any():
        raise ValueError(f"Column '{RESELLER_ID_COLUMN}' contains non-numeric values.")
    if not rid.is_unique:
        raise ValueError(f"Column '{RESELLER_ID_COLUMN}' must be unique.")
    if (rid <= 0).any():
        raise ValueError(f"Column '{RESELLER_ID_COLUMN}' must contain positive IDs.")
    return rid.astype(np.int64).to_numpy()

def _candidate_idx_adaptive(r: int, c: int, tree: cKDTree, n_points: int) -> np.ndarray:
    idx = np.array([], dtype=np.int32)
    found_primary = tree.query_ball_point([r, c], r=PRIMARY_SEARCH_RADIUS_KM, p=2)
    if len(found_primary) > 0:
        idx = np.asarray(found_primary, dtype=np.int32)
    if idx.size < MIN_CANDIDATES:
        found_max = tree.query_ball_point([r, c], r=MAX_SEARCH_RADIUS_KM, p=2)
        if len(found_max) > 0:
            idx = np.unique(np.concatenate([idx, np.asarray(found_max, dtype=np.int32)]))
    if idx.size < MIN_CANDIDATES:
        k = min(max(MIN_CANDIDATES, EXTRA_CANDIDATES_CAR), n_points)
        _, nn = tree.query([r, c], k=k)
        idx = np.unique(np.concatenate([idx, np.atleast_1d(nn).astype(np.int32)]))
    return idx

def _winner_from_dist(dist_row: np.ndarray, cand_idx: np.ndarray,
                      reseller_node: np.ndarray, reseller_id: np.ndarray,
                      reseller_attr: np.ndarray):
    dist_row = np.asarray(dist_row).reshape(-1)
    cand_nodes = reseller_node[cand_idx]
    if cand_nodes.size == 0 or dist_row.size == 0 or int(np.max(cand_nodes)) >= dist_row.size:
        return NODATA_INT, np.nan
    t = dist_row[cand_nodes]
    a = reseller_attr[cand_idx]
    valid_t = np.isfinite(t) & (t >= 0)
    scores = np.full(t.shape, -np.inf, dtype=np.float64)
    scores[valid_t] = a[valid_t] / np.power(t[valid_t] + EPS, BETA)
    if np.all(~np.isfinite(scores)):
        return NODATA_INT, np.nan
    j_local = int(np.nanargmax(scores))
    rid = int(reseller_id[cand_idx[j_local]])
    tmin = float(t[j_local]) if np.isfinite(t[j_local]) else np.nan
    return rid, tmin

def _run_mode_with_fallback(
    src_node: int,
    r: int,
    c: int,
    graph: csr_matrix,
    friction_min: float,
    base_idx: np.ndarray,
    r_rows: np.ndarray,
    r_cols: np.ndarray,
    reseller_node: np.ndarray,
    reseller_id: np.ndarray,
    reseller_attr: np.ndarray,
    src_component: int,
    reseller_component: np.ndarray,
    component_index_map: dict[int, np.ndarray],
    initial_limit_factor: float,
    final_limit_factor: float,
):
    cand = base_idx
    if cand.size > 0:
        cand = cand[reseller_component[cand] == src_component]
    if cand.size == 0:
        return NODATA_INT, UNASSIGNED_TIME_MIN, False, "no_candidate_in_component"

    lb = np.hypot(r_rows[cand] - r, r_cols[cand] - c) * CELL_SIZE_METERS * max(friction_min, EPS)
    limit = float(np.nanmax(lb) * initial_limit_factor + LIMIT_MARGIN_MIN) if lb.size > 0 else np.inf
    dist_row = dijkstra(csgraph=graph, directed=True, indices=src_node,
                        unweighted=False, limit=limit)
    dist_row = np.asarray(dist_row).reshape(-1)
    rid, tmin = _winner_from_dist(dist_row, cand, reseller_node, reseller_id, reseller_attr)
    if rid >= 0 and np.isfinite(tmin):
        return rid, float(tmin), False, "base_ok"

    cand_global = component_index_map.get(src_component, np.empty(0, dtype=np.int32))
    if cand_global.size == 0:
        return NODATA_INT, UNASSIGNED_TIME_MIN, True, "no_global_in_component"

    lb2 = np.hypot(r_rows[cand_global] - r, r_cols[cand_global] - c) * CELL_SIZE_METERS * max(friction_min, EPS)
    limit2 = float(np.nanmax(lb2) * final_limit_factor + LIMIT_MARGIN_MIN) if lb2.size > 0 else np.inf
    dist_row2 = dijkstra(csgraph=graph, directed=True, indices=src_node,
                         unweighted=False, limit=limit2)
    dist_row2 = np.asarray(dist_row2).reshape(-1)
    rid2, tmin2 = _winner_from_dist(dist_row2, cand_global, reseller_node, reseller_id, reseller_attr)
    if rid2 >= 0 and np.isfinite(tmin2):
        return rid2, float(tmin2), True, "fallback_ok"
    return NODATA_INT, UNASSIGNED_TIME_MIN, True, "fallback_unassigned"

def _build_component_index_map(component_labels: np.ndarray) -> dict[int, np.ndarray]:
    order = np.argsort(component_labels, kind="mergesort")
    labels_sorted = component_labels[order]
    uniq, start_idx = np.unique(labels_sorted, return_index=True)
    end_idx = np.r_[start_idx[1:], len(order)]
    return {int(lbl): order[s:e].astype(np.int32, copy=False)
            for lbl, s, e in zip(uniq, start_idx, end_idx)}

# =============================================================================
# MAIN CODE
# =============================================================================
t0 = time.time()
print("[1/8] Loading population reference raster...")
pop, ref_profile = _read_raster(POPULATION_RASTER)
transform = ref_profile["transform"]
crs = ref_profile["crs"]
height, width = pop.shape
print(f"Grid: {width} x {height}, CRS={crs}")

print("[2/8] Aligning car share and frictions...")
car_share_raw = _align_to_reference(CAR_SHARE_RASTER, ref_profile, Resampling.nearest)
walk_friction = _align_to_reference(WALK_FRICTION_RASTER, ref_profile, Resampling.bilinear)
moto_friction = _align_to_reference(MOTO_FRICTION_RASTER, ref_profile, Resampling.bilinear)

car_share = _to_fraction(car_share_raw)
walk_share = (1.0 - car_share).astype(np.float32)

walk_friction = np.where(walk_friction > 0, walk_friction, np.nan).astype(np.float32)
moto_friction = np.where(moto_friction > 0, moto_friction, np.nan).astype(np.float32)

walk_friction_min = float(np.nanpercentile(walk_friction[np.isfinite(walk_friction)], 5))
moto_friction_min = float(np.nanpercentile(moto_friction[np.isfinite(moto_friction)], 5))

print(f"Walk friction range (min/m): {np.nanmin(walk_friction):.6f} .. {np.nanmax(walk_friction):.6f}")
print(f"Moto friction range (min/m): {np.nanmin(moto_friction):.6f} .. {np.nanmax(moto_friction):.6f}")

# =============================================================================
# LOAD RESELLERS
# =============================================================================
print("[3/8] Loading reseller points...")
resellers = gpd.read_file(RESELLER_GPKG, layer=RESELLER_LAYER)
if resellers.empty:
    raise RuntimeError("Reseller layer is empty.")
if resellers.crs != crs:
    resellers = resellers.to_crs(crs)

if ATTRACTIVENESS_COLUMN not in resellers.columns:
    raise KeyError(f"Missing column '{ATTRACTIVENESS_COLUMN}' in reseller layer.")

resellers = resellers[resellers.geometry.notna()].copy()
resellers = resellers[resellers.geometry.geom_type.isin(["Point"])].copy()
resellers[ATTRACTIVENESS_COLUMN] = _safe_attractiveness(resellers[ATTRACTIVENESS_COLUMN])

r_rows, r_cols = rasterio.transform.rowcol(transform,
                                           resellers.geometry.x.values,
                                           resellers.geometry.y.values)
r_rows = np.asarray(r_rows, dtype=np.int32)
r_cols = np.asarray(r_cols, dtype=np.int32)

inside = (r_rows >= 0) & (r_rows < height) & (r_cols >= 0) & (r_cols < width)
resellers = resellers.loc[inside].copy()
r_rows = r_rows[inside]
r_cols = r_cols[inside]

reseller_id = _read_reseller_ids(resellers).astype(np.int32)
reseller_attr = resellers[ATTRACTIVENESS_COLUMN].astype(np.float64).to_numpy()
coords_rc = np.column_stack([r_rows, r_cols]).astype(np.float64)
reseller_tree = cKDTree(coords_rc)
print(f"Resellers on grid: {len(resellers)}")

# =============================================================================
# BUILD GRAPHS
# =============================================================================
print("[4/8] Building graph topology...")
valid = np.isfinite(walk_friction) & np.isfinite(moto_friction)
node_id = -np.ones((height, width), dtype=np.int32)
vr, vc = np.where(valid)
n_nodes = len(vr)
node_id[vr, vc] = np.arange(n_nodes, dtype=np.int32)
print(f"Valid graph nodes: {n_nodes:,}")

neighbors = [(-1,0),(1,0),(0,-1),(0,1)]
if USE_8_NEIGHBORS:
    neighbors += [(-1,-1),(-1,1),(1,-1),(1,1)]

edge_i, edge_j, edge_cost_walk, edge_cost_moto = [], [], [], []
diag_factor = np.sqrt(2.0)

for r, c in zip(vr, vc):
    n0 = node_id[r, c]
    fw0, fm0 = walk_friction[r, c], moto_friction[r, c]
    for dr, dc in neighbors:
        rr, cc = r + dr, c + dc
        if rr < 0 or rr >= height or cc < 0 or cc >= width:
            continue
        n1 = node_id[rr, cc]
        if n1 < 0:
            continue
        fw1, fm1 = walk_friction[rr, cc], moto_friction[rr, cc]
        if not (np.isfinite(fw1) and np.isfinite(fm1)):
            continue
        step_m = CELL_SIZE_METERS
        if dr != 0 and dc != 0:
            step_m *= diag_factor
        cw = 0.5 * (fw0 + fw1) * step_m
        cm = 0.5 * (fm0 + fm1) * step_m
        if cw <= 0 or cm <= 0:
            continue
        edge_i.append(n0)
        edge_j.append(n1)
        edge_cost_walk.append(float(cw))
        edge_cost_moto.append(float(cm))

graph_walk = csr_matrix((edge_cost_walk, (edge_i, edge_j)), shape=(n_nodes, n_nodes))
graph_moto = csr_matrix((edge_cost_moto, (edge_i, edge_j)), shape=(n_nodes, n_nodes))
print(f"Directed edges: {len(edge_i):,}")

n_comp_walk, comp_walk = connected_components(csgraph=graph_walk, directed=False, return_labels=True)
n_comp_moto, comp_moto = connected_components(csgraph=graph_moto, directed=False, return_labels=True)
print(f"Connected components (walk/moto): {n_comp_walk:,} / {n_comp_moto:,}")

# Map resellers to graph
reseller_node = node_id[r_rows, r_cols]
valid_res = reseller_node >= 0
reseller_node = reseller_node[valid_res]
reseller_id = reseller_id[valid_res]
reseller_attr = reseller_attr[valid_res]
r_rows = r_rows[valid_res]
r_cols = r_cols[valid_res]
coords_rc = np.column_stack([r_rows, r_cols]).astype(np.float64)
reseller_tree = cKDTree(coords_rc)
reseller_comp_walk = comp_walk[reseller_node]
reseller_comp_moto = comp_moto[reseller_node]
reseller_comp_index_walk = _build_component_index_map(reseller_comp_walk)
reseller_comp_index_moto = _build_component_index_map(reseller_comp_moto)
n_resellers = len(reseller_node)
k_extra_car = min(EXTRA_CANDIDATES_CAR, n_resellers)

# =============================================================================
# PREPARE INHABITED PIXELS
# =============================================================================
print("[5/8] Preparing inhabited pixel list...")
inhabited = np.isfinite(pop) & (pop > MIN_POP_PER_PIXEL) & valid
pix_rows, pix_cols = np.where(inhabited)
n_pix = len(pix_rows)
if MAX_PIXELS_DEBUG is not None:
    n_use = min(MAX_PIXELS_DEBUG, n_pix)
    pix_rows = pix_rows[:n_use]
    pix_cols = pix_cols[:n_use]
    n_pix = n_use
    print(f"DEBUG mode active: {n_pix} pixels")
print(f"Inhabited pixels to process: {n_pix:,}")

order = np.lexsort((pix_cols, pix_rows))
pix_rows = pix_rows[order]
pix_cols = pix_cols[order]
pix_nodes = node_id[pix_rows, pix_cols]
pix_comp_walk = comp_walk[pix_nodes]
pix_comp_moto = comp_moto[pix_nodes]

best_id_walk = np.full((height, width), NODATA_INT, dtype=np.int32)
best_time_walk = np.full((height, width), np.nan, dtype=np.float32)
best_id_car = np.full((height, width), NODATA_INT, dtype=np.int32)
best_time_car = np.full((height, width), np.nan, dtype=np.float32)

profile = {
    "enabled": ENABLE_PROFILING,
    "start_unix": float(time.time()),
    "config": {
        "chunk_size": CHUNK_SIZE,
        "outlier_percentile": OUTLIER_PERCENTILE,
        "init_limit_walk": INITIAL_LIMIT_FACTOR_WALK,
        "init_limit_car": INITIAL_LIMIT_FACTOR_CAR,
    },
    "counts": {
        "pixels_total": n_pix,
        "walk_fallback_used": 0,
        "car_fallback_used": 0,
        "car_copied_from_walk": 0,
        "outliers_processed": 0,
    },
    "timings_sec": {
        "total": 0.0,
        "precompute": 0.0,
        "walk_mode": 0.0,
        "car_mode": 0.0,
        "outliers": 0.0,
    },
}

# =============================================================================
# PRE-COMPUTE CANDIDATES AND LIMITS
# =============================================================================
print("[6/8] Pre‑computing candidates and limits...")
t_pre = time.time()

limits_walk = np.zeros(n_pix, dtype=np.float32)
cand_walk_list = [None] * n_pix
need_car_mask = np.zeros(n_pix, dtype=bool)

for i in range(n_pix):
    r, c = pix_rows[i], pix_cols[i]
    cand = _candidate_idx_adaptive(r, c, reseller_tree, n_resellers)
    cand_walk_list[i] = cand
    if cand.size > 0:
        lb = np.hypot(r_rows[cand] - r, r_cols[cand] - c) * CELL_SIZE_METERS * max(walk_friction_min, EPS)
        limits_walk[i] = float(np.nanmax(lb) * INITIAL_LIMIT_FACTOR_WALK + LIMIT_MARGIN_MIN) if lb.size > 0 else np.inf
    else:
        limits_walk[i] = 0.0

    cs = float(car_share[r, c])
    expected = cs * float(pop[r, c])
    if cs >= MIN_CAR_SHARE_FOR_CAR_MODE and expected >= MIN_EXPECTED_CAR_USERS_PER_PIXEL:
        need_car_mask[i] = True

    if (i+1) % PROGRESS_EVERY == 0:
        print(f"  Pre‑computed {i+1:,} / {n_pix:,} pixels")

profile["timings_sec"]["precompute"] = time.time() - t_pre
print(f"  Pre‑compute done in {_eta(profile['timings_sec']['precompute'])}")

# =============================================================================
# PROCESS CHUNKS
# =============================================================================
print("[7/8] Processing chunks...")
loop_t0 = time.time()
chunk_start = 0
processed = 0
outlier_indices = []

while chunk_start < n_pix:
    chunk_end = min(chunk_start + CHUNK_SIZE, n_pix)
    idx_slice = slice(chunk_start, chunk_end)
    rows = pix_rows[idx_slice]
    cols = pix_cols[idx_slice]
    nodes = pix_nodes[idx_slice]
    comp_w = pix_comp_walk[idx_slice]
    comp_m = pix_comp_moto[idx_slice]
    lims_w = limits_walk[idx_slice]
    cands_w = cand_walk_list[idx_slice]
    need_car_chunk = need_car_mask[idx_slice]

    if len(lims_w) > 0:
        threshold = np.percentile(lims_w[lims_w > 0], OUTLIER_PERCENTILE) if np.any(lims_w > 0) else 0.0
        is_outlier = (lims_w > threshold) & (lims_w > 0)
    else:
        is_outlier = np.zeros(len(lims_w), dtype=bool)

    outlier_positions = np.where(is_outlier)[0]
    if len(outlier_positions) > 0:
        keep_mask = ~is_outlier
        rows_clean = rows[keep_mask]
        cols_clean = cols[keep_mask]
        nodes_clean = nodes[keep_mask]
        comp_w_clean = comp_w[keep_mask]
        comp_m_clean = comp_m[keep_mask]
        lims_w_clean = lims_w[keep_mask]
        cands_w_clean = [cands_w[i] for i in range(len(cands_w)) if keep_mask[i]]
        need_car_clean = need_car_chunk[keep_mask]
        for off in outlier_positions:
            outlier_indices.append(chunk_start + off)
    else:
        rows_clean, cols_clean = rows, cols
        nodes_clean, comp_w_clean, comp_m_clean = nodes, comp_w, comp_m
        lims_w_clean, cands_w_clean = lims_w, cands_w
        need_car_clean = need_car_chunk
    n_clean = len(rows_clean)

    # Walk mode
    if n_clean > 0:
        global_limit_walk = float(np.max(lims_w_clean)) if np.any(lims_w_clean > 0) else np.inf
        tw_start = time.time()
        dist_walk = dijkstra(csgraph=graph_walk, directed=True,
                             indices=nodes_clean, limit=global_limit_walk)
        profile["timings_sec"]["walk_mode"] += time.time() - tw_start

        for i_local in range(n_clean):
            r, c = rows_clean[i_local], cols_clean[i_local]
            dist_row = dist_walk[i_local]
            cand_w = cands_w_clean[i_local]
            rid_w, t_w = _winner_from_dist(dist_row, cand_w, reseller_node, reseller_id, reseller_attr)
            if rid_w < 0 or not np.isfinite(t_w):
                src_node = nodes_clean[i_local]
                rid_w, t_w, used_fb, _ = _run_mode_with_fallback(
                    src_node, r, c, graph_walk, walk_friction_min, cand_w,
                    r_rows, r_cols, reseller_node, reseller_id, reseller_attr,
                    comp_w_clean[i_local], reseller_comp_walk, reseller_comp_index_walk,
                    INITIAL_LIMIT_FACTOR_WALK, FINAL_LIMIT_FACTOR_WALK)
                if used_fb:
                    profile["counts"]["walk_fallback_used"] += 1
            best_id_walk[r, c] = rid_w
            best_time_walk[r, c] = t_w

    # Car mode
    car_indices = np.where(need_car_clean)[0]
    if len(car_indices) > 0:
        cands_car = []
        limits_car = []
        car_sources = []
        for i_local in car_indices:
            r, c = rows_clean[i_local], cols_clean[i_local]
            base = cands_w_clean[i_local]
            _, nn = reseller_tree.query([r, c], k=k_extra_car)
            nn = np.atleast_1d(nn).astype(np.int32)
            cand_c = np.unique(np.concatenate([base, nn]))
            cands_car.append(cand_c)
            lb = np.hypot(r_rows[cand_c] - r, r_cols[cand_c] - c) * CELL_SIZE_METERS * max(moto_friction_min, EPS)
            lim = float(np.nanmax(lb) * INITIAL_LIMIT_FACTOR_CAR + LIMIT_MARGIN_MIN) if lb.size > 0 else np.inf
            limits_car.append(lim)
            car_sources.append(nodes_clean[i_local])

        global_limit_car = float(np.max(limits_car)) if limits_car else np.inf
        tc_start = time.time()
        dist_car = dijkstra(csgraph=graph_moto, directed=True,
                            indices=car_sources, limit=global_limit_car)
        profile["timings_sec"]["car_mode"] += time.time() - tc_start

        for pos, i_local in enumerate(car_indices):
            r, c = rows_clean[i_local], cols_clean[i_local]
            dist_row = dist_car[pos]
            cand_c = cands_car[pos]
            rid_c, t_c = _winner_from_dist(dist_row, cand_c, reseller_node, reseller_id, reseller_attr)
            if rid_c < 0 or not np.isfinite(t_c):
                src_node = nodes_clean[i_local]
                rid_c, t_c, used_fb, _ = _run_mode_with_fallback(
                    src_node, r, c, graph_moto, moto_friction_min, cand_c,
                    r_rows, r_cols, reseller_node, reseller_id, reseller_attr,
                    comp_m_clean[i_local], reseller_comp_moto, reseller_comp_index_moto,
                    INITIAL_LIMIT_FACTOR_CAR, FINAL_LIMIT_FACTOR_CAR)
                if used_fb:
                    profile["counts"]["car_fallback_used"] += 1
            best_id_car[r, c] = rid_c
            best_time_car[r, c] = t_c

    for i_local in range(n_clean):
        if not need_car_clean[i_local]:
            r, c = rows_clean[i_local], cols_clean[i_local]
            best_id_car[r, c] = best_id_walk[r, c]
            best_time_car[r, c] = best_time_walk[r, c]
            profile["counts"]["car_copied_from_walk"] += 1

    processed += n_clean
    chunk_start = chunk_end

    if (processed % PROGRESS_EVERY) == 0 or chunk_start >= n_pix:
        elapsed = time.time() - loop_t0
        speed = processed / max(elapsed, 1e-9)
        rem = (n_pix - processed) / max(speed, 1e-9)
        print(f"  {processed:,}/{n_pix:,} ({100*processed/n_pix:.1f}%) | {speed:.1f} pix/s | ETA {_eta(rem)}")

# =============================================================================
# PROCESS OUTLIERS
# =============================================================================
if outlier_indices:
    print(f"  Processing {len(outlier_indices)} outlier pixels individually...")
    t_out_start = time.time()
    for idx in outlier_indices:
        r, c = pix_rows[idx], pix_cols[idx]
        src_node = pix_nodes[idx]
        comp_w = pix_comp_walk[idx]
        comp_m = pix_comp_moto[idx]
        cand_w = cand_walk_list[idx]

        rid_w, t_w, used_fb, _ = _run_mode_with_fallback(
            src_node, r, c, graph_walk, walk_friction_min, cand_w,
            r_rows, r_cols, reseller_node, reseller_id, reseller_attr,
            comp_w, reseller_comp_walk, reseller_comp_index_walk,
            INITIAL_LIMIT_FACTOR_WALK, FINAL_LIMIT_FACTOR_WALK)
        if used_fb:
            profile["counts"]["walk_fallback_used"] += 1
        best_id_walk[r, c] = rid_w
        best_time_walk[r, c] = t_w

        if need_car_mask[idx]:
            _, nn = reseller_tree.query([r, c], k=k_extra_car)
            nn = np.atleast_1d(nn).astype(np.int32)
            cand_c = np.unique(np.concatenate([cand_w, nn]))
            rid_c, t_c, used_fb, _ = _run_mode_with_fallback(
                src_node, r, c, graph_moto, moto_friction_min, cand_c,
                r_rows, r_cols, reseller_node, reseller_id, reseller_attr,
                comp_m, reseller_comp_moto, reseller_comp_index_moto,
                INITIAL_LIMIT_FACTOR_CAR, FINAL_LIMIT_FACTOR_CAR)
            if used_fb:
                profile["counts"]["car_fallback_used"] += 1
            best_id_car[r, c] = rid_c
            best_time_car[r, c] = t_c
        else:
            best_id_car[r, c] = rid_w
            best_time_car[r, c] = t_w
            profile["counts"]["car_copied_from_walk"] += 1

    profile["timings_sec"]["outliers"] = time.time() - t_out_start
    profile["counts"]["outliers_processed"] = len(outlier_indices)

print("[7/8] Chunk processing completed.")

# =============================================================================
# WRITE OUTPUTS
# =============================================================================
print("[8/8] Writing outputs...")
best_id_walk_float = np.where(best_id_walk >= 0, best_id_walk.astype(np.float32), np.nan)
best_id_car_float = np.where(best_id_car >= 0, best_id_car.astype(np.float32), np.nan)

_write_multiband_pixel_raster(
    OUTPUT_PIXEL_RASTER,
    ref_profile,
    bands=[
        car_share.astype(np.float32),
        walk_share.astype(np.float32),
        best_id_walk_float,
        best_time_walk.astype(np.float32),
        best_id_car_float,
        best_time_car.astype(np.float32),
    ],
    names=[
        "car_share",
        "walk_share",
        "best_reseller_id_walk",
        "best_time_walk_min",
        "best_reseller_id_car",
        "best_time_car_min",
    ],
)
print(f"Saved: {OUTPUT_PIXEL_RASTER}")

lookup = pd.DataFrame({
    "reseller_id": reseller_id,
    "attractiveness": reseller_attr,
    "row": r_rows,
    "col": r_cols,
}).drop_duplicates(subset=["reseller_id"])
lookup.to_csv(OUTPUT_LOOKUP_CSV, index=False)
print(f"Saved lookup: {OUTPUT_LOOKUP_CSV}")

# =============================================================================
# SUMMARY
# =============================================================================
valid_walk = (best_id_walk[inhabited] >= 0) & np.isfinite(best_time_walk[inhabited]) & (best_time_walk[inhabited] < UNASSIGNED_TIME_MIN)
valid_car = (best_id_car[inhabited] >= 0) & np.isfinite(best_time_car[inhabited]) & (best_time_car[inhabited] < UNASSIGNED_TIME_MIN)

profile["timings_sec"]["total"] = float(time.time() - t0)
profile["counts"]["walk_assigned"] = int(valid_walk.sum())
profile["counts"]["car_assigned"] = int(valid_car.sum())
profile["counts"]["walk_unassigned"] = int(n_pix - valid_walk.sum())
profile["counts"]["car_unassigned"] = int(n_pix - valid_car.sum())

print("\n=== SUMMARY ===")
print(f"Pixels processed: {n_pix:,}")
print(f"Walk assigned: {int(valid_walk.sum()):,} ({100.0 * valid_walk.mean():.2f}%)")
print(f"Car assigned : {int(valid_car.sum()):,} ({100.0 * valid_car.mean():.2f}%)")
if valid_walk.any():
    w = best_time_walk[inhabited][valid_walk]
    print(f"Walk time min/median/max (min): {np.nanmin(w):.2f} / {np.nanmedian(w):.2f} / {np.nanmax(w):.2f}")
if valid_car.any():
    c = best_time_car[inhabited][valid_car]
    print(f"Car  time min/median/max (min): {np.nanmin(c):.2f} / {np.nanmedian(c):.2f} / {np.nanmax(c):.2f}")

if ENABLE_PROFILING:
    with open(PROFILE_LOG_JSON, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    print("\n=== PROFILING ===")
    print(f"Outliers processed: {profile['counts']['outliers_processed']:,}")
    print(f"Walk fallback used: {profile['counts']['walk_fallback_used']:,}")
    print(f"Car fallback used : {profile['counts']['car_fallback_used']:,}")
    print(f"Car copied from walk: {profile['counts']['car_copied_from_walk']:,}")
    print(f"Total time: {_eta(profile['timings_sec']['total'])}")
    print(f"  Pre‑compute: {_eta(profile['timings_sec']['precompute'])}")
    print(f"  Walk Dijkstra: {_eta(profile['timings_sec']['walk_mode'])}")
    print(f"  Car Dijkstra : {_eta(profile['timings_sec']['car_mode'])}")
    print(f"  Outliers     : {_eta(profile['timings_sec']['outliers'])}")
    print(f"Timing log: {PROFILE_LOG_JSON}")

print(f"\nDone in {_eta(time.time() - t0)}")