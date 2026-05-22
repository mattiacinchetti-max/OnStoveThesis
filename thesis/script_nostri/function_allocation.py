from __future__ import annotations

from parameters import *

import pandas as pd
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt 
import numpy as np
import rasterio
from rasterio.mask import mask as raster_mask
import shapely
from shapely.geometry import Point, box, shape, LineString
from skimage.graph import MCP_Geometric
from pyproj import Geod, CRS
from onstove.layer import VectorLayer, RasterLayer
from shapely.ops import nearest_points
from scipy.optimize import linprog
from openpyxl import Workbook
import warnings
import contextlib
import heapq
import io
import json
import math
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional
import rioxarray
import xarray as xr
from onstove import OnStove
from rasterio.features import geometry_mask, shapes
from rasterio.transform import rowcol
from rasterio.warp import Resampling, reproject, transform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
import function_tools as tool
import function_costs as cost
import function_process as process
from dataclasses import dataclass

def build_cost_surface(friction_path):
    """
    Build a cost surface from friction data and expose CRS utilities.
    Converts friction values to hours/km and sets nodata to inf.
    Returns the cost array, transform, geodesic helper, and CRS.
    Use the returned CRS for routing projections.
    """
    friction = RasterLayer(name='friction_surface', path=str(friction_path))
    cost_layer = friction.data.astype(np.float64).copy()
    cost_layer *= (1000.0 / 60.0)  # Convert to hours/km equivalent
    
    nodata = friction.meta.get('nodata')
    cost_layer[np.isnan(cost_layer)] = np.inf
    if nodata is not None:
        cost_layer[cost_layer == nodata] = np.inf
    cost_layer[cost_layer < 0] = np.inf
    
    crs_obj = friction.meta.get('crs')
    use_geodesic = bool(getattr(crs_obj, 'is_geographic', False))
    geod = Geod(ellps='WGS84') if use_geodesic else None
    
    return cost_layer, friction.meta['transform'], geod, crs_obj


def calculate_mcp_routes(origin_row, origin_col, dest_rows, dest_cols, cost_layer, transform, geod):
    """
    Compute travel time and path distance from one origin to many destinations.
    Uses MCP to find least-time paths on the cost surface.
    Returns arrays of hours and km aligned to destination lists.
    Invalid destinations remain NaN.
    """
    max_row, max_col = cost_layer.shape
    times_hours = np.full(len(dest_rows), np.nan, dtype=np.float64)
    distances_km = np.full(len(dest_rows), np.nan, dtype=np.float64)

    if not (0 <= origin_row < max_row and 0 <= origin_col < max_col):
        return times_hours, distances_km

    mcp = MCP_Geometric(cost_layer, fully_connected=True)
    cumulative_costs, _ = mcp.find_costs(starts=np.array([[origin_row, origin_col]], dtype=np.int64))

    for idx, (d_row, d_col) in enumerate(zip(dest_rows, dest_cols)):
        if 0 <= d_row < max_row and 0 <= d_col < max_col:
            val = cumulative_costs[d_row, d_col]
            if np.isfinite(val):
                times_hours[idx] = float(val)
                
                path = mcp.traceback((d_row, d_col))
                if len(path) <= 1:
                    distances_km[idx] = 0.0
                    continue
                    
                p_rows = np.array([p[0] for p in path], dtype=np.int64)
                p_cols = np.array([p[1] for p in path], dtype=np.int64)
                xs, ys = rasterio.transform.xy(transform, p_rows, p_cols, offset='center')
                
                xs = np.asarray(xs, dtype=np.float64)
                ys = np.asarray(ys, dtype=np.float64)

                if geod is not None:
                    _, _, seg_m = geod.inv(xs[:-1], ys[:-1], xs[1:], ys[1:])
                    distances_km[idx] = float(np.nansum(np.abs(seg_m)) / 1000.0)
                else:
                    dx = np.diff(xs)
                    dy = np.diff(ys)
                    distances_km[idx] = float(np.nansum(np.hypot(dx, dy)) / 1000.0)
                    
    return times_hours, distances_km


def assign_nearest_storage_by_traveltime(facilities_gdf, storage_gdf, friction_path=None, cost_surface_data=None):
    """
    Assign nearest storage by least-time path and compute route distance.
    Adds storage id/name, travel time (minutes), and route distance (km).
    Uses the friction raster cost surface for routing.
    Returns a GeoDataFrame in the original CRS.
    """
    facilities = facilities_gdf.copy()

    if facilities.empty:
        facilities['tank_fid'] = pd.Series(dtype='object')
        facilities['tank_name'] = pd.Series(dtype='object')
        facilities['tank_distance'] = pd.Series(dtype='float64')
        facilities['tank_traveltime'] = pd.Series(dtype='float64')
        return facilities

    if storage_gdf is None or storage_gdf.empty:
        facilities['tank_fid'] = None
        facilities['tank_name'] = None
        facilities['tank_distance'] = np.nan
        facilities['tank_traveltime'] = np.nan
        return facilities

    # Build cost surface and get projection info once
    if cost_surface_data is not None:
        cost_layer, transform, geod, friction_crs = cost_surface_data
    else:
        friction_file = Path(friction_path)
        if not friction_file.exists():
            raise FileNotFoundError(f"Missing friction raster: {friction_file}")
        cost_layer, transform, geod, friction_crs = build_cost_surface(friction_file)

    if friction_crs is None:
        raise ValueError("Friction raster has no CRS in metadata")

    facilities_tt = facilities.to_crs(friction_crs).copy()
    storage_tt = storage_gdf.to_crs(friction_crs).copy().reset_index(drop=True)

    # Build a guaranteed unique tank identifier
    fid_col = next((col for col in ['FID', 'fid', 'id', 'id_res&fil'] if col in storage_tt.columns), None)

    storage_tt['__tank_fid'] = storage_tt.index.astype(str)
    if fid_col is not None:
        fid_vals = storage_tt[fid_col].astype(str)
        invalid = fid_vals.str.strip().isin(['', 'nan', 'None'])
        duplicated = fid_vals.duplicated(keep=False)
        use_col = (~invalid) & (~duplicated)
        storage_tt.loc[use_col, '__tank_fid'] = fid_vals.loc[use_col]

    name_col = next((col for col in storage_tt.columns if str(col).lower() == 'name'), None)
    if name_col is None:
        storage_tt['__tank_name'] = [f'primary_storage_{i}' for i in storage_tt.index]
        name_col = '__tank_name'
    else:
        storage_tt[name_col] = storage_tt[name_col].astype(str)
        missing_names = storage_tt[name_col].str.strip().isin(['', 'nan', 'None'])
        storage_tt.loc[missing_names, name_col] = [f'primary_storage_{i}' for i in storage_tt.index[missing_names]]

    facility_points = facilities_tt.geometry
    if not facilities_tt.geom_type.eq('Point').all():
        facility_points = facilities_tt.geometry.centroid

    storage_points = storage_tt.geometry
    if not storage_tt.geom_type.eq('Point').all():
        storage_points = storage_tt.geometry.centroid

    # Precompute raster indices for facilities and tanks once
    fac_rows, fac_cols = [], []
    for geom in facility_points:
        row, col = rasterio.transform.rowcol(transform, geom.x, geom.y)
        fac_rows.append(row)
        fac_cols.append(col)

    tank_rows, tank_cols = [], []
    for geom in storage_points:
        row, col = rasterio.transform.rowcol(transform, geom.x, geom.y)
        tank_rows.append(row)
        tank_cols.append(col)

    travel_matrix = np.full((len(facilities_tt), len(storage_tt)), np.nan, dtype=np.float64)
    distance_matrix = np.full((len(facilities_tt), len(storage_tt)), np.nan, dtype=np.float64)

    # Compute routes from each tank to all facilities
    for tank_idx, (t_row, t_col) in enumerate(zip(tank_rows, tank_cols)):
        t_times, t_dists = calculate_mcp_routes(
            t_row, t_col, fac_rows, fac_cols, cost_layer, transform, geod
        )
        travel_matrix[:, tank_idx] = t_times
        distance_matrix[:, tank_idx] = t_dists

    # Assign nearest storage based on travel time
    tank_id_supply = np.array([None] * len(facilities_tt), dtype=object)
    tank_name = np.array([None] * len(facilities_tt), dtype=object)
    tank_distance = np.full(len(facilities_tt), np.nan, dtype=np.float64)
    tank_traveltime = np.full(len(facilities_tt), np.nan, dtype=np.float64)

    valid = np.any(np.isfinite(travel_matrix), axis=1)
    if np.any(valid):
        best_tank_idx = np.nanargmin(travel_matrix[valid], axis=1)
        valid_rows = np.where(valid)[0]

        storage_ids = storage_tt['id_supply'].astype(str).to_numpy()
        storage_names = storage_tt[name_col].astype(str).to_numpy()
        
        tank_id_supply[valid] = storage_ids[best_tank_idx]
        tank_name[valid] = storage_names[best_tank_idx]

        best_tt_hours = travel_matrix[valid_rows, best_tank_idx]
        best_tt_min = best_tt_hours * 60.0
        tank_traveltime[valid] = best_tt_min

        tank_distance[valid] = distance_matrix[valid_rows, best_tank_idx]

    facilities['tank_id_supply'] = tank_id_supply
    facilities['tank_name'] = tank_name
    facilities['tank_distance'] = tank_distance
    facilities['tank_traveltime'] = tank_traveltime
    
    return facilities

def calculate_percentages_supply(refineries_gdf, ports_gdf, gas_plants_gdf, border_points_gdf, nat_shares_df):
    """
    Calculate facility-level market shares from national shares and capacities.
    Uses capacity-based weights for refineries, ports, and gas plants.
    Keeps border point percentages computed earlier.
    Returns a combined GeoDataFrame and a per-category dict.
    """

    # Choose a common CRS for all facility layers before concatenation
    target_crs = refineries_gdf.crs or ports_gdf.crs or gas_plants_gdf.crs or border_points_gdf.crs

    # For refineries, use crude_capacity for proportional split
    refineries_weight_col = 'crude_capacity'

    def _numeric_series_or_zero(gdf, col):
        if col in gdf.columns:
            return pd.to_numeric(gdf[col], errors='coerce').fillna(0.0)
        return pd.Series(0.0, index=gdf.index, dtype='float64')

    # Calculate totals
    totals = {
        'refineries': _numeric_series_or_zero(refineries_gdf, refineries_weight_col).sum(),
        'ports': _numeric_series_or_zero(ports_gdf, 'LPG_capacity').sum(),
        'gas_plants': _numeric_series_or_zero(gas_plants_gdf, 'LPG_prod').sum(),
    }

    # Get national shares from raster-derived values (no fallback defaults)
    nat_shares = {
        'refineries': nat_shares_df.loc['refineries'].iloc[0] / 100,
        'ports': nat_shares_df.loc['ports'].iloc[0] / 100,
        'gas_plants': nat_shares_df.loc['gas_plants'].iloc[0] / 100,
        'border_points': nat_shares_df.loc['border_points'].iloc[0] / 100,
    }

    print(f"Shares: {nat_shares}")

    by_category = {}
    specs = [
        ('refineries', refineries_gdf, refineries_weight_col),
        ('ports', ports_gdf, 'LPG_capacity'),
        ('gas_plants', gas_plants_gdf, 'LPG_prod'),
    ]

    for src_name, src_gdf, weight_col in specs:
        gdf = src_gdf.copy()

        # Reproject each layer to a common CRS to avoid concat CRS conflicts
        if target_crs is not None and gdf.crs is not None and gdf.crs != target_crs:
            gdf = gdf.to_crs(target_crs)

        # Ensure required columns used later in the pipeline are present
        if 'name' not in gdf.columns:
            gdf['name'] = [f'{src_name}_{idx}' for idx in gdf.index]

        if 'LPG_price' not in gdf.columns:
            gdf['LPG_price'] = np.nan
        else:
            gdf['LPG_price'] = pd.to_numeric(gdf['LPG_price'], errors='coerce')

        if 'country' not in gdf.columns:
            gdf['country'] = 'Unknown'
        else:
            gdf['country'] = gdf['country'].fillna('Unknown')

        # Keep category in each layer and compute percentage
        gdf['category'] = src_name if src_name != 'gas_plants' else 'gasplants'

        weights = _numeric_series_or_zero(gdf, weight_col)

        if totals[src_name] > 0:
            gdf['percentage'] = nat_shares[src_name] * (weights / totals[src_name])
        else:
            gdf['percentage'] = 0.0

        by_category[src_name] = gdf

    # Border points: keep precomputed percentages from processing step
    border = border_points_gdf.copy()
    if target_crs is not None and border.crs is not None and border.crs != target_crs:
        border = border.to_crs(target_crs)

    if 'name' not in border.columns:
        border['name'] = [f'border_points_{idx}' for idx in border.index]

    if 'LPG_price' not in border.columns:
        border['LPG_price'] = np.nan
    else:
        border['LPG_price'] = pd.to_numeric(border['LPG_price'], errors='coerce')

    if 'country' not in border.columns:
        border['country'] = 'Unknown'
    else:
        border['country'] = border['country'].fillna('Unknown')

    border['category'] = 'border_points'
    if 'percentage' not in border.columns:
        border['percentage'] = 0.0
    else:
        border['percentage'] = pd.to_numeric(border['percentage'], errors='coerce').fillna(0.0)

    by_category['border_points'] = border

    result = gpd.GeoDataFrame(
        pd.concat(
            [
                by_category['refineries'],
                by_category['ports'],
                by_category['gas_plants'],
                by_category['border_points'],
            ],
            ignore_index=True,
        ),
        crs=target_crs,
    )

    print(f"✓ {len(result)} sources calculated | Total share: {result['percentage'].sum():.2%}")
    return result, by_category

def calculate_percentages_filling(filling_gdf):
    """
    Calculate demand-based percentage shares for filling points.
    Searches for demand columns in priority order: percentage_demand, total_fil_clients, clients.
    If found column represents counts, converts to percentage share of total demand.
    Adds 'percentage_demand' column to the GeoDataFrame.
    """
    filling = filling_gdf.copy()
    
    # Determine the column to use for demand (priority: percentage, then client counts)
    demand_col = next((c for c in ["percentage_demand", "total_fil_clients", "clients"] if c in filling.columns), None)

    if demand_col:
        demand_vals = pd.to_numeric(filling[demand_col], errors="coerce").fillna(0.0)
        
        # If the column represents raw counts (clients), convert to a percentage share of total demand
        if demand_col != "percentage_demand":
            total_demand = demand_vals.sum()
            filling["percentage_demand"] = (demand_vals / total_demand) if total_demand > 0 else 0.0
        else:
            filling["percentage_demand"] = demand_vals
            
        print(f"✓ Demand shares calculated using column: {demand_col}")
        return filling
    else:
        raise ValueError("No demand-related column found in filling_gdf (expected: percentage_demand, total_fil_clients, or clients)")


def aggregate_supply_to_storage(primary_storage_gdf, refineries_gdf, ports_gdf, gas_plants_gdf, border_points_gdf):
    """
    Aggregate supply percentages from all sources (refineries, ports, gas plants, border points)
    to primary storage facilities at the storage level.
    """
    ps = primary_storage_gdf.copy()
    ps["id_supply"] = ps["id_supply"].astype(str)
    
    # Collect all supply layers
    supply_layers = [refineries_gdf, ports_gdf, gas_plants_gdf, border_points_gdf]
    supply_layers = [g for g in supply_layers if g is not None and not g.empty]
    
    if supply_layers:
        # Align CRS across all layers
        target_crs = next((g.crs for g in supply_layers if g.crs is not None), None)
        if target_crs is not None:
            aligned_layers = [g.to_crs(target_crs) if g.crs is not None and g.crs != target_crs else g for g in supply_layers]
        else:
            aligned_layers = supply_layers
        
        all_supply = pd.concat(aligned_layers, ignore_index=True)
    else:
        all_supply = gpd.GeoDataFrame()
    
    # Aggregate supply by tank_id_supply
    if not all_supply.empty and "tank_id_supply" in all_supply.columns:
        supply_sum = all_supply.groupby("tank_id_supply")["percentage"].sum()
    else:
        supply_sum = pd.Series(dtype="float64")
    
    ps["percentage_supply"] = ps["id_supply"].map(supply_sum).fillna(0.0)
    
    print(f"✓ Supply aggregated to {len(ps)} storage facilities | Total supply: {ps['percentage_supply'].sum():.4f}")
    return ps


def aggregate_filling_to_storage(primary_storage_gdf, filling_points_gdf):
    """
    Aggregate filling percentages from all filling points to primary storage facilities
    at the storage level.
    """
    ps = primary_storage_gdf.copy()
    ps["id_supply"] = ps["id_supply"].astype(str)
    
    # Aggregate filling percentages by tank_id_supply
    if filling_points_gdf is not None and not filling_points_gdf.empty and "tank_id_supply" in filling_points_gdf.columns:
        filling_sum = filling_points_gdf.groupby("tank_id_supply")["percentage_demand"].sum()
    else:
        filling_sum = pd.Series(dtype="float64")
    
    ps["percentage_demand"] = ps["id_supply"].map(filling_sum).fillna(0.0)
    
    print(f"✓ Filling aggregated to {len(ps)} storage facilities | Total filling: {ps['percentage_demand'].sum():.4f}")
    return ps


def build_storage_routing_matrices(primary_storage_gdf, friction_path):
    """
    Build travel time and distance matrices for inter-storage routing.
    Computes N×N matrices where N = number of storage facilities.
    """
    ps = primary_storage_gdf.copy()
    n_storage = len(ps)
    storage_ids = ps.index.tolist()
    
    # Build cost surface
    cost_layer, transform, geod, crs_obj = build_cost_surface(friction_path)
    
    if crs_obj is None:
        raise ValueError("Friction raster has no CRS in metadata")
    
    # Project storage to friction CRS
    storage_tt = ps.to_crs(crs_obj).copy()
    storage_points = storage_tt.geometry
    if not storage_tt.geom_type.eq('Point').all():
        storage_points = storage_tt.geometry.centroid
    
    # Precompute pixel coordinates
    storage_rows, storage_cols = [], []
    for geom in storage_points:
        r, c = rasterio.transform.rowcol(transform, geom.x, geom.y)
        storage_rows.append(r)
        storage_cols.append(c)
    
    # Compute N×N routing matrices
    travel_time_matrix = np.full((n_storage, n_storage), np.inf, dtype=np.float64)
    distance_matrix = np.full((n_storage, n_storage), np.nan, dtype=np.float64)
    
    for i, (orig_row, orig_col) in enumerate(zip(storage_rows, storage_cols)):
        t_times, t_dists = calculate_mcp_routes(
            orig_row, orig_col, storage_rows, storage_cols, cost_layer, transform, geod
        )
        travel_time_matrix[i, :] = t_times * 60.0  # Convert hours to minutes
        distance_matrix[i, :] = t_dists
    
    print(f"✓ Routing matrices computed for {n_storage} storage facilities\n")
    return travel_time_matrix, distance_matrix, crs_obj, storage_ids


def optimize_storage_allocation(primary_storage_gdf, travel_time_matrix):
    """
    Solve Linear Programming problem to optimize inter-storage allocation.
    Minimizes total travel time subject to supply/demand balance constraints.
    """
    ps = primary_storage_gdf.copy()
    n_storage = len(ps)
    
    # Flatten cost vector (travel times)
    c = travel_time_matrix.flatten()
    
    # Build constraint matrices
    A_eq = []
    b_eq = []
    
    # Supply constraints: sum(x[i,:]) = supply[i]
    for i in range(n_storage):
        row = np.zeros(n_storage * n_storage)
        row[i * n_storage:(i+1) * n_storage] = 1
        A_eq.append(row)
        b_eq.append(float(ps.iloc[i]['percentage_supply']))
    
    # Demand constraints: sum(x[:,j]) = demand[j]
    for j in range(n_storage):
        row = np.zeros(n_storage * n_storage)
        for i in range(n_storage):
            row[i * n_storage + j] = 1
        A_eq.append(row)
        b_eq.append(float(ps.iloc[j]['percentage_demand']))
    
    A_eq = np.array(A_eq)
    b_eq = np.array(b_eq)
    
    # Solve LP
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')
    
    if result.success:
        allocation_matrix = result.x.reshape((n_storage, n_storage))
        print(f"✓ Optimization successful\n")
        print(f"  Objective (total travel time): {result.fun:.4f} minutes\n")
    else:
        print(f"✗ Optimization failed: {result.message}\n")
        allocation_matrix = np.zeros((n_storage, n_storage))
    
    return allocation_matrix, result


def write_allocation_excel(primary_storage_gdf, travel_time_matrix, distance_matrix, allocation_matrix, output_path):
    """
    Write storage allocation results to Excel workbook with three sheets.
    """
    ps = primary_storage_gdf.copy()
    storage_ids = ps.index.tolist()
    storage_labels = [f"{i}" for i in storage_ids]
    n_storage = len(ps)
    
    wb = Workbook()
    wb.remove(wb.active)
    
    # Sheet 1: Travel Time (minutes)
    ws1 = wb.create_sheet("storage_allocation_time")
    ws1.append(["Storage ID"] + storage_labels)
    for i, label_i in enumerate(storage_labels):
        row_data = [label_i] + travel_time_matrix[i, :].tolist()
        ws1.append(row_data)
    
    # Sheet 2: Distance (km)
    ws2 = wb.create_sheet("storage_allocation_distance")
    ws2.append(["Storage ID"] + storage_labels)
    for i, label_i in enumerate(storage_labels):
        row_data = [label_i] + distance_matrix[i, :].tolist()
        ws2.append(row_data)
    
    # Sheet 3: Allocation (percentages)
    ws3 = wb.create_sheet("storage_allocation_percentage")
    ws3.append(["Storage ID"] + storage_labels)
    for i, label_i in enumerate(storage_labels):
        row_data = [label_i] + allocation_matrix[i, :].tolist()
        ws3.append(row_data)
    
    wb.save(output_path)
    print(f"✓ Saved: {output_path}")
    
    return Path(output_path)


def allocation_process(primary_storage_gdf, friction_path, output_excel_path):
    """
    Orchestrate complete storage allocation optimization workflow.
    Builds routing matrices, solves LP, validates results, and exports to Excel.
    """
    ps = primary_storage_gdf.copy()
    n_storage = len(ps)
    
    print(f"Primary storage facilities: {n_storage}")
    print(f"Supply: {ps['percentage_supply'].sum():.4f}")
    print(f"Demand: {ps['percentage_demand'].sum():.4f}\n")
    print()
    
    # Step 1: Build routing matrices
    travel_time_matrix, distance_matrix, crs_obj, storage_ids = build_storage_routing_matrices(ps, friction_path)
    print()
    
    # Step 2: Optimize allocation
    allocation_matrix, opt_result = optimize_storage_allocation(ps, travel_time_matrix)
    print()
    
    # Step 3: Validate
    total_allocated = allocation_matrix.sum()
    total_supply = ps['percentage_supply'].sum()
    total_demand = ps['percentage_demand'].sum()
    
    print(f"Validation:")
    print(f"  Total allocation: {total_allocated:.6f}")
    print(f"  Total supply: {total_supply:.6f}")
    print(f"  Total demand: {total_demand:.6f}\n")
    
    if abs(total_allocated - total_supply) > 1e-6 or abs(total_allocated - total_demand) > 1e-6:
        print(f"  ⚠ Warning: allocation does not balance!\n")
    else:
        print(f"  ✓ Allocation balanced\n")
    print()
    
    # Step 4: Export
    write_allocation_excel(ps, travel_time_matrix, distance_matrix, allocation_matrix, output_excel_path)
    
    return {
        'allocation_matrix': allocation_matrix,
        'travel_time_matrix': travel_time_matrix,
        'distance_matrix': distance_matrix,
        'optimization_result': opt_result,
        'n_storage': n_storage,
        'total_supply': total_supply,
        'total_demand': total_demand,
    }
