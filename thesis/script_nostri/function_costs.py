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
from rasterio.transform import rowcol
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
import function_allocation as alloc
import function_process as process
from dataclasses import dataclass

# Francesco part

def ports(gdf, import_cost_df):
    """
    Calculates extra costs (Import, STS, Pre-bottling) and updates final price.
    Each cost component is saved as a separate attribute.
    """
    gdf = gdf.copy()
    
    # Retrieve base import rate from the dataframe
    import_sea_rate = import_cost_df.loc['import_sea'].iloc[0]

    # Initialize individual cost columns
    gdf['cost_import_sea'] = 0.0
    gdf['cost_sts'] = 0.0
    gdf['cost_pre_bottling'] = 0.0

    # 1. Sea Import Cost: Applied to all ports based on the base price
    gdf['cost_import_sea'] = gdf['cost_source'] * import_sea_rate

    # 2. STS Cost: Applied if storage is available (LPG_compliance) but port is NOT VLGC compliant
    mask_sts = (gdf['LPG_compliance'] == True) & (gdf['VLGC_compliance'] == False)
    gdf.loc[mask_sts, 'cost_sts'] = STS_cost

    # 3. Pre-bottling Cost: Applied only if no nearby storage is available
    # It is calculated on the price already including sea import costs
    mask_no_storage = (gdf['LPG_compliance'] == False)
    price_after_import = gdf['cost_source'] + gdf['cost_import_sea']
    gdf.loc[mask_no_storage, 'cost_pre_bottling'] = price_after_import * pre_bottling_cost

    # Check for remaining missing prices
    missing_prices = int(gdf['cost_source'].isna().sum())
    if missing_prices > 0:
        print(f"Warning: {missing_prices} port rows have missing cost_source after cost calculation")

    return gdf

def border_points(gdf, import_cost_df, vehicle: Vehicle):
    """
    Calculates Land Import costs, Border Time costs, and Ferry costs.
    Updates final cost_source and stores individual components as attributes.
    """
    bp = gdf.copy()
    
    # Initialize cost attribute columns
    bp['cost_import_land'] = 0.0
    bp['cost_border_wait'] = 0.0
    bp['cost_ferry'] = 0.0

    # 1. Land Import Cost (Percentage markup on base price)
    import_land_rate = float(import_cost_df.loc['import_land'].iloc[0])
    bp['cost_import_land'] = bp['cost_source'] * import_land_rate

    # Logistics constants for cost calculation
    effective_load_kg = vehicle.capacity_kg * vehicle.utilization_factor
    driver_hourly_cost_usd = (driver_annual_salary_usd * salary_multiplier) / (hours_per_day * days_per_year)

    # 2. Border Waiting Time Cost
    # Logic: OSBP (One-Stop Border Post) reduces waiting time by half
    osbp_flags = tool._as_bool_series(bp['osbp']).to_numpy(dtype=bool)
    wait_hours = np.where(osbp_flags, border_point_time_hours / 2.0, border_point_time_hours)
    border_time_cost_trip = wait_hours * driver_hourly_cost_usd
    bp['cost_border_wait'] = border_time_cost_trip / effective_load_kg

    # 3. Ferry Cost (if applicable)
    if 'border_ferry' in bp.columns:
        is_ferry = tool._as_bool_series(bp['border_ferry']).to_numpy(dtype=bool)
        if is_ferry.any():
            # Ferry cost = distance * (rate per km + time-based driver cost)
            ferry_extra_cost_trip = bp['distance_ferry'] * (
                bp.get('ferry_cost_per_km', ferry_cost_per_km) + 
                (bp.get('ferry_time_per_km', ferry_time_per_km) * driver_hourly_cost_usd)
            )
            bp.loc[is_ferry, 'cost_ferry'] = ferry_extra_cost_trip / effective_load_kg

    return bp

def crf(rate: float, years: int) -> float:
    """
    Compute the capital recovery factor.
    Used to annualize upfront costs over an asset lifetime.
    Returns a dimensionless multiplier.
    """
    return (rate * (1.0 + rate) ** years) / ((1.0 + rate) ** years - 1.0)

driver_hourly_cost_usd = (driver_annual_salary_usd * salary_multiplier) / (hours_per_day * days_per_year)

def single_travel_fixed(avg_one_way_dist_km: float, vehicle: Vehicle) -> float:
    """
    Compute fixed transport cost per kg for an average route.
    Uses annualized capital and license costs.
    Scales by expected trips and effective load.
    """
    avg_round_trip_km = 2.0 * avg_one_way_dist_km
    trips_per_year = vehicle.annual_km / avg_round_trip_km
    effective_load_kg = vehicle.capacity_kg * vehicle.utilization_factor
    crf_vehicle = crf(discount_rate, vehicle.life_years)
    crf_license = crf(discount_rate, licence_life_years)
    annual_capital_cost_usd = vehicle.overnight_cost_usd * crf_vehicle
    annual_license_cost_usd = license_cost_usd * crf_license
    fixed_annual_cost_usd = annual_capital_cost_usd + annual_license_cost_usd
    fixed_cost_per_kg = fixed_annual_cost_usd / (effective_load_kg * trips_per_year)
    return fixed_cost_per_kg

def single_travel_variable(one_way_distance_km: float, one_way_time_min: float, vehicle: Vehicle) -> float:
    """
    Compute variable transport cost per kg for a single route.
    Includes distance-based operating costs and labor time.
    Uses the effective load to normalize per kg.
    """
    round_trip_hours = (one_way_time_min * 2.0 / 60.0) + vehicle.fixed_loading_unloading_hours
    round_trip_distance_km = 2.0 * one_way_distance_km
    variable_cost_trip = (
        vehicle.variable_cost_per_km * round_trip_distance_km 
        + driver_hourly_cost_usd * round_trip_hours
    )
    effective_load_kg = vehicle.capacity_kg * vehicle.utilization_factor
    variable_cost_per_kg = variable_cost_trip / effective_load_kg
    return variable_cost_per_kg

def single_travel(
    one_way_distance_km: float,
    one_way_time_min: float,
    avg_one_way_dist_km: float,
    vehicle: Vehicle
) -> dict:
    """
    Compute fixed, variable, and total transport costs per kg.
    Fixed cost uses average route distance; variable uses per-route inputs.
    Returns a dict with component and total values.
    """
    fixed_cost = single_travel_fixed(avg_one_way_dist_km, vehicle)
    variable_cost = single_travel_variable(one_way_distance_km, one_way_time_min, vehicle)
    total_cost = fixed_cost + variable_cost
    
    return {
        'fixed_cost_per_kg': fixed_cost,
        'variable_cost_per_kg': variable_cost,
        'total_cost_per_kg': total_cost
    }

def storage(
    primary_storage_gdf, 
    filling_points_gdf, 
    lpg_household_shares_df,
    use_default_time=False,
    use_differentiated=False
):
    """
    Computes LPG storage costs based on three possible residence time strategies:
    Default (15 days), Differentiated (Supply vs Demand), or National Average.
    """
    ps = primary_storage_gdf.copy()
    
    # 1. Capacity Processing (m3 to kg)
    # Convert capacity to numeric and fill missing values with the mean
    raw_cap = pd.to_numeric(ps.get("LPG_capacity"), errors="coerce")
    avg_cap = raw_cap[np.isfinite(raw_cap) & (raw_cap > 0)].mean()
    total_cap_kg = raw_cap.fillna(avg_cap) * kg_per_m3
    
    # Apply household share adjustment
    pct_house = float(lpg_household_shares_df.loc['percentage_house'].iloc[0])
    used_cap_kg = total_cap_kg * pct_house
    ps["used_capacity_kg"] = used_cap_kg

    # 2. National Flow Calculation
    # Annual consumption per client based on energy requirements
    annual_kg_client = (meals_per_day * 365.0 * energy_per_meal_mj) / (efficiency * energy_content_mj_per_kg)
    total_clients = pd.to_numeric(filling_points_gdf.get("total_fil_clients"), errors="coerce").fillna(0.0).sum()
    total_annual_flow = float(total_clients) * annual_kg_client

    # 3. Residence Time Determination (years)
    if use_default_time:
        first_res_years = np.full(len(ps), 15.0 / 365.0)
        second_res_years = np.zeros(len(ps))
        
    elif use_differentiated:
        # Based on allocated supply percentages
        pct_supply = pd.to_numeric(ps.get("percentage_supply"), errors="coerce").fillna(0.0)
        tank_supply = total_annual_flow * pct_supply
        first_res_years = np.where(tank_supply > 0, used_cap_kg / tank_supply, np.nan)

        # Based on allocated demand percentages
        pct_demand = pd.to_numeric(ps.get("percentage_demand"), errors="coerce").fillna(0.0)
        tank_demand = total_annual_flow * pct_demand
        second_res_years = np.where(tank_demand > 0, used_cap_kg / tank_demand, np.nan)
        
    else:
        # National Average logic
        total_nat_cap = used_cap_kg.sum()
        nat_res_years = (total_nat_cap / total_annual_flow) if total_annual_flow > 0 else np.nan
        first_res_years = np.full(len(ps), nat_res_years)
        second_res_years = np.full(len(ps), nat_res_years)

    # 4. Internal Cost Calculation Logic
    def _get_cost(res_time_series):
        res_time_series = pd.Series(res_time_series, index=ps.index)
        if np.all(res_time_series == 0):
            return pd.Series(0.0, index=ps.index)
        
        crf_capex = crf(discount_rate_storage, storage_life_years)
        crf_license = crf(discount_rate_storage, storage_license_life_years)
        
        # Avoid division by zero for capacity
        safe_cap = total_cap_kg.replace(0, np.nan)
        
        capex = (storage_overnight_cost_usd_kg + (storage_installation_fee / safe_cap)) * crf_capex
        license = (storage_license_cost_usd / safe_cap) * crf_license
        
        return ((capex + license) * res_time_series).clip(upper=storage_cap)

    # 5. Apply calculation to both stages
    ps["cost_storage"] = _get_cost(first_res_years).fillna(0.0)
    ps["cost_storage_second"] = _get_cost(second_res_years).fillna(0.0)

    return ps

def average_transport_distance(points_dict):
    """
    Calculate weighted average one-way transport distance from all facilities to storage.
    Weight is based on allocation percentages (percentage column).
    
    Formula: avg_dist = Σ(distance_i × percentage_i) / Σ(percentage_i)
    
    This ensures facilities with higher allocation percentages influence the 
    average distance more, reflecting actual logistics flows.
    """
    facility_keys = ["refineries", "ports", "gas_plants", "border_points"]
    
    weighted_distances = []
    total_weight = 0
    
    for key in facility_keys:
        gdf = points_dict.get(key)
        if gdf is None or gdf.empty:
            continue
        if 'tank_distance' not in gdf.columns or 'percentage' not in gdf.columns:
            continue
        
        # Extract distances and percentages
        dist = pd.to_numeric(gdf['tank_distance'], errors='coerce').to_numpy(dtype=float)
        pct = pd.to_numeric(gdf['percentage'], errors='coerce').to_numpy(dtype=float)
        
        # Valid rows: both distance and percentage are finite and positive
        valid = np.isfinite(dist) & (dist > 0) & np.isfinite(pct) & (pct > 0)
        
        if valid.any():
            valid_dist = dist[valid]
            valid_pct = pct[valid]
            
            # Weighted sum for this facility type
            weighted_distances.append(np.sum(valid_dist * valid_pct))
            total_weight += np.sum(valid_pct)
    
    if total_weight == 0:
        raise ValueError("No valid weighted distances found (all allocation percentages are zero)")
    
    avg_distance = float(np.sum(weighted_distances) / total_weight)
    
    print(f"✓ Average transport distance (allocation-weighted): {avg_distance:.2f} km")
    
    return avg_distance

def transport(
    layer1: gpd.GeoDataFrame,
    layer2: gpd.GeoDataFrame,
    vehicle: Vehicle,
    mode: str,  # 'convergence', 'cross_exchange', 'divergence'
    layer1_name: Optional[str] = None,
    layer2_name: Optional[str] = None,
    weight1: Optional[str] = None,
    weight2: Optional[str] = None,
    distance_matrix: Optional[np.ndarray] = None,
    time_matrix: Optional[np.ndarray] = None,
    weight_matrix: Optional[np.ndarray] = None,
    id_col: str = 'id',
    output_col: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Compute transport costs and optionally aggregate.

    Modes:
        convergence (ex nearest): layer1 has nearest_{layer2_name}_distance/time/id + weight1.
            weight2 is in layer2. Returns layer2 with weighted average cost per destination.
            Weight used is weight1 / weight2.
        cross_exchange (ex all): uses distance_matrix, time_matrix, weight_matrix (n1 x n2).
            Returns layer2 with weighted average cost per destination.
        divergence (new): layer2 has nearest_{layer1_name}_distance/time.
            Returns layer1 with individual cost per origin.
    """
    # Helper to compute individual costs for valid rows using nearest_{name}_* columns
    def _cost_series(df, name, wcol=None):
        dcol = f"nearest_{name}_distance"
        tcol = f"nearest_{name}_time"
        valid = df[dcol].notna() & df[tcol].notna() & (df[dcol] > 0) & (df[tcol] >= 0)
        if wcol:
            valid = valid & df[wcol].notna() & (df[wcol] > 0)
        if not valid.any():
            return pd.Series(index=df.index, dtype=float)
        avg_dist = df.loc[valid, dcol].mean()
        costs = []
        for idx in df.index[valid]:
            d, t = df.loc[idx, dcol], df.loc[idx, tcol]
            costs.append(single_travel(d, t, avg_dist, vehicle)['total_cost_per_kg'])
        return pd.Series(costs, index=df.index[valid])

    if mode == 'divergence':
        if not layer1_name:
            raise ValueError("mode 'divergence' requires layer1_name")
        out = layer2.copy()
        col = output_col or f"cost_transport_{layer1_name}_to_{layer2_name}"
        out[col] = _cost_series(layer2, layer1_name)
        return out

    if mode == 'convergence':
        if not layer1_name or not weight1 or not weight2:
            raise ValueError("mode 'convergence' requires layer1_name, weight1, and weight2")
        if weight1 not in layer1.columns:
            raise ValueError(f"weight1 column '{weight1}' not found in layer1")
        if weight2 not in layer2.columns:
            raise ValueError(f"weight2 column '{weight2}' not found in layer2")
        
        costs = _cost_series(layer1, layer2_name, weight1)
        
        # Prepare data for ratio calculation
        w1 = pd.to_numeric(layer1[weight1], errors='coerce')
        
        # Create weight2 mapping from layer2
        w2_mapping = pd.Series(
            pd.to_numeric(layer2[weight2], errors='coerce').values,
            index=layer2[id_col].astype(str).values
        )
        
        temp = pd.DataFrame({
            'dest': layer1[f"nearest_{layer2_name}_id"].astype(str),
            'w1': w1,
            'c': costs
        })
        
        # Add weight2 based on destination
        temp['w2'] = temp['dest'].map(w2_mapping)
        
        # Calculate weight1/weight2 ratio
        temp['w'] = temp['w1'] / temp['w2']
        
        temp = temp.dropna(subset=['dest', 'w', 'c'])
        
        if temp.empty:
            out = layer2.copy()
            out[output_col or f"cost_transport_{layer1_name}_to_{layer2_name}"] = np.nan
            return out
        
        agg = temp.groupby('dest').apply(lambda g: (g['w'] * g['c']).sum() / g['w'].sum(), include_groups=False)
        out = layer2.copy()
        col = output_col or f"cost_transport_{layer1_name}_to_{layer2_name}"
        out[col] = out[id_col].astype(str).map(agg)
        return out

    if mode == 'cross_exchange':
        if any(x is None for x in [distance_matrix, time_matrix, weight_matrix]):
            raise ValueError("mode 'cross_exchange' requires distance_matrix, time_matrix, weight_matrix")
        n1, n2 = len(layer1), len(layer2)
        if not all(x.shape == (n1, n2) for x in (distance_matrix, time_matrix, weight_matrix)):
            raise ValueError("Matrix dimensions mismatch")
        valid = np.isfinite(distance_matrix) & (distance_matrix > 0) & np.isfinite(time_matrix) & (time_matrix >= 0)
        avg_dist = np.mean(distance_matrix[valid]) if valid.any() else 0.0
        cost_matrix = np.full((n1, n2), np.nan)
        for i in range(n1):
            for j in range(n2):
                if valid[i, j]:
                    cost_matrix[i, j] = single_travel(distance_matrix[i, j], time_matrix[i, j], avg_dist, vehicle)['total_cost_per_kg']
        wcost = np.full(n2, np.nan)
        for j in range(n2):
            w, c = weight_matrix[:, j], cost_matrix[:, j]
            m = np.isfinite(w) & (w > 0) & np.isfinite(c)
            if m.any():
                wcost[j] = np.sum(w[m] * c[m]) / np.sum(w[m])
        out = layer2.copy()
        col = output_col or f"cost_transport_{layer1_name}_to_{layer2_name}"
        out[col] = wcost
        return out

    raise ValueError("mode must be 'convergence', 'cross_exchange', or 'divergence'")

def tracking(
    layer1: gpd.GeoDataFrame,
    layer2: gpd.GeoDataFrame,
    mode: str,
    layer1_name: str,
    layer2_name: str,
    weight1: Optional[str] = None,
    weight2: Optional[str] = None,
    id_col: str = 'id_supply',
) -> gpd.GeoDataFrame:
    """
    Transfer cost columns (starting with 'cost_') from layer1 to layer2 based on nearest assignment.
    In convergence mode, weight used is weight1 / weight2.
    """
    cost_cols = [c for c in layer1.columns if c.startswith('cost_')]
    if not cost_cols:
        return layer2.copy()

    result = layer2.copy()
    if result.empty:
        for c in cost_cols:
            result[f'{c}'] = np.nan
        return result

    if mode == 'convergence':
        if weight1 is None or weight2 is None:
            raise ValueError("mode='convergence' requires weight1 and weight2")
        if weight1 not in layer1.columns:
            raise ValueError(f"weight1 column '{weight1}' not found in layer1")
        if weight2 not in layer2.columns:
            raise ValueError(f"weight2 column '{weight2}' not found in layer2")
        
        nearest = f"nearest_{layer2_name}_id"
        if nearest not in layer1.columns:
            raise ValueError(f"Layer1 missing {nearest}")
        
        # Prepare data
        temp = layer1[[nearest, weight1] + cost_cols].copy().rename(columns={nearest: 'dest_id'})
        
        # Add weight2 based on destination
        w2_mapping = pd.Series(
            pd.to_numeric(layer2[weight2], errors='coerce').values,
            index=layer2[id_col].astype(str).values
        )
        temp['w2'] = temp['dest_id'].astype(str).map(w2_mapping)
        
        # Numeric conversion
        for col in [weight1] + cost_cols:
            temp[col] = pd.to_numeric(temp[col], errors='coerce')
        
        # Calculate weight1/weight2 ratio
        temp['w'] = temp[weight1] / temp['w2']
        
        temp = temp.dropna(subset=['dest_id', 'w'] + cost_cols)
        temp = temp[temp['w'] > 0]
        
        if temp.empty:
            for c in cost_cols:
                result[f'{c}'] = np.nan
            return result
        
        for c in cost_cols:
            temp[f'wsum_{c}'] = temp[c] * temp['w']
            agg = temp.groupby('dest_id').agg(wsum=(f'wsum_{c}', 'sum'), wtotal=('w', 'sum'))
            result[f'{c}'] = result[id_col].astype(str).map((agg['wsum'] / agg['wtotal']))
        return result

    # divergence
    nearest = f"nearest_{layer1_name}_id"
    if nearest not in result.columns:
        raise ValueError(f"Layer2 missing {nearest}")
    if id_col not in layer1.columns:
        raise ValueError(f"Layer1 missing {id_col}")
    map_df = layer1[[id_col] + cost_cols].set_index(id_col).apply(pd.to_numeric, errors='coerce')
    map_df.index = map_df.index.astype(str)
    result[nearest] = result[nearest].astype(str)
    for c in cost_cols:
        result[f'{c}'] = result[nearest].map(map_df[c])
    return result

def propagate(
    layer1: gpd.GeoDataFrame,
    allocation_matrix: np.ndarray,
    exceptions: Optional[List[str]] = None,
) -> gpd.GeoDataFrame:
    """
    Propagate costs among storage facilities using the allocation matrix,
    then compute final LPG price.

    Automatically detects all columns starting with 'cost_' as source costs,
    excluding those specified in exceptions. For each cost column, creates a
    weighted average after rebalancing based on allocation matrix.

    Parameters
    ----------
    layer1 : gpd.GeoDataFrame
        GeoDataFrame with columns starting with 'cost_' from upstream aggregation.
        Must have the same number of rows as allocation_matrix.
    allocation_matrix : np.ndarray
        Square matrix (n x n) where element [i, j] is the percentage of product
        sent from storage i to storage j.
    exceptions : List[str], optional
        Column names to exclude from propagation. These columns are simply
        passed through or initialized to zero. Defaults to None.

    Returns
    -------
    gpd.GeoDataFrame
        Storage GeoDataFrame with propagated costs and final 'cost_source' column.
    """
    ps = layer1.copy()
    n = len(ps)
    if allocation_matrix.shape != (n, n):
        raise ValueError(f"allocation_matrix shape {allocation_matrix.shape} != ({n},{n})")

    # Check if cost_source was in the original input (to avoid recalculating if already propagated)
    cost_source_in_input = 'cost_source' in layer1.columns

    # Handle exceptions
    if exceptions is None:
        exceptions = []

    # Automatically detect cost columns (excluding exceptions)
    all_cost_cols = [col for col in ps.columns if col.startswith('cost_')]
    if not all_cost_cols:
        raise ValueError("No columns starting with 'cost_' found.")
    
    # Cost components to propagate (all cost columns except exceptions)
    cost_components = [col for col in all_cost_cols if col not in exceptions]

    # Propagate: for each destination, weighted average of costs from senders
    for j in range(n):
        weights = allocation_matrix[:, j]
        senders = np.where(weights > 0)[0]
        if len(senders) == 0:
            continue
        w_sum = weights[senders].sum()
        if w_sum <= 0:
            continue
        for comp in cost_components:
            if comp in ps.columns:
                vals = ps.iloc[senders][comp].to_numpy(dtype=float)
                ps.loc[ps.index[j], comp] = float(np.nansum(vals * weights[senders]) / w_sum)

    # Handle exception columns (pass through or initialize to zero)
    for exc in exceptions:
        if exc not in ps.columns:
            ps[exc] = 0.0
        else:
            ps[exc] = pd.to_numeric(ps[exc], errors='coerce').fillna(0.0)

    return ps

def total(gdf: gpd.GeoDataFrame, output_col: str = 'cost_total') -> gpd.GeoDataFrame:
    """
    Sum all columns starting with 'cost_' and create a new column with the total.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame containing cost columns.
    output_col : str, default 'cost_total'
        Name of the new column to store the sum.

    Returns
    -------
    gpd.GeoDataFrame
        A copy of the input GeoDataFrame with the additional output column.
    """
    df = gdf.copy()
    # Select cost columns
    cost_cols = [col for col in df.columns if col.startswith('cost_')]
    if output_col in cost_cols:
        cost_cols = [col for col in cost_cols if col != output_col]
    if not cost_cols:
        raise ValueError("No columns starting with 'cost_' found.")
    # Sum across rows, fill NaN with 0
    df[output_col] = df[cost_cols].fillna(0).sum(axis=1)
    return df

# Mattia part

def filling(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculates the internal operating costs of the filling plant and adds it as 'cost_fil_plants'.
    """
    df = gdf.copy()
    
    # 1. Determine inbound cost (accumulated total cost up to this point)
    cost_in = pd.to_numeric(df.get('cost_total', 0.0), errors='coerce').fillna(0.0)
    
    # 2. Compute Annualized Fixed Costs
    crf_op = crf(discount_rate, op_license_validity_years_fill)
    crf_fill = crf(discount_rate, lifetime_filling)
    
    annual_op_license = op_license_fee_usd_fill * crf_op
    annual_cap_cost = overnight_filling_usd * crf_fill
    
    total_fixed_annual = (fom_annual_usd_fill + annual_op_license + 
                          (annual_cap_cost * filling_allocation_fraction))
    
    fixed_cost_per_kg = total_fixed_annual / reference_filling_capacity_kg
    added_cost_per_kg = fixed_cost_per_kg + vom_usd_per_kg_fill
    
    # 3. Apply operational loss factor to inbound cost
    effective_in_cost = cost_in / (1.0 - loss_factor_fill)
    
    # 4. Outbound cost and markup cap
    cost_out_uncapped = effective_in_cost + added_cost_per_kg
    max_cost_out = cost_in * (1.0 + cost_increase_cap_pct)
    
    cost_out = np.where(cost_in > 0, np.minimum(cost_out_uncapped, max_cost_out), 0.0)
    
    # 5. Store specific filling plant cost component
    df['cost_fil_plants'] = np.maximum(0.0, cost_out - cost_in)
    
    return df


def reseller(resell_gdf: gpd.GeoDataFrame, filling_gdf: gpd.GeoDataFrame, income_raster_path: str, pop_raster_path: str, urban_raster_path: str) -> gpd.GeoDataFrame:
    """
    Calculates the internal reseller shop cost (cost_res_shop) based on local income, 
    estimated minimum wage, and system-wide average demand.
    """
    resell = resell_gdf.copy()
    filling = filling_gdf.copy()
    
    # 1. System average demand calculation
    assigned_col = next((c for c in ['assigned_fil_clients', 'total_fil_clients'] if c in filling.columns), None)
    if assigned_col == 'assigned_fil_clients':
        total_assigned_clients = pd.to_numeric(filling['assigned_fil_clients'], errors='coerce').fillna(0).sum()
    elif assigned_col == 'total_fil_clients':
        total_assigned_clients = (pd.to_numeric(filling['total_fil_clients'], errors='coerce').fillna(0).sum() - 
                                  pd.to_numeric(filling.get('clients', 0), errors='coerce').fillna(0).sum())
    else:
        total_assigned_clients = 0.0
        
    n_resellers = len(resell)
    if n_resellers == 0:
        resell['cost_res_shop'] = 0.0
        return resell
        
    avg_reseller_clients = max(0, total_assigned_clients) / n_resellers
    
    annual_kg_client = (meals_per_day * 365.0 * energy_per_meal_mj) / (efficiency * energy_content_mj_per_kg)
    avg_reseller_demand_kg = avg_reseller_clients * annual_kg_client
    
    if avg_reseller_demand_kg <= 0:
        resell['cost_res_shop'] = 0.0
        return resell

    # 2. Extract local income per capita using 2x2 raster window (vectorized)
    resell_income_monthly = np.full(n_resellers, np.nan)

    if (os.path.exists(income_raster_path)
            and os.path.exists(urban_raster_path)):
        with rasterio.open(income_raster_path) as inc_src, \
            rasterio.open(urban_raster_path) as urb_src:

            work_gdf = resell.to_crs(inc_src.crs)
            xs = np.array([g.x if (g and not g.is_empty) else np.nan
                        for g in work_gdf.geometry], dtype=np.float64)
            ys = np.array([g.y if (g and not g.is_empty) else np.nan
                        for g in work_gdf.geometry], dtype=np.float64)
            valid_geom = np.isfinite(xs) & np.isfinite(ys)

            if valid_geom.any():
                # Convert all coordinates to (row, col) in one vectorised call
                rows, cols = rowcol(inc_src.transform, xs[valid_geom], ys[valid_geom])
                rows = rows.astype(np.int64)
                cols = cols.astype(np.int64)

                H, W = inc_src.height, inc_src.width

                # Read the full raster once — much faster than N windowed reads
                inc_full = inc_src.read(1).astype(np.float64)
                urb_full = urb_src.read(1).astype(np.float64)

                # For each reseller, average the valid 2x2 neighbourhood
                inc_vals = np.full(valid_geom.sum(), np.nan)
                for k, (r, c) in enumerate(zip(rows, cols)):
                    r0, r1 = max(r - 1, 0), min(r + 1, H)   # clip to array bounds
                    c0, c1 = max(c - 1, 0), min(c + 1, W)
                    patch_inc = inc_full[r0:r1+1, c0:c1+1]
                    patch_urb = urb_full[r0:r1+1, c0:c1+1]
                    valid = np.isfinite(patch_inc) & (patch_inc > 0)
                    if valid.any():
                        inc_vals[k] = patch_inc[valid].mean()

                resell_income_monthly[valid_geom] = inc_vals
                    
    # 3. Apply minimum wage floor
    income_monthly_used = np.where(
        np.isfinite(resell_income_monthly) & (resell_income_monthly > minimum_wage_usd_per_month),
        resell_income_monthly,
        minimum_wage_usd_per_month
    )
    
    # 4. Fixed and Variable Retail Costs
    annual_salary_usd = income_monthly_used * 12.0
    total_fixed_annual = ((rent_per_month_usd_res * 12.0 + annual_salary_usd) * retail_allocation_fraction + license_annual_usd_res)
    
    fixed_cost_per_kg = total_fixed_annual / avg_reseller_demand_kg
    added_cost_per_kg = fixed_cost_per_kg + vom_usd_per_kg_res
    
    # 5. Determine Reseller Cost component and cap
    cost_in = pd.to_numeric(resell.get('cost_total', 0.0), errors='coerce').fillna(0.0)
    
    cost_out_uncapped = cost_in + added_cost_per_kg
    max_cost_out = cost_in * (1.0 + cost_increase_cap_pct)
    
    cost_out = np.where(cost_in > 0, np.minimum(cost_out_uncapped, max_cost_out), 0.0)
    
    resell['cost_res_shop'] = np.maximum(0.0, cost_out - cost_in)
    
    return resell

def _compute_spatial_vot(income_array: np.ndarray) -> np.ndarray:
    """Computes the spatial Value of Time (VOT) based on local income."""
    valid = np.isfinite(income_array)
    vot = np.full(income_array.shape, default_vot_usd_per_hour, dtype=np.float32)
    
    if not np.any(valid):
        return vot
        
    min_value = float(np.nanmin(income_array[valid]))
    max_value = float(np.nanmax(income_array[valid]))
    
    if not np.isfinite(min_value) or not np.isfinite(max_value) or max_value <= min_value:
        return vot
        
    wage_min, wage_max = wage_range
    norm = (income_array - min_value) / (max_value - min_value)
    norm = np.clip(norm, 0.0, 1.0)
    
    wage_factor = wage_min + norm * float(wage_max - wage_min)
    vot_valid = wage_factor * (minimum_wage_usd_per_month / work_hours_per_month)
    vot[valid] = vot_valid[valid].astype(np.float32)
    
    return vot

def _build_cost_maps(points: dict):
    """Builds reference cost dictionaries directly from the current in-memory points layer."""
    resell = points['reseller_points']
    filling = points['filling_points']
    
    map_resell_cost = dict(zip(
        pd.to_numeric(resell['id_res&fil'], errors='coerce').dropna().astype(np.int64), 
        pd.to_numeric(resell['cost_total'], errors='coerce').dropna().astype(float)
    ))
    
    map_fill_cost = dict(zip(
        pd.to_numeric(filling['id_res&fil'], errors='coerce').dropna().astype(np.int64), 
        pd.to_numeric(filling['cost_total'], errors='coerce').dropna().astype(float)
    ))
    
    fill_ref = resell.get('filling_reference', resell['id_res&fil'])
    map_reseller_to_filling = dict(zip(
        pd.to_numeric(resell['id_res&fil'], errors='coerce').dropna().astype(np.int64),
        pd.to_numeric(fill_ref, errors='coerce').dropna().astype(np.int64)
    ))
    
    return map_resell_cost, map_fill_cost, map_reseller_to_filling


def _map_effective_cost(ids_int: np.ndarray, map_resell: dict, map_fill: dict) -> np.ndarray:
    """Maps the effective outlet cost to the pixel grids."""
    out = np.full(ids_int.shape, np.nan, dtype=np.float64)
    valid = ids_int > 0
    if not np.any(valid):
        return out
        
    uniq, inv = np.unique(ids_int[valid], return_inverse=True)
    mapped = np.array([map_resell.get(int(rid), map_fill.get(int(rid), np.nan)) for rid in uniq], dtype=np.float64)
    out[valid] = mapped[inv]
    
    return out

def end_user(points: dict, huff_raster_path: str|Path, income_raster_path: str|Path, 
             pop_raster_path: str|Path,urban_raster_path: str|Path,  lpg_share_path: str|Path, output_path: str|Path, 
             use_spatial_vot: bool = True):
    """
    Calculates unified end-user costs (walking and driving) and generates the 19-band final TIF.
    """
    # 1. Build dictionary mappings for quick cost lookup
    map_resell_cost, map_fill_cost, map_reseller_to_filling = _build_cost_maps(points)
    
    # 2. Read the full 8-band Huff allocation raster efficiently
    with rasterio.open(huff_raster_path) as src:
        car_share = src.read(1).astype(np.float32)
        walk_share = src.read(2).astype(np.float32)
        walk_id = src.read(3).astype(np.float32)
        walk_time = src.read(4).astype(np.float32)
        walk_dist = src.read(5).astype(np.float32)
        car_id = src.read(6).astype(np.float32)
        car_time = src.read(7).astype(np.float32)
        car_dist = src.read(8).astype(np.float32)
        profile = src.profile.copy()
        
    height, width = walk_share.shape
    n = height * width
    
    # 3. Read LPG Use Share
    try:
        lpg_use_share, _ = tool._read_single_band(lpg_share_path)
    except Exception as e:
        raise FileNotFoundError(f"Cannot read LPG use share raster: {lpg_share_path}") from e

    # 4. Compute Spatial VOT if enabled
    if use_spatial_vot and Path(income_raster_path).exists():
        _, pop_profile = tool._read_single_band(pop_raster_path)
        income_aligned = tool.reproject_to_reference(Path(income_raster_path), pop_profile, Resampling.bilinear)
        urban_aligned = tool.reproject_to_reference(
            Path(urban_raster_path), pop_profile, Resampling.nearest  
        )
        income_per_capita = tool.income_household_to_per_capita(        
            income_aligned, urban_aligned
        )
        vot_raster = _compute_spatial_vot(income_per_capita) 
    else:
        vot_raster = np.full((height, width), default_vot_usd_per_hour, dtype=np.float32)
        
    vot_flat = vot_raster.reshape(-1).astype(np.float64)

    # =========================================================
    # A. WALK COST CALCULATION
    # =========================================================
    walk_id_flat = walk_id.reshape(-1).astype(np.float64)
    walk_time_flat = walk_time.reshape(-1).astype(np.float64)
    
    with np.errstate(invalid='ignore'):
        walk_id_int = np.where(np.isfinite(walk_id_flat) & (walk_id_flat > 0), walk_id_flat.astype(np.int64), -1)
        
    ref_cost_walk_arr = _map_effective_cost(walk_id_int, map_resell_cost, map_fill_cost)
    
    filling_id_walk_arr = np.full(n, -1, dtype=np.int32)
    filling_cost_walk_arr = np.full(n, np.nan, dtype=np.float64)
    valid_walk_id = (walk_id_int > 0)
    
    if np.any(valid_walk_id):
        uniq_ids, inv = np.unique(walk_id_int[valid_walk_id], return_inverse=True)
        fid_mapped = np.array([map_reseller_to_filling.get(int(rid), int(rid)) for rid in uniq_ids], dtype=np.int32)
        fcost_mapped = np.array([map_fill_cost.get(int(fid), np.nan) for fid in fid_mapped], dtype=np.float64)
        filling_id_walk_arr[valid_walk_id] = fid_mapped[inv]
        filling_cost_walk_arr[valid_walk_id] = fcost_mapped[inv]
        
    # Strictly one-way time scaled to round-trip
    walk_round_trip_hours = (2.0 * walk_time_flat) / 60.0
    total_trip_time_hours_walk = walk_round_trip_hours + fixed_time_at_retailer_hours
    collection_cost_per_kg_walk = (total_trip_time_hours_walk * vot_flat) / cylinder_weight_kg
    
    valid_walk_mask = (np.isfinite(ref_cost_walk_arr) & (walk_id_int > 0) & 
                       np.isfinite(walk_round_trip_hours) & (walk_round_trip_hours >= 0) & np.isfinite(vot_flat))
                       
    cost_walk_final = np.full(n, np.nan, dtype=np.float64)
    cost_walk_final[valid_walk_mask] = ref_cost_walk_arr[valid_walk_mask] + collection_cost_per_kg_walk[valid_walk_mask]

    # =========================================================
    # B. CAR COST CALCULATION
    # =========================================================
    car_id_flat = car_id.reshape(-1).astype(np.float64)
    car_time_flat = car_time.reshape(-1).astype(np.float64)
    car_dist_flat = car_dist.reshape(-1).astype(np.float64)
    
    with np.errstate(invalid='ignore'):
        car_id_int = np.where(np.isfinite(car_id_flat) & (car_id_flat > 0), car_id_flat.astype(np.int64), -1)
        
    ref_cost_car_arr = _map_effective_cost(car_id_int, map_resell_cost, map_fill_cost)
    
    filling_id_car_arr = np.full(n, -1, dtype=np.int32)
    filling_cost_car_arr = np.full(n, np.nan, dtype=np.float64)
    valid_car_id = (car_id_int > 0)
    
    if np.any(valid_car_id):
        uniq_ids, inv = np.unique(car_id_int[valid_car_id], return_inverse=True)
        fid_mapped = np.array([map_reseller_to_filling.get(int(rid), int(rid)) for rid in uniq_ids], dtype=np.int32)
        fcost_mapped = np.array([map_fill_cost.get(int(fid), np.nan) for fid in fid_mapped], dtype=np.float64)
        filling_id_car_arr[valid_car_id] = fid_mapped[inv]
        filling_cost_car_arr[valid_car_id] = fcost_mapped[inv]
        
    round_trip_distance_km_car = 2.0 * car_dist_flat
    round_trip_drive_hours = (car_time_flat * 2.0) / 60.0
    total_trip_time_hours_car = round_trip_drive_hours + fixed_time_at_retailer_hours
    
    vehicle_operating_cost_trip_usd = round_trip_distance_km_car * car_variable_cost_per_km
    driver_time_cost_trip_usd = total_trip_time_hours_car * vot_flat
    collection_cost_per_kg_car = (vehicle_operating_cost_trip_usd + driver_time_cost_trip_usd) / cylinder_weight_kg
    
    valid_driver_mask = (np.isfinite(ref_cost_car_arr) & (car_id_int > 0) & 
                         np.isfinite(car_time_flat) & (car_time_flat >= 0) & 
                         np.isfinite(car_dist_flat) & (car_dist_flat >= 0) & 
                         np.isfinite(total_trip_time_hours_car) & np.isfinite(vot_flat))
                         
    cost_car_final = np.full(n, np.nan, dtype=np.float64)
    cost_car_final[valid_driver_mask] = ref_cost_car_arr[valid_driver_mask] + collection_cost_per_kg_car[valid_driver_mask]

    # =========================================================
    # C. MEAN & MAJORITY AGGREGATIONS (modified)
    # =========================================================
    share_walk_arr = np.clip(walk_share.reshape(-1).astype(np.float64), 0.0, 1.0)
    share_car_arr = np.clip(car_share.reshape(-1).astype(np.float64), 0.0, 1.0)
    share_sum = share_walk_arr + share_car_arr
    share_valid = share_sum > 0

    share_walk_norm = np.zeros(n, dtype=np.float64)
    share_car_norm = np.zeros(n, dtype=np.float64)
    share_walk_norm[share_valid] = share_walk_arr[share_valid] / share_sum[share_valid]
    share_car_norm[share_valid] = share_car_arr[share_valid] / share_sum[share_valid]

    # --- NEW: only consider pixels where BOTH walk and car costs exist ---
    both_costs_valid = np.isfinite(cost_walk_final) & np.isfinite(cost_car_final) & share_valid

    mean_user_cost_flat = np.full(n, np.nan, dtype=np.float64)
    if np.any(both_costs_valid):
        total_weight = share_walk_norm[both_costs_valid] + share_car_norm[both_costs_valid]  # always 1 if shares valid
        mean_user_cost_flat[both_costs_valid] = (
            cost_walk_final[both_costs_valid] * share_walk_norm[both_costs_valid] +
            cost_car_final[both_costs_valid]  * share_car_norm[both_costs_valid]
        ) / total_weight

    # Majority cost stays unchanged (or you could apply the same condition if desired)
    majority_cost_flat = np.full(n, np.nan, dtype=np.float64)
    walk_majority = (share_walk_norm > share_car_norm)
    majority_cost_flat[walk_majority] = cost_walk_final[walk_majority]
    car_majority = ~walk_majority & (share_walk_norm + share_car_norm > 0)
    majority_cost_flat[car_majority] = cost_car_final[car_majority]

    # =========================================================
    # D. EXPORT TO 19-BAND TIF
    # =========================================================
    out_bands = [
        car_share, walk_share,
        np.where(np.isfinite(walk_id) & (walk_id >= 0), walk_id, np.nan),
        walk_time, walk_dist,
        np.where(np.isfinite(car_id) & (car_id >= 0), car_id, np.nan),
        car_time, car_dist,
        ref_cost_walk_arr.reshape(height, width).astype(np.float32),
        ref_cost_car_arr.reshape(height, width).astype(np.float32),
        cost_walk_final.reshape(height, width).astype(np.float32),
        cost_car_final.reshape(height, width).astype(np.float32),
        filling_id_walk_arr.reshape(height, width).astype(np.float32),
        filling_id_car_arr.reshape(height, width).astype(np.float32),
        filling_cost_walk_arr.reshape(height, width).astype(np.float32),
        filling_cost_car_arr.reshape(height, width).astype(np.float32),
        lpg_use_share.astype(np.float32),
        majority_cost_flat.reshape(height, width).astype(np.float32),
        mean_user_cost_flat.reshape(height, width).astype(np.float32)
    ]
    
    out_names = [
        "car_share", "walk_share", "best_reseller_id_walk", "best_time_walk_min", "best_distance_walk_km",
        "best_reseller_id_car", "best_time_car_min", "best_distance_car_km",
        "res_cost_kg_out_walk_ref", "res_cost_kg_out_car_ref",
        "cost_kg_walker", "cost_kg_driver",
        "filling_id_walk", "filling_id_car",
        "cost_fil_kg_out_walk_ref", "cost_fil_kg_out_car_ref",
        "lpg_use_share", "majority_cost_kg", "mean_user_cost"
    ]
    
    tool.write_multiband_raster(Path(output_path), profile, out_bands, out_names)
    
    print(f"✓ 19-Band End-User Price TIF generated successfully: {output_path}")
    print(f"  - Valid Walker pixels: {valid_walk_mask.sum():,}")
    print(f"  - Valid Driver pixels: {valid_driver_mask.sum():,}")