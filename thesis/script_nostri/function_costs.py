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
import function_allocation as alloc
import function_process as process
from dataclasses import dataclass

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
    gdf['cost_import_sea'] = gdf['LPG_price'] * import_sea_rate

    # 2. STS Cost: Applied if storage is available (LPG_compliance) but port is NOT VLGC compliant
    mask_sts = (gdf['LPG_compliance'] == True) & (gdf['VLGC_compliance'] == False)
    gdf.loc[mask_sts, 'cost_sts'] = STS_cost

    # 3. Pre-bottling Cost: Applied only if no nearby storage is available
    # It is calculated on the price already including sea import costs
    mask_no_storage = (gdf['LPG_compliance'] == False)
    price_after_import = gdf['LPG_price'] + gdf['cost_import_sea']
    gdf.loc[mask_no_storage, 'cost_pre_bottling'] = price_after_import * pre_bottling_cost

    # Final Price Calculation: Sum of base price and all extra components
    gdf['LPG_price'] = (
        gdf['LPG_price'] + 
        gdf['cost_import_sea'] + 
        gdf['cost_sts'] + 
        gdf['cost_pre_bottling']
    )

    # Check for remaining missing prices
    missing_prices = int(gdf['LPG_price'].isna().sum())
    if missing_prices > 0:
        print(f"Warning: {missing_prices} port rows have missing LPG_price after cost calculation")

    return gdf

def border_points(gdf, import_cost_df, vehicle: Vehicle):
    """
    Calculates Land Import costs, Border Time costs, and Ferry costs.
    Updates final LPG_price and stores individual components as attributes.
    """
    bp = gdf.copy()
    
    # Initialize cost attribute columns
    bp['cost_import_land'] = 0.0
    bp['cost_border_wait'] = 0.0
    bp['cost_ferry'] = 0.0

    # 1. Land Import Cost (Percentage markup on base price)
    import_land_rate = float(import_cost_df.loc['import_land'].iloc[0])
    bp['cost_import_land'] = bp['LPG_price'] * import_land_rate

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

    # Update final price
    bp['LPG_price'] = (
        bp['LPG_price'] + 
        bp['cost_import_land'] + 
        bp['cost_border_wait'] + 
        bp['cost_ferry']
    )

    return bp

def crf(rate: float, years: int) -> float:
    """
    Compute the capital recovery factor.
    Used to annualize upfront costs over an asset lifetime.
    Returns a dimensionless multiplier.
    """
    return (rate * (1.0 + rate) ** years) / ((1.0 + rate) ** years - 1.0)


# Calculate fixed cost inputs used by the cost per kg functions.
# This makes the downstream functions depend only on distance/time.


driver_hourly_cost_usd = (
    driver_annual_salary_usd * salary_multiplier / (hours_per_day * days_per_year)
)

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


def transport_to_storage(points_dict, vehicle: Vehicle):  # TODO delete if replaced by trasport()
    """
    Calculate transport cost from each facility to its assigned storage.
    
    This function computes the cost_transport_to_storage for all facility types
    (refineries, ports, gas_plants, border_points) based on their distance and 
    travel time to assigned storage facilities.
    """
    facility_keys = ["refineries", "ports", "gas_plants", "border_points"]
    
    # calculate avg_one_way_dist_km 
    avg_one_way_dist_km = average_transport_distance(points_dict)

    # Validate avg_one_way_dist_km
    if not np.isfinite(avg_one_way_dist_km) or avg_one_way_dist_km <= 0:
        print(f"⚠ Warning: Invalid avg_one_way_dist_km = {avg_one_way_dist_km}. All costs set to NaN")
        for key in facility_keys:
            gdf = points_dict.get(key)
            if gdf is not None:
                gdf['cost_transport_to_storage'] = np.nan
                points_dict[key] = gdf
        return points_dict
    
    # Calculate transport cost for each facility
    for key in facility_keys:
        gdf = points_dict.get(key)
        if gdf is None:
            continue
        
        # Check for required columns
        if 'tank_distance' not in gdf.columns or 'tank_traveltime' not in gdf.columns:
            gdf['cost_transport_to_storage'] = np.nan
            points_dict[key] = gdf
            continue
        
        # Extract distance and time arrays
        d_km = pd.to_numeric(gdf['tank_distance'], errors='coerce').to_numpy(dtype=float)
        t_min = pd.to_numeric(gdf['tank_traveltime'], errors='coerce').to_numpy(dtype=float)
        out_cost = np.full(len(gdf), np.nan, dtype=float)
        
        # Identify valid rows (all values are finite and positive)
        valid = np.isfinite(d_km) & (d_km > 0) & np.isfinite(t_min) & (t_min >= 0)
        valid_idx = np.where(valid)[0]
        
        # Compute tanker cost for each valid row
        for i in valid_idx:
            out_cost[i] = single_travel(
                one_way_distance_km=float(d_km[i]),
                one_way_time_min=float(t_min[i]),
                avg_one_way_dist_km=avg_one_way_dist_km,
                vehicle=vehicle
            )['total_cost_per_kg']
        
        gdf['cost_transport_to_storage'] = out_cost
        points_dict[key] = gdf
    
    return points_dict


def aggregate_transport_costs_to_storage(points_dict):   # TODO replace if transport() is used
    """
    Aggregate facility-level transport costs to primary storage using weighted average.
    
    This function collects cost_transport_to_storage from all facility types,
    applies a weighted average by facility percentage, and merges results back
    to the primary_storage GeoDataFrame.
   
    """
    facility_keys = ["refineries", "ports", "gas_plants", "border_points"]
    
    # Initialize primary storage
    primary_storage = points_dict["primary_storage"].copy()
    if 'name' not in primary_storage.columns:
        primary_storage['name'] = [f'primary_storage_{i}' for i in range(len(primary_storage))]
    
    # Gather facility data
    weighted_df = pd.concat([
        points_dict[k][['tank_name', 'cost_transport_to_storage', 'percentage']]
        for k in facility_keys if points_dict.get(k) is not None
    ], ignore_index=True)
    
    # Conversions and filtering
    weighted_df = weighted_df.assign(
        cost_transport_to_storage=lambda x: pd.to_numeric(x['cost_transport_to_storage'], errors='coerce'),
        percentage=lambda x: pd.to_numeric(x['percentage'], errors='coerce').fillna(0)
    ).dropna(subset=['cost_transport_to_storage'])
    weighted_df = weighted_df[weighted_df['percentage'] > 0]
    
    # Calculate weighted average
    if len(weighted_df) > 0:
        weighted_df = weighted_df.assign(
            weighted=lambda x: x['cost_transport_to_storage'] * x['percentage']
        )
        
        agg = weighted_df.groupby('tank_name', as_index=False).agg(
            weighted_cost_sum=('weighted', 'sum'),
            weight_sum=('percentage', 'sum')
        )
        
        agg['cost_transport_to_storage'] = agg['weighted_cost_sum'] / agg['weight_sum']
        
        # Merge back to primary storage
        primary_storage = primary_storage.merge(
            agg[['tank_name', 'cost_transport_to_storage']], 
            left_on='name', 
            right_on='tank_name', 
            how='left'
        ).drop('tank_name', errors='ignore')
    else:
        primary_storage['cost_transport_to_storage'] = np.nan
    
    # Update points dict
    points_dict['primary_storage'] = primary_storage
    return points_dict

from typing import Union, Optional
import numpy as np
import pandas as pd
import geopandas as gpd

def transport(
    layer1: gpd.GeoDataFrame,
    layer2: gpd.GeoDataFrame,
    vehicle: Vehicle,
    mode: str,  # 'convergence', 'cross_exchange', 'divergence'
    layer1_name: Optional[str] = None,
    layer2_name: Optional[str] = None,
    weight: Optional[str] = None,
    distance_matrix: Optional[np.ndarray] = None,
    time_matrix: Optional[np.ndarray] = None,
    weight_matrix: Optional[np.ndarray] = None,
    id_col: str = 'id',
    output_col: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Compute transport costs and optionally aggregate.

    Modes:
        convergence (ex nearest): layer1 has nearest_{layer2_name}_distance/time/id + weight.
            Returns layer2 with weighted average cost per destination.
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
        if not layer1_name or not weight:
            raise ValueError("mode 'convergence' requires layer1_name and weight")
        costs = _cost_series(layer1, layer2_name, weight)
        temp = pd.DataFrame({
            'dest': layer1[f"nearest_{layer2_name}_id"].astype(str),
            'w': layer1[weight].astype(float),
            'c': costs
        }).dropna()
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
    weight: Optional[str] = None,
    id_col: str = 'id_supply',
) -> gpd.GeoDataFrame:
    """
    Transfer cost columns (starting with 'cost_') from layer1 to layer2 based on nearest assignment.
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
        if weight is None:
            raise ValueError("mode='convergence' requires weight")
        nearest = f"nearest_{layer2_name}_id"
        if nearest not in layer1.columns:
            raise ValueError(f"Layer1 missing {nearest}")
        temp = layer1[[nearest, weight] + cost_cols].copy().rename(columns={nearest: 'dest_id'})
        for col in [weight] + cost_cols:
            temp[col] = pd.to_numeric(temp[col], errors='coerce')
        temp = temp.dropna(subset=['dest_id', weight] + cost_cols)
        temp = temp[temp[weight] > 0]
        if temp.empty:
            for c in cost_cols:
                result[f'origin_{c}'] = np.nan
            return result
        for c in cost_cols:
            temp[f'wsum_{c}'] = temp[c] * temp[weight]
            agg = temp.groupby('dest_id').agg(wsum=(f'wsum_{c}', 'sum'), wtotal=(weight, 'sum'))
            result[f'origin_{c}'] = result[id_col].astype(str).map((agg['wsum'] / agg['wtotal']))
        return result

    # divergence
    nearest = f"nearest_{layer1_name}_id"
    if nearest not in result.columns:
        raise ValueError(f"Layer2 missing {nearest}")
    if id_col not in layer1.columns:
        raise ValueError(f"Layer1 missing {id_col}")
    map_df = layer1[[id_col] + cost_cols].set_index(id_col).apply(pd.to_numeric, errors='coerce')
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

    Automatically detects all columns starting with 'origin_' as the source costs
    (e.g., origin_cost_import_sea). For each such column, creates a corresponding
    column without the 'origin_' prefix (e.g., cost_import_sea) containing the
    weighted average after rebalancing.

    Parameters
    ----------
    layer1 : gpd.GeoDataFrame
        GeoDataFrame with columns 'origin_{cost_name}' from upstream aggregation.
        Must have the same number of rows as allocation_matrix.
    allocation_matrix : np.ndarray
        Square matrix (n x n) where element [i, j] is the percentage of product
        sent from storage i to storage j.
    exceptions : List[str], optional
        Additional column names in layer1 to add to the final price. These are not
        propagated via allocation; they are simply summed.

    Returns
    -------
    gpd.GeoDataFrame
        Storage GeoDataFrame with:
        - Allocated cost columns (without 'origin_' prefix)
        - Final 'LPG_price' column
    """
    ps = layer1.copy()
    n = len(ps)
    if allocation_matrix.shape != (n, n):
        raise ValueError(f"allocation_matrix shape {allocation_matrix.shape} != ({n},{n})")

    # Automatically detect origin cost columns
    origin_cols = [col for col in ps.columns if col.startswith('origin_')]
    if not origin_cols:
        raise ValueError("No columns starting with 'origin_' found. Run tracking with mode='convergence' first.")

    # Derive cost component names (strip 'origin_' prefix)
    cost_components = [col[7:] for col in origin_cols]  # len('origin_') = 7

    # Initialize allocated cost columns
    for comp in cost_components:
        ps[comp] = np.nan

    # Propagate: for each destination, weighted average of origin costs from senders
    for j in range(n):
        weights = allocation_matrix[:, j]
        senders = np.where(weights > 0)[0]
        if len(senders) == 0:
            continue
        w_sum = weights[senders].sum()
        if w_sum <= 0:
            continue
        for comp in cost_components:
            origin_vals = ps.iloc[senders][f'origin_{comp}'].to_numpy(dtype=float)
            ps.loc[ps.index[j], comp] = float(np.nansum(origin_vals * weights[senders]) / w_sum)

    # Handle exception columns (e.g., cost_storage_second, cost_rebalancing)
    if exceptions is None:
        exceptions = []
    for exc in exceptions:
        if exc not in ps.columns:
            ps[exc] = 0.0
        else:
            ps[exc] = pd.to_numeric(ps[exc], errors='coerce').fillna(0.0)

    # Compute final LPG price
    cost_cols = cost_components + exceptions
    ps['LPG_price'] = ps[cost_cols].sum(axis=1, min_count=1).fillna(0.0)

    return ps

def total(gdf: gpd.GeoDataFrame, output_col: str = 'LPG_price') -> gpd.GeoDataFrame:
    """
    Sum all columns starting with 'cost_' and create a new column with the total.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame containing cost columns.
    output_col : str, default 'LPG_price'
        Name of the new column to store the sum.

    Returns
    -------
    gpd.GeoDataFrame
        A copy of the input GeoDataFrame with the additional output column.
    """
    df = gdf.copy()
    # Select cost columns
    cost_cols = [col for col in df.columns if col.startswith('cost_')]
    if not cost_cols:
        raise ValueError("No columns starting with 'cost_' found.")
    # Sum across rows, fill NaN with 0
    df[output_col] = df[cost_cols].fillna(0).sum(axis=1)
    return df