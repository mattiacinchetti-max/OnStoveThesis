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
import function_allocation as alloc
from dataclasses import dataclass

def fill_from_buffer(gdf, default_gdf, cols, buffer_km=BUFFER_KM):
    """
    Fill missing values by searching nearby features within a buffer radius.
    Uses a metric CRS to compute distances safely.
    Keeps original geometry and CRS on output.
    """
    gdf, default_gdf = gdf.copy(), default_gdf.to_crs(gdf.crs)
    if gdf.crs is None or default_gdf.crs is None:
        raise ValueError("Input GeoDataFrame has no CRS")
    gdf_metric, default_metric = tool._metric_views_for_buffer(gdf, default_gdf)
    buffer_dist = buffer_km * 1000.0

    for col in cols:
        if col not in gdf.columns:
            continue
        missing_idx = gdf[gdf[col].isna() | (gdf[col] == '')].index
        if len(missing_idx) == 0:
            continue

        for idx in missing_idx:
            geom = gdf_metric.loc[idx, 'geometry']
            if geom is None or geom.is_empty:
                continue
            port_buffer = geom.buffer(buffer_dist)
            nearby = default_metric[default_metric.geometry.intersects(port_buffer)]

            if len(nearby) > 0 and col in nearby.columns:
                val = nearby[col].dropna()
                if len(val) > 0:
                    gdf.at[idx, col] = val.iloc[0]
    return gdf


def fill_lpg_price_from_raster(gdf, raster_path, band_index=1, overwrite=False):
    """
    Sample a raster at facility locations and fill cost_source.
    Handles non-point geometries via centroids.
    Only fills missing values unless overwrite=True.
    """
    out = gdf.copy()

    if out.crs is None:
        raise ValueError("Input GeoDataFrame has no CRS")

    if 'cost_source' not in out.columns:
        out['cost_source'] = np.nan

    raster_file = Path(raster_path)
    if not raster_file.exists():
        raise FileNotFoundError(f"Missing LPG price raster: {raster_file}")

    with rasterio.open(raster_file) as src:
        if src.crs is None:
            raise ValueError("LPG price raster has no CRS")
        if band_index < 1 or band_index > src.count:
            raise ValueError(
                f"Requested band {band_index} for {raster_file.name}, but raster has {src.count} band(s)"
            )

        work = out.to_crs(src.crs).copy()
        geom_series = work.geometry
        if not work.geom_type.eq('Point').all():
            geom_series = work.geometry.centroid

        sampled_values = np.full(len(work), np.nan, dtype=np.float64)
        valid_positions = []
        valid_coords = []

        for pos, geom in enumerate(geom_series):
            if geom is None or geom.is_empty:
                continue
            valid_positions.append(pos)
            valid_coords.append((geom.x, geom.y))

        if len(valid_coords) > 0:
            for pos, sample in zip(valid_positions, src.sample(valid_coords, indexes=band_index)):
                sampled_values[pos] = float(sample[0])

        nodata = src.nodata
        if nodata is not None:
            sampled_values[sampled_values == nodata] = np.nan

    sampled_series = pd.Series(sampled_values, index=out.index)
    if overwrite:
        mask = sampled_series.notna()
    else:
        mask = out['cost_source'].isna() & sampled_series.notna()

    out.loc[mask, 'cost_source'] = sampled_series.loc[mask]
    return out

def redistribute_small_national_shares(nat_shares_df, min_share_pct=1.0):
    """
    Remove shares below a threshold and re-normalize to 100%.
    Preserves the original index and emits a sanity warning if needed.
    """
    if nat_shares_df.shape[1] == 0:
        raise ValueError("nat_shares_df has no numeric column")

    out = nat_shares_df.copy()
    share_col = out.columns[0]

    # Convert to numeric, handle missing values
    shares = pd.to_numeric(out[share_col], errors='coerce').fillna(0.0)

    # Identify which shares to keep
    keep = shares >= float(min_share_pct)

    # Set dropped shares to 0
    redistributed = shares.where(keep, 0.0)
    total_keep = float(redistributed.sum())

    # Redistribute proportionally to reach 100%
    redistributed = redistributed * (100.0 / total_keep)
    out[share_col] = redistributed
    
    return out

def refineries(gdf, default_gdf, lpg_price_raster_path, price_band=1):
    """
    Fill missing values and add LPG price from raster sampling.
    Ensures crude capacity is numeric and non-null.
    Applies buffer-based defaults before raster sampling.
    Returns a cleaned GeoDataFrame in the original CRS.
    """
    gdf = gdf.copy()

    # Phase 1: Fill from buffer
    fill_cols = ['crude_capacity', 'cost_source']
    gdf = fill_from_buffer(gdf, default_gdf, fill_cols)

    # Phase 2: Ensure production and fill cost_source from raster
    if 'crude_capacity' not in gdf.columns:
        gdf['crude_capacity'] = 0.0
    else:
        gdf['crude_capacity'] = pd.to_numeric(gdf['crude_capacity'], errors='coerce').fillna(0.0)

    gdf = fill_lpg_price_from_raster(gdf, lpg_price_raster_path, band_index=price_band, overwrite=False)

    missing_prices = int(gdf['cost_source'].isna().sum())
    if missing_prices > 0:
        print(f"Warning: {missing_prices} refinery rows still have missing cost_source after raster sampling")

    return gdf


def gas_plants(gdf, default_gdf, lpg_price_raster_path, price_band=1):
    """
    Fill missing values and add LPG price from raster sampling.
    Normalizes LPG production to numeric values.
    Applies buffer-based defaults before raster sampling.
    Returns a cleaned GeoDataFrame in the original CRS.
    """
    gdf = gdf.copy()

    # Phase 1: Fill from buffer
    fill_cols = ['LPG_prod', 'cost_source']
    gdf = fill_from_buffer(gdf, default_gdf, fill_cols)

    # Phase 2: Ensure production and fill cost_source from raster
    if 'LPG_prod' not in gdf.columns:
        gdf['LPG_prod'] = 0.0
    else:
        gdf['LPG_prod'] = pd.to_numeric(gdf['LPG_prod'], errors='coerce').fillna(0.0)

    gdf = fill_lpg_price_from_raster(gdf, lpg_price_raster_path, band_index=price_band, overwrite=False)

    missing_prices = int(gdf['cost_source'].isna().sum())
    if missing_prices > 0:
        print(f"Warning: {missing_prices} gas plant rows still have missing cost_source after raster sampling")

    return gdf


def ports(gdf, default_gdf = None, storage_gdf = None, lpg_price_raster_path = None, price_band=1, only_phase4 = False):
    """
    Handles data cleaning, spatial linking with storage facilities, 
    and sampling the base LPG price from the raster.
    """
    gdf = gdf.copy()

    if not only_phase4:
        if lpg_price_raster_path is None and default_gdf is None:
            raise ValueError("lpg_price_raster_path and default_gdf are required for ports when only_phase4=False")

        # Phase 1: Fill missing values from buffer
        fill_cols = ['VLGC_compliance', 'cost_source']
        gdf = fill_from_buffer(gdf, default_gdf, fill_cols)

        # Phase 2: Ensure cost_source exists and sample from raster
        if 'cost_source' not in gdf.columns:
            gdf['cost_source'] = np.nan
        gdf = fill_lpg_price_from_raster(gdf, lpg_price_raster_path, band_index=price_band, overwrite=False)

        # Phase 3: Initialize spatial and technical fields
        gdf['LPG_capacity'] = 0.0
        gdf['tanks_nearby'] = 0.0
        gdf['LPG_compliance'] = False

        # Ensure VLGC_compliance exists and handle NaNs
        if 'VLGC_compliance' not in gdf.columns:
            gdf['VLGC_compliance'] = False
        else:
            gdf['VLGC_compliance'] = gdf['VLGC_compliance'].fillna(False)

    # Phase 4: Spatial join with storage facilities
    if storage_gdf is not None:
        gdf = gdf.to_crs(storage_gdf.crs)
        gdf_metric, storage_metric = tool._metric_views_for_buffer(gdf, storage_gdf)
        buffer_dist = BUFFER_KM * 1000.0

        # Precompute buffers for all ports
        port_buffers = {idx: geom.buffer(buffer_dist) for idx, geom in gdf_metric.geometry.items()}

        # Count connections per storage to allow capacity de-duplication
        storage_connection_counts = {}
        for s_idx, s_row in storage_metric.iterrows():
            s_geom = s_row.geometry
            n_connected_ports = sum(1 for p_buf in port_buffers.values() if p_buf.intersects(s_geom))
            storage_connection_counts[s_idx] = n_connected_ports

        # Allocate capacity to ports based on nearby storage
        for idx, port_buffer in port_buffers.items():
            nearby = storage_metric[storage_metric.geometry.intersects(port_buffer)]
            if len(nearby) > 0:
                allocated_caps = []
                for s_idx, s_row in nearby.iterrows():
                    raw_cap = float(s_row['LPG_capacity']) if ('LPG_capacity' in nearby.columns and pd.notna(s_row.get('LPG_capacity'))) else 0.0
                    n_conn = storage_connection_counts.get(s_idx, 0)
                    if n_conn > 0:
                        allocated_caps.append(raw_cap / n_conn)

                gdf.at[idx, 'LPG_capacity'] = float(sum(allocated_caps))
                gdf.at[idx, 'tanks_nearby'] = len(nearby)
                gdf.at[idx, 'LPG_compliance'] = True
            else:
                gdf.at[idx, 'LPG_compliance'] = False

    return gdf


def primary_storage(gdf, default_gdf, lpg_price_raster_path, price_band=1):
    """
    Ensure storage capacity fields exist and are numeric.
    Fills LPG_capacity from nearby defaults when missing.
    Initializes cost_source as NaN for later steps.
    Returns a cleaned GeoDataFrame in the original CRS.
    """
    gdf = gdf.copy()

    # Phase 1: Fill from buffer for LPG_capacity 
    fill_cols = ['LPG_capacity']
    gdf = fill_from_buffer(gdf, default_gdf, fill_cols)

    # Phase 2: Ensure storage capacity
    if 'LPG_capacity' not in gdf.columns:
        gdf['LPG_capacity'] = 0.0
    else:
        gdf['LPG_capacity'] = gdf['LPG_capacity'].fillna(0.0)

    gdf['cost_source'] = np.nan

    return gdf


def border_points(gdf, default_gdf, nat_shares_df, lpg_price_raster_path, price_band=1):
    """
    Handles data initialization, spatial filling, raster sampling, 
    and supply share allocation for border points.
    """
    bp = gdf.copy()

    if bp.empty:
        for col in ['cost_source', 'osbp', 'border_ferry', 'percentage', 'category']:
            bp[col] = pd.Series(dtype='float64' if col not in ['osbp', 'border_ferry', 'category'] else 'object')
        return bp

    # Phase 1: Initialize columns and handle missing data
    if 'cost_source' not in bp.columns: bp['cost_source'] = np.nan
    if 'osbp' not in bp.columns: 
        bp['osbp'] = np.nan
    if 'border_ferry' not in bp.columns: 
        bp['border_ferry'] = np.nan

    # Phase 2: Fill from buffer and raster
    bp = fill_from_buffer(bp, default_gdf, ['cost_source', 'osbp', 'border_ferry'])
    bp = fill_lpg_price_from_raster(bp, lpg_price_raster_path, band_index=price_band, overwrite=False)

    bp['osbp'] = bp['osbp'].replace('', np.nan).fillna('no')
    bp['border_ferry'] = bp['border_ferry'].replace('', np.nan).fillna('no')

    # Phase 3: Share (percentage) logic
    border_share = float(nat_shares_df.loc['border_points'].iloc[0]) / 100.0

    if 'percentage' in bp.columns and bp['percentage'].notna().any():
        bp['percentage'] = bp['percentage'].fillna(0.0)
    elif 'LPG_capacity' in bp.columns and pd.to_numeric(bp['LPG_capacity'], errors='coerce').sum() > 0:
        weights = pd.to_numeric(bp['LPG_capacity'], errors='coerce').fillna(0.0)
        bp['percentage'] = border_share * (weights / weights.sum())
    else:
        osbp_flag = tool._as_bool_series(bp['osbp'])
        weights = pd.Series(np.where(osbp_flag, 2.0, 1.0), index=bp.index)
        bp['percentage'] = border_share * (weights / weights.sum())
        
    return bp

#new mattia part
import io
import contextlib
import shutil
import numpy as np
import xarray as xr
import rioxarray
from rasterio.warp import Resampling
from scipy.ndimage import gaussian_filter
from onstove import OnStove

import function_tools as tool
from parameters import (
    INCOME_GINI_VALUE, INCOME_GDP_PC_VALUE, INCOME_PARETO_WEIGHT, 
    VEHICLE_INCOME_EXPONENT, VEHICLE_TARGET_TOTAL, VEHICLE_F_URBAN, 
    VEHICLE_F_RURAL, URBAN_THRESHOLD, MIN_VEHICLE_SHARE
)

def generate_income_raster(model_pickle_path, scenario_csv_path, output_directory, output_raster_name="income_nigeria", output_variable="absolute_wealth"):
    """Estimates income utilizing the OnStove AWE method and outputs a base income raster, houseold-annual wealth in USD."""
    os.makedirs(output_directory, exist_ok=True)
    
    model = OnStove.read_model(model_pickle_path)
    model.read_scenario_data(scenario_csv_path, delimiter=",")
    model.output_directory = output_directory

    model.specs["gini"] = float(INCOME_GINI_VALUE)
    model.specs["gdp_pc"] = float(INCOME_GDP_PC_VALUE)

    with contextlib.redirect_stdout(io.StringIO()):
        model.income_estimation(awe=True, income_data=None, pareto_weight=INCOME_PARETO_WEIGHT)
        model.to_raster(variable=output_variable)

    src_name = f"{output_variable}_mean.tif"
    src_path = os.path.join(output_directory, "Rasters", src_name)
    dst_path = os.path.join(output_directory, f"{output_raster_name}.tif")

    if os.path.exists(src_path):
        if os.path.exists(dst_path):
            os.remove(dst_path)
        os.replace(src_path, dst_path)
    rasters_dir = os.path.join(output_directory, "Rasters")
    if os.path.exists(rasters_dir):
        shutil.rmtree(rasters_dir, ignore_errors=True)


def awe_limitation_recovery(
    income_da,
    urban_raster_path,
    output_nodata,
    original_nodata=None,
    population_raster_path=None,
):
    """
    Recover missing pixels (NaNs) caused by OnStove AWE methodology limitations
    using a two-tier approach: Spatial Gaussian Imputation (Local Mean) and 
    Urban/Rural National Floor fallback.
    """
    data = income_da.values.astype(np.float32, copy=True)
    valid_original = (np.isfinite(data)) & (data != output_nodata)
    if original_nodata is not None:
        valid_original &= (data != original_nodata)
    valid_original &= (data > 0)

    allowed_mask = np.ones_like(data, dtype=bool)
    if population_raster_path is not None:
        pop_da = rioxarray.open_rasterio(str(population_raster_path))
        pop_da_aligned = pop_da.rio.reproject_match(income_da, resampling=Resampling.nearest)
        pop_data = pop_da_aligned.values.astype(np.float32, copy=False)
        pop_nodata = pop_da_aligned.rio.nodata
        pop_da_aligned.close()
        pop_da.close()

        allowed_mask = np.isfinite(pop_data)
        if pop_nodata is not None:
            allowed_mask &= (pop_data != pop_nodata)
    
    urban_da = rioxarray.open_rasterio(str(urban_raster_path))
    urban_da_aligned = urban_da.rio.reproject_match(income_da, resampling=Resampling.nearest)
    urban_data = urban_da_aligned.values.astype(np.float32, copy=False)
    urban_da_aligned.close()
    urban_da.close()
    
    valid_original &= allowed_mask

    data_clean = np.where(valid_original, data, 0.0)
    weights = valid_original.astype(float)
    filled_filtered = gaussian_filter(data_clean, sigma=2)
    weights_filtered = gaussian_filter(weights, sigma=2)
    spatial_imputed = filled_filtered / np.where(weights_filtered == 0, 1, weights_filtered)
    
    mean_rural_floor = np.mean(data[valid_original & (urban_data < URBAN_THRESHOLD)]) if np.any(valid_original & (urban_data < URBAN_THRESHOLD)) else 2053.34
    mean_urban_floor = np.mean(data[valid_original & (urban_data >= URBAN_THRESHOLD)]) if np.any(valid_original & (urban_data >= URBAN_THRESHOLD)) else 4000.0
    
    is_missing = (~valid_original) & allowed_mask
    data_imputed = data.copy()
    
    # Tier 1 Local Spatial Imputation
    data_imputed = np.where(is_missing & (spatial_imputed > 0), spatial_imputed, data_imputed)
    # Tier 2 Fixed Stratified Floor Fallback
    still_missing = is_missing & (spatial_imputed == 0)
    data_imputed = np.where(still_missing & (urban_data < URBAN_THRESHOLD), mean_rural_floor, data_imputed)
    data_imputed = np.where(still_missing & (urban_data >= URBAN_THRESHOLD), mean_urban_floor, data_imputed)
    
    data_imputed[~allowed_mask] = output_nodata

    valid = (np.isfinite(data_imputed)) & (data_imputed > 0) & allowed_mask
    print(f"[AWE Recovery] Input Pixels: {valid_original.sum():,}, Recovered Pixels: {valid.sum():,}")
    return data_imputed, valid


def generate_vehicle_possibility(
    input_path,
    output_vehicle_path,
    urban_raster_path,
    population_raster_path=None,
    target_crs="EPSG:3857",
    target_resolution=1000,
    output_nodata=-9999.0,
):
    """Reprojects the income raster, normalizes it, and calculates vehicle possession probabilities."""
    income_da = rioxarray.open_rasterio(input_path)
    original_nodata = income_da.rio.nodata

    income_3857 = income_da.rio.reproject(
        target_crs,
        resolution=target_resolution,
        resampling=Resampling.nearest
    )
    income_3857 = income_3857.fillna(output_nodata)
    income_3857.rio.write_nodata(output_nodata, inplace=True)

    data, valid = awe_limitation_recovery(
        income_da=income_3857,
        urban_raster_path=urban_raster_path,
        output_nodata=output_nodata,
        original_nodata=original_nodata,
        population_raster_path=population_raster_path,
    )

    if valid.any():
        vmin, vmax = data[valid].min(), data[valid].max()
        if vmax > vmin:
            data[valid] = (data[valid] - vmin) / (vmax - vmin)
        else:
            data[valid] = 0.0

        data[valid] = np.clip(data[valid], 0.0, 1.0)
        data[valid] = np.power(data[valid], VEHICLE_INCOME_EXPONENT, dtype=np.float32)
        data[~valid] = output_nodata

    vehicle_possibility = xr.DataArray(
        data,
        coords=income_3857.coords,
        dims=income_3857.dims,
        attrs=income_3857.attrs,
        name="vehicle_possibility",
    )
    vehicle_possibility.rio.write_crs(target_crs, inplace=True)
    vehicle_possibility.rio.write_nodata(output_nodata, inplace=True)
    vehicle_possibility.rio.to_raster(output_vehicle_path, compress="DEFLATE", nodata=output_nodata)


def _total_vehicles_for_alpha(alpha, s_norm, pop, F_arr, min_share):
    """Internal helper to calculate total allocated vehicles via power relation logic."""
    rates = np.power(s_norm, alpha, dtype=np.float64)
    rates = np.where(rates < min_share, 0.0, rates)
    vehicles = pop * rates / F_arr
    return float(np.sum(vehicles, dtype=np.float64))


def allocate_vehicles(vehicle_possibility_path, population_path, urban_raster_path, output_share_car_path, output_share_walk_path, output_cars_path, output_nodata=-9999.0):
    """Allocates vehicle and walking distributions factoring urban vs. rural divides across populations."""
    score_da = rioxarray.open_rasterio(vehicle_possibility_path)
    pop_da = rioxarray.open_rasterio(population_path).rio.reproject_match(score_da, resampling=Resampling.nearest)
    urban_da = rioxarray.open_rasterio(urban_raster_path).rio.reproject_match(score_da, resampling=Resampling.nearest)

    score = score_da.values.astype(np.float64, copy=False)
    pop = pop_da.values.astype(np.float64, copy=False)
    urban = urban_da.values.astype(np.float64, copy=False)

    valid = (np.isfinite(score) & np.isfinite(pop) & np.isfinite(urban) & (pop > 0))
    if score_da.rio.nodata is not None: valid &= score != score_da.rio.nodata
    if pop_da.rio.nodata is not None: valid &= pop != pop_da.rio.nodata
    if urban_da.rio.nodata is not None: valid &= urban != urban_da.rio.nodata

    s = score[valid]
    s_min, s_max = float(np.min(s)), float(np.max(s))
    s_norm = np.clip((s - s_min) / (s_max - s_min), 0.0, 1.0) if s_max > s_min else s
    s_norm[int(np.argmax(s))] = 1.0

    pop_valid = pop[valid]
    F_valid = np.where(urban[valid] >= URBAN_THRESHOLD, VEHICLE_F_URBAN, VEHICLE_F_RURAL).astype(np.float64)

    left, right = 0.0, 40.0
    for _ in range(80):
        mid = 0.5 * (left + right)
        veh_mid = _total_vehicles_for_alpha(mid, s_norm, pop_valid, F_valid, MIN_VEHICLE_SHARE)
        if veh_mid > VEHICLE_TARGET_TOTAL:
            left = mid
        else:
            right = mid
            
    alpha = 0.5 * (left + right)
    final_veh = _total_vehicles_for_alpha(alpha, s_norm, pop_valid, F_valid, MIN_VEHICLE_SHARE)
    print(f"Calibration complete: Final Alpha={alpha:.4f}")
    print(f"Allocated Vehicles: {final_veh:,.0f} (Target: {VEHICLE_TARGET_TOTAL:,})")
    
    raw_rates_valid = np.power(s_norm, alpha, dtype=np.float64)
    rates_valid = np.where(raw_rates_valid < MIN_VEHICLE_SHARE, 0.0, raw_rates_valid)

    share_car_percent = np.full(score.shape, output_nodata, dtype=np.float32)
    walk_share_percent = np.full(score.shape, output_nodata, dtype=np.float32)
    n_effettivo = np.full(score.shape, output_nodata, dtype=np.float32)
    

    share_car_percent[valid] = (rates_valid * 100.0).astype(np.float32)
    walk_share_percent[valid] = (100.0 - share_car_percent[valid]).astype(np.float32)
    n_effettivo[valid] = (pop_valid * rates_valid / F_valid).astype(np.float32)

    def _save_and_cache(data, path, name):
        da = xr.DataArray(data, coords=score_da.coords, dims=score_da.dims, name=name)
        da.rio.write_crs(score_da.rio.crs, inplace=True)
        da.rio.write_nodata(output_nodata, inplace=True)
        da.rio.to_raster(path, compress="DEFLATE", nodata=output_nodata)

    _save_and_cache(share_car_percent, output_share_car_path, "share_car_access")
    _save_and_cache(walk_share_percent, output_share_walk_path, "walk_allocation_share")
    _save_and_cache(n_effettivo, output_cars_path, "n_effettivo")

