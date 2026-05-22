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
    Sample a raster at facility locations and fill LPG_price.
    Handles non-point geometries via centroids.
    Only fills missing values unless overwrite=True.
    """
    out = gdf.copy()

    if out.crs is None:
        raise ValueError("Input GeoDataFrame has no CRS")

    if 'LPG_price' not in out.columns:
        out['LPG_price'] = np.nan

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
        mask = out['LPG_price'].isna() & sampled_series.notna()

    out.loc[mask, 'LPG_price'] = sampled_series.loc[mask]
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
    fill_cols = ['crude_capacity', 'LPG_price']
    gdf = fill_from_buffer(gdf, default_gdf, fill_cols)

    # Phase 2: Ensure production and fill LPG_price from raster
    if 'crude_capacity' not in gdf.columns:
        gdf['crude_capacity'] = 0.0
    else:
        gdf['crude_capacity'] = pd.to_numeric(gdf['crude_capacity'], errors='coerce').fillna(0.0)

    gdf = fill_lpg_price_from_raster(gdf, lpg_price_raster_path, band_index=price_band, overwrite=False)

    missing_prices = int(gdf['LPG_price'].isna().sum())
    if missing_prices > 0:
        print(f"Warning: {missing_prices} refinery rows still have missing LPG_price after raster sampling")

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
    fill_cols = ['LPG_prod', 'LPG_price']
    gdf = fill_from_buffer(gdf, default_gdf, fill_cols)

    # Phase 2: Ensure production and fill LPG_price from raster
    if 'LPG_prod' not in gdf.columns:
        gdf['LPG_prod'] = 0.0
    else:
        gdf['LPG_prod'] = pd.to_numeric(gdf['LPG_prod'], errors='coerce').fillna(0.0)

    gdf = fill_lpg_price_from_raster(gdf, lpg_price_raster_path, band_index=price_band, overwrite=False)

    missing_prices = int(gdf['LPG_price'].isna().sum())
    if missing_prices > 0:
        print(f"Warning: {missing_prices} gas plant rows still have missing LPG_price after raster sampling")

    return gdf


def ports(gdf, default_gdf, storage_gdf, lpg_price_raster_path, price_band=1):
    """
    Handles data cleaning, spatial linking with storage facilities, 
    and sampling the base LPG price from the raster.
    """
    gdf = gdf.copy()

    # Phase 1: Fill missing values from buffer
    fill_cols = ['VLGC_compliance', 'LPG_price']
    gdf = fill_from_buffer(gdf, default_gdf, fill_cols)

    # Phase 2: Ensure LPG_price exists and sample from raster
    if 'LPG_price' not in gdf.columns:
        gdf['LPG_price'] = np.nan
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
    Initializes LPG_price as NaN for later steps.
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

    gdf['LPG_price'] = np.nan

    return gdf


def border_points(gdf, default_gdf, nat_shares_df, lpg_price_raster_path, price_band=1):
    """
    Handles data initialization, spatial filling, raster sampling, 
    and supply share allocation for border points.
    """
    bp = gdf.copy()

    if bp.empty:
        for col in ['LPG_price', 'osbp', 'border_ferry', 'percentage', 'category']:
            bp[col] = pd.Series(dtype='float64' if col not in ['osbp', 'border_ferry', 'category'] else 'object')
        return bp

    # Phase 1: Initialize columns and handle missing data
    if 'LPG_price' not in bp.columns: bp['LPG_price'] = np.nan
    if 'osbp' not in bp.columns: 
        bp['osbp'] = np.nan
    if 'border_ferry' not in bp.columns: 
        bp['border_ferry'] = np.nan

    # Phase 2: Fill from buffer and raster
    bp = fill_from_buffer(bp, default_gdf, ['LPG_price', 'osbp', 'border_ferry'])
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

