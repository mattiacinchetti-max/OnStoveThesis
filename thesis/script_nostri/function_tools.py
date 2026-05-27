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
import function_costs as cost
import function_allocation as alloc
import function_process as process
from dataclasses import dataclass  

def _as_vector_layer(gdf, name="mask"):
    """
    Wrap a GeoDataFrame in a VectorLayer for in-memory masking.
    Avoids writing temporary files and keeps attributes intact.
    Returns a VectorLayer ready for VectorLayer.mask.
    """
    vect = VectorLayer(name=name)
    vect.data = gdf.copy()
    return vect


def load_raster_means(raster_path, band_names, mask_gdf):
    """
    Clip each band by mask and compute mean values.
    Returns a single-column DataFrame indexed by band names.
    Raises if any band has no valid pixels inside the mask.
    """
    if not raster_path.exists():
        raise FileNotFoundError(f"Missing raster file: {raster_path}")

    if mask_gdf.crs is None:
        raise ValueError("mask_layer has no CRS")

    mask_vect = _as_vector_layer(mask_gdf, name="mask_layer")
    values = {}

    for band_idx, band_name in enumerate(band_names, start=1):
        raster = RasterLayer(name=f"{raster_path.stem}_{band_name}", path=str(raster_path), band=band_idx)
        raster.mask(mask_vect, crop=True, all_touched=False)

        arr = raster.data.astype(np.float64)
        nodata = raster.meta.get('nodata')
        if nodata is not None:
            if pd.notna(nodata):
                arr[arr == nodata] = np.nan

        mean_val = np.nanmean(arr)
        if np.isnan(mean_val):
            raise ValueError(
                f"Band '{band_name}' in {raster_path.name} has no valid data inside mask_layer"
            )
        values[band_name] = float(mean_val)

    result = pd.DataFrame.from_dict(values, orient='index')
    result.columns = [1]
    return result


def load_or_use_default(gpkg_name, default_gpkg_path, default_layer_name, mask_gdf, data_dir):
    """
    Load a layer from user data or defaults, then clip it to mask_gdf.
    Returns a GeoDataFrame with a stable id_supply column.
    No files are written during the fallback path.
    """
    gpkg_path = data_dir / f"{gpkg_name}.gpkg"
    mask_vect = _as_vector_layer(mask_gdf, name="mask_layer")

    if gpkg_path.exists():
        # Phase 1: User data found
        layer = VectorLayer(name=gpkg_name, path=str(gpkg_path))
        layer.mask(mask_vect, keep_geom_type=False)
        gdf = layer.data
        gdf = gdf.reset_index(drop=True)
        gdf['id_supply'] = f"{gpkg_name}_" + gdf.index.astype(str)
        print(f"{gpkg_name}: loaded and clipped in memory ({len(gdf)} records)")
        return gdf

    # Phase 2: Use default.gpkg layer and clip with mask_layer (no file creation)
    if not default_gpkg_path.exists():
        raise FileNotFoundError(f"Missing default gpkg: {default_gpkg_path}")

    default_gdf = gpd.read_file(default_gpkg_path, layer=default_layer_name)
    default_layer = _as_vector_layer(default_gdf, name=gpkg_name)
    default_layer.mask(mask_vect, keep_geom_type=False)

    gdf = default_layer.data
    gdf = gdf.reset_index(drop=True)
    gdf['id_supply'] = f"{gpkg_name}_" + gdf.index.astype(str)
    print(f"{gpkg_name}: clipped from default.gpkg/{default_layer_name} in memory ({len(gdf)} records)")
    return gdf


def load_border_points_or_default(gpkg_name, default_gpkg_path, default_layer_name, mask_gdf, data_dir, near_km=5):
    """
    Load border points, then keep, snap, or drop by distance to mask_gdf.
    Points within near_km get snapped to the mask boundary.
    Returns a GeoDataFrame with id_supply and original attributes.
    """
    user_path = data_dir / f"{gpkg_name}.gpkg"

    if user_path.exists():
        try:
            points = gpd.read_file(user_path, layer="border_points")
        except Exception:
            points = gpd.read_file(user_path)
        source_name = f"{user_path.name}"
    else:
        if not default_gpkg_path.exists():
            raise FileNotFoundError(f"Missing default gpkg: {default_gpkg_path}")
        points = gpd.read_file(default_gpkg_path, layer=default_layer_name)
        source_name = f"{default_gpkg_path.name}/{default_layer_name}"

    if points.empty:
        print(f"border_points: loaded 0 records from {source_name}")
        points = points.reset_index(drop=True)
        points['id_supply'] = "border_points_" + points.index.astype(str)
        return points

    if mask_gdf.crs is None:
        raise ValueError("mask_layer has no CRS")
    if points.crs is None:
        raise ValueError("border_points has no CRS")

    pts = points.to_crs(mask_gdf.crs).copy()

    pts_m = pts.to_crs("EPSG:3857").copy()
    mask_m = mask_gdf.to_crs("EPSG:3857")
    mask_union_m = mask_m.geometry.union_all()

    inside_count = 0
    moved_count = 0
    dropped_count = 0

    keep_rows = []
    new_geom_m = []

    for idx, geom_m in zip(pts_m.index, pts_m.geometry):
        if geom_m is None or geom_m.is_empty:
            dropped_count += 1
            continue

        if mask_union_m.covers(geom_m):
            inside_count += 1
            keep_rows.append(idx)
            new_geom_m.append(geom_m)
            continue

        dist_m = float(geom_m.distance(mask_union_m))
        if dist_m <= near_km * 1000:
            snapped = nearest_points(geom_m, mask_union_m)[1]
            moved_count += 1
            keep_rows.append(idx)
            new_geom_m.append(snapped)
        else:
            dropped_count += 1

    if len(keep_rows) == 0:
        out = pts.iloc[0:0].copy()
    else:
        out_m = pts_m.loc[keep_rows].copy()
        out_m.geometry = new_geom_m
        out = out_m.to_crs(mask_gdf.crs)

    out = out.reset_index(drop=True)
    out['id_supply'] = "border_points_" + out.index.astype(str)
    print(
        f"border_points: loaded from {source_name} | inside={inside_count}, moved_to_mask={moved_count}, dropped_far={dropped_count}"
    )
    return out


def _metric_crs_for_gdf(gdf):
    """
    Return a metric CRS for distance operations.
    If input is projected, return its CRS.
    If geographic, derive a local azimuthal projection from the centroid.
    Raises if CRS is missing or geometry is empty.
    """
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame has no CRS")
    if not gdf.crs.is_geographic:
        return gdf.crs
    centroid = gdf.geometry.union_all().centroid
    if centroid.is_empty:
        raise ValueError("Cannot derive metric CRS from empty geometry")
    return CRS.from_proj4(
        f"+proj=aeqd +lat_0={centroid.y} +lon_0={centroid.x} +datum=WGS84 +units=m +no_defs"
    )


def _metric_views_for_buffer(gdf, other_gdf):
    """
    Return both layers in a metric CRS for buffering.
    If input is geographic, reproject both to a local metric CRS.
    Otherwise return the inputs unchanged.
    """
    metric_crs = _metric_crs_for_gdf(gdf)
    if gdf.crs.is_geographic:
        return gdf.to_crs(metric_crs), other_gdf.to_crs(metric_crs)
    return gdf, other_gdf

def _as_bool_series(series):
    """
    Normalize mixed-type flag values to booleans.
    Accepts common truthy strings and numeric flags.
    Missing values are treated as False.
    """
    def _to_bool(v):
        if isinstance(v, bool):
            return v
        if pd.isna(v):
            return False
        s = str(v).strip().lower()
        return s in {'true', '1', 'yes', 'y', 't'}

    return series.apply(_to_bool)

def export_gpkg(points, data_dir, stage=""):
    output_path = f"{data_dir}/supply_chain_layer.gpkg"
    order = ['refineries', 'ports', 'gas_plants', 'border_points', 'primary_storage', 'primary_storage_allocated', 'filling_points', 'resell']
    written = 0
    for name in order:
        gdf = points.get(name)
        if gdf is None or gdf.empty:
            continue

        current_mode = "w"
        gdf.to_file(output_path, layer=name, driver="GPKG", mode=current_mode)
        written += 1

def combine_layers(points: dict, keys: list) -> gpd.GeoDataFrame:
    """
    Combines layers in a single GeoDataFrame 
    with necessary columns for transport cost calculation.
    """
    layers = []
    for key in keys:
        gdf = points.get(key)
        if gdf is not None and not gdf.empty:
            gdf = gdf.copy()
            gdf['supply_category'] = key   
            layers.append(gdf)
    
    if not layers:
        return gpd.GeoDataFrame()
    
    combined = pd.concat(layers, ignore_index=True)
    
    # Fill NaN values in cost columns with 0
    cost_cols = [c for c in combined.columns if c.startswith('cost_')]
    for col in cost_cols:
        combined[col] = combined[col].fillna(0.0)
    
    return gpd.GeoDataFrame(combined, crs=layers[0].crs)