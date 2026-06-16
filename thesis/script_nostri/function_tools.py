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
import function_allocation as alloc
import function_process as process
from dataclasses import dataclass  

# Francesco part

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

import geopandas as gpd
from shapely.ops import nearest_points

def load_or_use_default(gpkg_name, default_gpkg_path, default_layer_name, mask_gdf, data_dir, near_km=5, inward_shift_m=500):
    """
    Load a layer from user data or defaults. Keep geometries within mask_gdf,
    snap points within near_km to a buffered inner boundary (inward_shift_m), and drop the rest.
    Returns a GeoDataFrame with a stable "id_supply" column.
    """
    gpkg_path = data_dir / f"{gpkg_name}.gpkg"

    # Phase 1 & 2: Data Loading
    if gpkg_path.exists():
        gdf = gpd.read_file(gpkg_path)
        source_name = f"{gpkg_path.name}"
    else:
        if not default_gpkg_path.exists():
            raise FileNotFoundError(f"Missing default gpkg: {default_gpkg_path}")
        gdf = gpd.read_file(default_gpkg_path, layer=default_layer_name)
        source_name = f"{default_gpkg_path.name}/{default_layer_name}"

    if gdf.empty:
        print(f"{gpkg_name}: loaded 0 records from {source_name}")
        gdf = gdf.reset_index(drop=True)
        gdf['id_supply'] = f"{gpkg_name}_" + gdf.index.astype(str)
        return gdf

    # CRS Validation
    if mask_gdf.crs is None:
        raise ValueError("mask_gdf has no CRS")
    if gdf.crs is None:
        raise ValueError(f"{gpkg_name} has no CRS")

    # Projection for Metric Distance Calculations
    original_crs = mask_gdf.crs
    gdf_m = gdf.to_crs("EPSG:3857").copy()
    mask_m = mask_gdf.to_crs("EPSG:3857")
    mask_union_m = mask_m.geometry.union_all()

    inner_mask_m = mask_union_m.buffer(-inward_shift_m)
    if inner_mask_m.is_empty:

        inner_mask_m = mask_union_m

    inside_count = 0
    moved_count = 0
    dropped_count = 0

    keep_rows = []
    new_geom_m = []

    # Keeping, Snapping, and Dropping Logic
    for idx, geom_m in zip(gdf_m.index, gdf_m.geometry):
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
            snapped = nearest_points(geom_m, inner_mask_m)[1]
            moved_count += 1
            keep_rows.append(idx)
            new_geom_m.append(snapped)
        else:
            dropped_count += 1

    # GeoDataFrame Reconstruction
    if len(keep_rows) == 0:
        out = gdf.iloc[0:0].copy()
    else:
        out_m = gdf_m.loc[keep_rows].copy()
        out_m.geometry = new_geom_m
        out = out_m.to_crs(original_crs)

    out = out.reset_index(drop=True)
    out['id_supply'] = f"{gpkg_name}_" + out.index.astype(str)
    
    print(f"{gpkg_name}: loaded from {source_name} | inside={inside_count}, moved_inside_mask={moved_count}, dropped_far={dropped_count}")
    
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

def export_gpkg(points, data_dir, country):
    output_path = f"{data_dir}/supply_chain_layer_{country}.gpkg"
    order = ['refineries', 'ports', 'gas_plants', 'border_points', 'primary_storage', 'primary_storage_allocated', 'filling_points', 'reseller_points']
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

def split_layers(combined_gdf: gpd.GeoDataFrame) -> dict[str, gpd.GeoDataFrame]:
    """
    Splits a single combined GeoDataFrame back into separate layers
    based on 'id_supply' prefixes or 'id_res&fil' default naming.
    Handles compound names like 'border_points' and 'gas_plants'.
    """
    if combined_gdf.empty:
        return {}

    combined = combined_gdf.copy()
    target_series = pd.Series(index=combined.index, dtype='object')

    # Rule 1: Extract everything before the LAST underscore to support compound names
    if 'id_supply' in combined.columns:
        valid_supply = combined['id_supply'].notna() & (combined['id_supply'] != '')
        if np.any(valid_supply):
            prefixes = combined.loc[valid_supply, 'id_supply'].astype(str).str.rsplit('_', n=1).str[0]
            target_series[valid_supply] = prefixes

    # Rule 2: If 'id_res&fil' exists, assign 'filling_points'
    if 'id_res&fil' in combined.columns:
        valid_resfil = combined['id_res&fil'].notna() & (combined['id_res&fil'] != '')
        if np.any(valid_resfil):
            target_series[valid_resfil] = 'filling_points'

    combined['temporary_layer_name'] = target_series
    cleaned_combined = combined.dropna(subset=['temporary_layer_name'])

    separated_layers = {}
    for layer_name, group in cleaned_combined.groupby('temporary_layer_name'):
        cols_to_drop = ['temporary_layer_name']
        if 'supply_category' in group.columns:
            cols_to_drop.append('supply_category')
            
        final_gdf = group.drop(columns=cols_to_drop)
        separated_layers[str(layer_name)] = gpd.GeoDataFrame(final_gdf, crs=combined_gdf.crs)

    return separated_layers

# Mattia part

def _read_raster_base(path: str | Path, positive_only: bool = False):
    """Shared raster read logic for _read_raster and _read_friction."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32, copy=False)
        nodata = src.nodata
        profile = src.profile.copy()
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr).astype(np.float32)
    if positive_only:
        arr = np.where(arr > 0, arr, np.nan).astype(np.float32)
    return arr, profile, nodata


def _read_raster(path: str | Path):
    return _read_raster_base(path, positive_only=False)


def _read_friction(path: Path):
    arr, profile, _ = _read_raster_base(path, positive_only=True)
    return arr, profile

def _read_multiband_raster(path: str | Path):
    with rasterio.open(path) as src:
        arr = src.read()
        desc = list(src.descriptions)
        nodata = src.nodata
    return arr, desc, nodata

def write_multiband_raster(
    out_path: Path,
    ref_profile: Dict,
    bands: Sequence[np.ndarray],
    band_names: Sequence[str],
) -> None:
    """Write a multi‑band GeoTIFF with LZW compression and band descriptions."""
    profile = ref_profile.copy()
    profile.update(
        driver="GTiff",
        dtype="float32",
        count=len(bands),
        compress="lzw",
        nodata=NODATA,
        tiled=True,
        bigtiff="IF_SAFER",
    )
    # Ensure tile sizes are multiples of 16
    h, w = profile["height"], profile["width"]
    tile_x = min(256, w)
    tile_y = min(256, h)
    tile_x = max(16, (tile_x // 16) * 16)
    tile_y = max(16, (tile_y // 16) * 16)
    profile["blockxsize"] = tile_x
    profile["blockysize"] = tile_y
    with rasterio.open(out_path, "w", **profile) as dst:
        for i, (arr, name) in enumerate(zip(bands, band_names), start=1):
            dst.write(arr.astype(np.float32), i)
            dst.set_band_description(i, name)

def _map_points_to_grid(gdf: gpd.GeoDataFrame, transform, width: int, height: int):
    rows, cols = rasterio.transform.rowcol(
        transform, gdf.geometry.x.values, gdf.geometry.y.values
    )
    rows = np.asarray(rows, dtype=np.int32)
    cols = np.asarray(cols, dtype=np.int32)
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    return rows, cols, inside

def _safe_ratio(num: float, den: float) -> float:
    """Return num/den, or 0.0 if den == 0."""
    return float(num / den) if den else 0.0

def _print_progress(
    mode: str,
    assigned: int,
    total: int,
    elapsed: float,
) -> None:
    """Print progress with speed and ETA."""
    if total <= 0:
        return
    pct = 100.0 * assigned / total
    rate = _safe_ratio(assigned, elapsed)
    remaining = max(total - assigned, 0)
    eta_sec = _safe_ratio(remaining, max(rate, 1e-12))
    print(
        f"[{mode}] {assigned:,}/{total:,} ({pct:.2f}%) | "
        f"{rate:.1f} px/s | ETA {eta_sec/60:.1f} min"
    )

def read_reference_population(path: Path) -> Tuple[np.ndarray, Dict]:
    """Read population raster and return array + profile."""
    with rasterio.open(path) as src:
        pop = src.read(1).astype(np.float32)
        profile = src.profile.copy()
    return pop, profile

def reproject_to_reference(
    src_path: Path,
    ref_profile: Dict,
    resampling: Resampling,
) -> np.ndarray:
    """Reproject (or resample) a raster to match the reference grid."""
    dst = np.full(
        (ref_profile["height"], ref_profile["width"]),
        np.nan,
        dtype=np.float32,
    )
    with rasterio.open(src_path) as src:
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

def normalize_car_share(arr: np.ndarray) -> np.ndarray:
    """Convert car share to fraction (0–1). Assumes input may be 0–100."""
    out = arr.astype(np.float32, copy=True)
    finite = np.isfinite(out)
    if finite.any():
        vmax = float(np.nanmax(out[finite]))
        if vmax > 1.0 + 1e-4 and vmax <= 100.0 + 1e-4:
            out[finite] = out[finite] / 100.0
    out[~finite] = 0.0
    np.clip(out, 0.0, 1.0, out=out)
    return out

def sanitize_friction(arr: np.ndarray) -> np.ndarray:
    """Set non‑positive or non‑finite friction to NaN."""
    out = arr.astype(np.float32, copy=True)
    invalid = ~np.isfinite(out) | (out <= 0.0)
    out[invalid] = np.nan
    return out

def _read_pixel_preference_raster(path: str) -> dict[str, np.ndarray]:
    with rasterio.open(path) as src:
        if src.count !=8:   
            raise RuntimeError(f"Expected 8 bands in {path}, found {src.count}.")
        arr = src.read(indexes=[1, 2, 3, 4, 5, 6, 7, 8]).astype(np.float32, copy=False) 
        nodata = src.nodata
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr).astype(np.float32)
    return {
        "car_share": arr[0],
        "walk_share": arr[1],
        "best_reseller_id_walk": arr[2],
        "best_time_walk_min": arr[3],
        "best_distance_walk_km": arr[4],
        "best_reseller_id_car": arr[5],
        "best_time_car_min": arr[6],
        "best_distance_car_km": arr[7]
    }

def _read_reseller_ids(gdf: gpd.GeoDataFrame) -> np.ndarray:
    if RESELLER_ID_COLUMN not in gdf.columns:
        raise KeyError(f"Missing column '{RESELLER_ID_COLUMN}' in reseller layer.")
    rid = pd.to_numeric(gdf[RESELLER_ID_COLUMN], errors="coerce")
    if rid.isna().any():
        raise ValueError(f"Column '{RESELLER_ID_COLUMN}' contains non-numeric values.")
    if (rid <= 0).any():
        raise ValueError(f"Column '{RESELLER_ID_COLUMN}' must contain positive values.")
    if not rid.is_unique:
        raise ValueError(f"Column '{RESELLER_ID_COLUMN}' must be unique in layer '{RESELLER_LAYER}'.")
    return rid.astype(np.int64).to_numpy()

def _read_single_band(path: Path) -> tuple[np.ndarray, dict]:
    arr, profile, _ = _read_raster(path)
    return arr, profile

def income_household_to_per_capita(
    income_array: np.ndarray,
    urban_array: np.ndarray,
    urban_threshold: float = URBAN_THRESHOLD,
    f_urban: float = VEHICLE_F_URBAN,
    f_rural: float = VEHICLE_F_RURAL,
) -> np.ndarray:
    """
    Convert annual household income raster to annual per-capita income.
    Uses spatially-differentiated household sizes (urban vs rural).
    """
    hh_size = np.where(urban_array >= urban_threshold, f_urban, f_rural).astype(np.float32)
    return (income_array / hh_size).astype(np.float32)

def stack_huff_rasters(
    output_path: Path, 
    car_share_path: Path, 
    walk_share_path: Path, 
    huff_walk_path: Path, 
    huff_moto_path: Path
):
    """
    Combines individual share and huff rasters into the 8-band format 
    required by the cost model downstream.
    """
    with rasterio.open(car_share_path) as src_cs, \
         rasterio.open(walk_share_path) as src_ws, \
         rasterio.open(huff_walk_path) as src_hw, \
         rasterio.open(huff_moto_path) as src_hm:
        
        profile = src_hw.profile.copy()
        profile.update(count=8, nodata=-9999.0, compress="lzw")

        def _read_aligned(src: rasterio.io.DatasetReader, band: int) -> np.ndarray:
            arr = src.read(band).astype(np.float32, copy=False)
            if (
                src.transform == profile["transform"]
                and src.crs == profile["crs"]
                and src.width == profile["width"]
                and src.height == profile["height"]
            ):
                return arr

            aligned = np.full(
                (profile["height"], profile["width"]),
                profile["nodata"],
                dtype=np.float32,
            )
            reproject(
                source=arr,
                destination=aligned,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=profile["transform"],
                dst_crs=profile["crs"],
                dst_nodata=profile["nodata"],
                resampling=Resampling.nearest,
            )
            return aligned

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(_read_aligned(src_cs, 1), 1)
            dst.write(_read_aligned(src_ws, 1), 2)
            dst.write(_read_aligned(src_hw, 1), 3) # Walk ID
            dst.write(_read_aligned(src_hw, 2), 4) # Walk Time
            dst.write(_read_aligned(src_hw, 3), 5) # Walk Dist
            dst.write(_read_aligned(src_hm, 1), 6) # Car ID
            dst.write(_read_aligned(src_hm, 2), 7) # Car Time
            dst.write(_read_aligned(src_hm, 3), 8) # Car Dist
            
            # Set metadata
            for i, name in enumerate(["car_share", "walk_share", "best_reseller_id_walk", 
                                      "best_time_walk_min", "best_distance_walk_km",
                                      "best_reseller_id_car", "best_time_car_min", 
                                      "best_distance_car_km"], 1):
                dst.set_band_description(i, name)