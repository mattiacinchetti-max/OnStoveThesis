# %% [markdown]
# # **1. Load and process data**

# %% [markdown]
# ### 1.1. Parameters, packages and paths

# %%


# %%
from __future__ import annotations

# %%
# constants and parameters
from parameters import *

# Core dependencies for tabular, spatial, and raster work
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
import rioxarray
from rasterio.warp import Resampling
import fiona

from dataclasses import dataclass

# Workspace data folder 


import function_tools as tool
import function_costs as cost
import function_allocation as alloc
import function_process as process
data_dir = Path("dataset")
data_dir.mkdir(exist_ok=True)
country = "NGA"
scenario_name = "NGA_morris"
# Default gpkg and layer names
DEFAULT_GPKG = data_dir / "default.gpkg" #TODO ricorda di modificare
DEFAULT_LAYERS = {
    'refineries': 'refineries',
    'ports': 'ports',
    'gas_plants': 'gas_plants',
    'primary_storage': 'primary_storage',
    'border_points': 'border_points',
}

# Define mask layer
mask_layer = gpd.read_file(data_dir / "mask_layer_f.gpkg") # TODO: sistema la questione mask layer - fai in modo che anche se non c'è la friction nel posto dove è collocato un punto, non ce problema)
# TODO crea pre aggiustamento di tutti i CRS

#settings
cost_risk_aversion = False # TODO remember to modify
#for MORRIS
price_variation = factor_import # TODO remember to modify 

# %% [markdown]
# ### 1.2. Data loading

# %% [markdown]
# ##### 1.2.1. Part 1

# %%
# Initialize points dictionary to store processed facility layers
points = {}

# Load facility layers (user files first, otherwise defaults)
try:
    for p_type in ['refineries', 'ports', 'gas_plants', 'primary_storage', 'border_points']:
        points[p_type] = tool.load_or_use_default(p_type, DEFAULT_GPKG, DEFAULT_LAYERS[p_type], mask_layer, data_dir)
    print("\n")
          
    # Read multiband rasters and use average over mask_layer as actual values
    national_shares = tool.load_raster_means(data_dir / "national_shares.tif", ['ports', 'refineries', 'gas_plants', 'border_points'], mask_layer)
    lpg_import_cost = tool.load_raster_means(data_dir / "lpg_import_cost.tif", ['import_land', 'import_sea'], mask_layer)

    # Calculate the mean of lpg_household_shares.tif over the masked area
    lpg_household_shares_df = tool.load_raster_means(data_dir / "lpg_household_shares.tif", ['percentage_house'], mask_layer)

    print("\nNational shares (mean over masked area, raw):")
    print(national_shares)

    # Redistribute shares below 1% across the remaining supply modes
    national_shares = process.redistribute_small_national_shares(national_shares, min_share_pct=1.0)

    print("\nNational shares (after redistribution of modes < 1%):")
    print(national_shares)

    print("\nLPG import costs (mean over masked area):")
    print(lpg_import_cost)

    # Load defaults once
    default_data = {name: gpd.read_file(DEFAULT_GPKG, layer=layer_name) for name, layer_name in DEFAULT_LAYERS.items()}

except (FileNotFoundError, ValueError) as e:
    print(f"Data loading error: {e}")

# %% [markdown]
# ##### 1.2.2. Part 2

# %%
MODEL_PICKLE_PATH          = data_dir / "model_inputs_mattia.pkl"
SCENARIO_CSV_PATH          = data_dir / "soc_specs.csv"
INCOME_RASTER_PATH         = data_dir / "income.tif"
VEHICLE_POSSIBILITY_PATH   = data_dir / "vehicle_possibility.tif"
VEHICLES_SHARE_PATH        = data_dir / "vehicles_allocation_share.tif"
WALK_SHARE_PATH            = data_dir / "walk_allocation_share.tif"
VEHICLES_N_PATH            = data_dir / "vehicles_allocation_n_effettivo.tif"
POPULATION_RASTER_PATH     = data_dir / "Population.tif"
URBAN_RASTER_PATH          = data_dir / "Urban.tif"
COUNTRY_BOUNDARIES_PATH    = data_dir / "Country_boundaries.geojson"
HUFF_PIXEL_RASTER_PATH     = data_dir / "huff_preferred_distributor_per_pixel.tif"
LPG_USE_SHARE_PATH         = data_dir / "lpg_use_share.tif"
END_USER_PRICE_PATH        = data_dir / f"end_user_price_{scenario_name}.tif"
SUPPLY_CHAIN_LAYER_PATH    = data_dir / f"supply_chain_layer_{scenario_name}.gpkg"
FRICTION_MOTO              = data_dir / "friction_moto.tif"   
FRICTION_WALK              = data_dir / "friction_walk.tif"
FULL_CHAIN_GPKG_PATH       = data_dir / "full_lpg_chain_nig_3857.gpkg"  

# TODO applicare mask layer a tutti i raster
# TODO uniformare CRS
# TODO rimuovere duplicati nei path



# --- GEOPACKAGE MERGE LOGIC ---
ACTIVE_GPKG = data_dir / "active_default.gpkg"
combined_layers = {}

# 1. Load all layers from the original default file
if DEFAULT_GPKG.exists():
    for layer_name in fiona.listlayers(str(DEFAULT_GPKG)):
        combined_layers[layer_name] = gpd.read_file(DEFAULT_GPKG, layer=layer_name)

# 2. Overwrite or add layers from the "full_chain" file
if FULL_CHAIN_GPKG_PATH.exists():
    print(f" Found {FULL_CHAIN_GPKG_PATH.name}: integrating/overwriting layers in memory.")
    for layer_name in fiona.listlayers(str(FULL_CHAIN_GPKG_PATH)):
        combined_layers[layer_name] = gpd.read_file(FULL_CHAIN_GPKG_PATH, layer=layer_name)
else:
    print(f" {FULL_CHAIN_GPKG_PATH.name} not found. Using only the base layers.")

# 3. Save everything to the new active file (creating it or overwriting from scratch)
if ACTIVE_GPKG.exists():
    ACTIVE_GPKG.unlink() 
print(f"Saving the merged file to: {ACTIVE_GPKG.name}...")
for layer_name, gdf in combined_layers.items():
    gdf.to_file(ACTIVE_GPKG, layer=layer_name, driver="GPKG")
print(" Merge completed.")

# 4. Point the rest of the script to the new file
DEFAULT_GPKG = ACTIVE_GPKG



SELL_LAYER = "resell_and_filling" #filling can serve clients # TODO assicurarsi che se l'utente vuole inserire i reseller/filling la cosa sia il piu semplice possibile (esempio: creazione degli id avvenga internamente al processo di caricamento)
FILLING_LAYER = "filling"
ID_COL = "id_res&fil"
ATTRACTIVENESS_COL = "attractiveness"


points['reseller_points'] = tool.load_or_use_default(
    gpkg_name="sell_point",          # dummy name – will not exist, fallback used
    default_gpkg_path=DEFAULT_GPKG,
    default_layer_name=SELL_LAYER,      # "resell_and_filling"
    mask_gdf=mask_layer,
    data_dir=data_dir,
    near_km=0.1,
)                                  

# Load filling points similarly
points['filling_points'] = tool.load_or_use_default(
    gpkg_name="filling_points",
    default_gpkg_path=DEFAULT_GPKG,
    default_layer_name=FILLING_LAYER,     # "filling"
    mask_gdf=mask_layer,
    data_dir=data_dir,
    near_km=0.1,
)


origins = points['reseller_points'].copy()
# Ensure attractiveness column exists (if not, add default)
#TODO: metilo quando carica utente
if ATTRACTIVENESS_COL not in origins.columns:
    origins[ATTRACTIVENESS_COL] = 1.0


# %% [markdown]
# ### 1.3. Data processing

# %% [markdown]
# ##### 1.3.1. Part 1

# %%
# Apply processing for each facility type
lpg_price_raster = data_dir / 'fob_per_kg.tif'   # TODO CEPII? dovrebbe essere apposto

points['refineries'] = process.refineries(points['refineries'], default_data['refineries'], lpg_price_raster_path=lpg_price_raster, mask_layer=mask_layer)
points['primary_storage'] = process.primary_storage(points['primary_storage'], default_data['primary_storage'], lpg_price_raster_path=lpg_price_raster)
# for ports, there is the option to use as cost_source some historic data for that port:
##    - set use_historic_data to true
##    - set historic_data_col to the name of the column in default_data['ports'] that contains the historic data to use as cost_source
##      (default.gpkg contains the columns source_price_xxxx_yyy, where xxxx is the year and yyy is either 'imp' or 'exp')
points['ports'] = process.ports(points['ports'], default_data['ports'], points['primary_storage'], lpg_price_raster_path=lpg_price_raster, mask_layer=mask_layer)
points['gas_plants'] = process.gas_plants(points['gas_plants'], default_data['gas_plants'], lpg_price_raster_path=lpg_price_raster, mask_layer=mask_layer)
points['border_points'] = process.border_points(points['border_points'], default_data['border_points'], national_shares, lpg_price_raster_path=lpg_price_raster, country=country, mask_layer=mask_layer)

# %% [markdown]
# ##### 1.3.2. Part 2

# %%
# 1) Generate the base Income Raster
process.generate_income_raster(
    model_pickle_path=os.path.normpath(str(MODEL_PICKLE_PATH)),
    scenario_csv_path=os.path.normpath(str(SCENARIO_CSV_PATH)),
    output_directory=str(data_dir),

    output_raster_name="income"
)

# 2) Normalize and map it to Vehicle Possibility
process.generate_vehicle_possibility(
    input_path=str(INCOME_RASTER_PATH),
    output_vehicle_path=str(VEHICLE_POSSIBILITY_PATH),
    urban_raster_path=str(URBAN_RASTER_PATH),
    population_raster_path=str(POPULATION_RASTER_PATH),
)

# 3) Target Alpha Allocation (Car, Walk, Users parameters)
process.allocate_vehicles(
    vehicle_possibility_path=str(VEHICLE_POSSIBILITY_PATH),
    population_path=str(POPULATION_RASTER_PATH),
    urban_raster_path=str(URBAN_RASTER_PATH),
    output_share_car_path=str(VEHICLES_SHARE_PATH),
    output_share_walk_path=str(WALK_SHARE_PATH),
    output_cars_path=str(VEHICLES_N_PATH)
)

# %%
tool.export_gpkg(points, data_dir, country)

# %% [markdown]
# # **2. Allocations**

# %%
if points['primary_storage'].geometry.is_empty.all():
    storage = False
else:
    storage = True

# %% [markdown]
# ### 2.1. Supply    Storage

# %% [markdown]
# ##### 2.1.1. Assign facilities

# %%
# Precompute friction surface once to save memory and processing time
surface_data = alloc.build_surface(FRICTION_MOTO)

# %%
if storage:  # Assign each supply point to the nearest primary storage point based on the friction surface
    for supply in ['refineries', 'ports', 'gas_plants', 'border_points']:
        points[supply] = alloc.assign_nearest(
            layer1=points['primary_storage'],
            layer2=points[supply],
            layer1_name='primary_storage',
            surface_data=surface_data,
            layer1_id_col='id_supply'
        )

tool.export_gpkg(points, data_dir, country)

# %% [markdown]
# ##### 2.1.2. Calculate percentages
# for ports, among the various options on which base the percentage calculation, consider these three:
#  - LPG_capacity: allocate based on capacity of storage sites nearby
#  - LPG_import_xxxx: allocate based on import volumes of year xxxx, with xxxx = 2023, 2024, 2025 available in default.gpkg
#  - [NIGERIA ONLY] percentage_CITAC: allocate based on the percentage of total imports for that port over the total imports of all ports of Nigeria, as reported by CITAC

# %%
# Calculate supply percentages for each supply point, then aggregate to storage level

step = "supply"

first_step, layers_by_category = alloc.calculate_percentages(step, 
    refineries_gdf = points["refineries"],
        
    ports_gdf = points["ports"], ports_cols = ['percentage_CITAC'], 
    gas_plants_gdf = points["gas_plants"], 
    border_points_gdf = points["border_points"], border_points_cols=f"percentage_importer",
    nat_shares_df = national_shares)

points = {   # TODO si puo direttamente assegnare points come risultato della funzione calculate_percentages?
    "refineries": layers_by_category["refineries"],
    "ports": layers_by_category["ports"],
    "gas_plants": layers_by_category["gas_plants"],
    "primary_storage": points["primary_storage"],
    "border_points": layers_by_category["border_points"],
    "reseller_points": points["reseller_points"],
    "filling_points": points["filling_points"]
}

if storage:
    # Aggregate supply percentages to storage level
    points["primary_storage"] = alloc.aggregate_to_storage(step,
        primary_storage_gdf=points["primary_storage"],
        refineries_gdf=points["refineries"],
        ports_gdf=points["ports"],
        gas_plants_gdf=points["gas_plants"],
        border_points_gdf=points["border_points"]
    )

tool.export_gpkg(points, data_dir, country)

# %% [markdown]
# ### 2.2. Filling    Reseller    Households

# %% [markdown]
# ##### 2.2.1. Assign facilities

# %%
# --- Reproject friction_moto to match population.tif (reference grid) ---
# population.tif is the authority for CRS, resolution, and extent.

pop_ref = rioxarray.open_rasterio(str(POPULATION_RASTER_PATH))
moto_raw = rioxarray.open_rasterio(str(FRICTION_MOTO))
moto_reprojected = moto_raw.rio.reproject_match(
    pop_ref,
    resampling=Resampling.nearest  
)
# Save to a temporary aligned path (used only for this run)
FRICTION_MOTO_ALIGNED = data_dir / "friction_moto_aligned.tif"
moto_reprojected.rio.to_raster(str(FRICTION_MOTO_ALIGNED), compress="DEFLATE")
pop_ref.close()
moto_raw.close()
moto_reprojected.close()
print(f"friction_moto reprojected to match population.tif    {FRICTION_MOTO_ALIGNED}")
FRICTION_MOTO = FRICTION_MOTO_ALIGNED  # Use the aligned version for the rest of the process
FRICTION_MOTO = FRICTION_MOTO_ALIGNED


alloc.assign_best(         # anziché usare il layer "reseller and filling" fai in modo che questo layer non esista piu!
    origins=origins,
    origin_id_col=ID_COL,
    destinations=None,
    layer1_name="reseller_walk",
    friction_path=FRICTION_WALK,
    huff=True,
    attractiveness_col=ATTRACTIVENESS_COL,
    output_raster_path=data_dir / "huff_walk.tif",
    pop_mask=str(POPULATION_RASTER_PATH),
)

alloc.assign_best(
    origins=origins,
    origin_id_col=ID_COL,
    destinations=None,
    layer1_name="reseller_moto",
    friction_path=FRICTION_MOTO,
    huff=True,
    attractiveness_col=ATTRACTIVENESS_COL,
    output_raster_path=data_dir / "huff_moto.tif",
    pop_mask=str(POPULATION_RASTER_PATH),
)

# 2. Join the data into the 8-band master file needed by the cost functions
tool.stack_huff_rasters(
    output_path=HUFF_PIXEL_RASTER_PATH,
    car_share_path=VEHICLES_SHARE_PATH,
    walk_share_path=WALK_SHARE_PATH,
    huff_walk_path=data_dir / "huff_walk.tif",
    huff_moto_path=data_dir / "huff_moto.tif"
)

# 3. Filling    reseller (nearest, no Huff)
print("\n=== Filling assignment ===")
filling = points['filling_points']
resell = points['reseller_points']

points['reseller_points'] = alloc.assign_best(
    origins=filling,
    origin_id_col=ID_COL,
    destinations=resell,
    layer1_name="filling_points",
    friction_path=FRICTION_MOTO,
    huff=False,
)

tool.export_gpkg(points, data_dir, country)

# %% [markdown]
# ##### 2.2.2. Enrich with additional data

# %%
filling_for_demand = points['filling_points'].copy()

# Enrich the in-memory layers with client demand calculations directly
points['reseller_points'], points['filling_points'] = alloc.calibrate_demand_and_allocate_clients(
    pixel_pref_path=str(HUFF_PIXEL_RASTER_PATH),
    pop_path=str(POPULATION_RASTER_PATH),
    urban_path=str(URBAN_RASTER_PATH),
    boundary_path=str(COUNTRY_BOUNDARIES_PATH),
    resell_gdf=points['reseller_points'],
    filling_gdf=filling_for_demand,
    output_lpg_use_path=str(LPG_USE_SHARE_PATH)
)

# %% [markdown]
# ### 2.3. Storage    Filling

# %% [markdown]
# ##### 2.3.1. Assign nearest

# %%
# TODO ensure that the code clips filling points and resells on the mask layer

if storage:
    points['filling_points'] = alloc.assign_nearest(
        layer1=points['primary_storage'],
        layer2=points['filling_points'],
        layer1_name='primary_storage',
        surface_data=surface_data,
        layer1_id_col='id_supply'
    )

    tool.export_gpkg(points, data_dir, country)

# %% [markdown]
# ##### 2.3.2. Calculate percentages

# %%
step = "filling"

points['filling_points'] = alloc.calculate_percentages(step, filling_points_gdf=points['filling_points']) 

if storage:
# 2.4.1: Aggregate demand percentages to storage level
    points["primary_storage"] = alloc.aggregate_to_storage(step,
        primary_storage_gdf=points["primary_storage"],
        filling_points_gdf=points['filling_points']
    )

# %% [markdown]
# ### 2.4. Storage ←   Storage

# %%
friction_path = data_dir / "friction_moto.tif"
output_excel_path = data_dir / "storage_allocation.xlsx"

# %%
if storage:
    # Run complete optimization workflow
    results = alloc.allocation_process(
        primary_storage_gdf=points['primary_storage'],
        friction_path=FRICTION_MOTO_ALIGNED,
        output_excel_path=output_excel_path
    )

# %% [markdown]
# ### 2.5. supply    filling
# ##### *for countries without storage*

# %%
if not storage:
    target_keys = ['ports', 'border_points', 'gas_plants', 'refineries', 'filling_points']

    reference_crs = None
    for key in target_keys:
        gdf = points.get(key)
        if gdf is not None and not gdf.empty and gdf.crs is not None:
            reference_crs = gdf.crs
            break

    prepared_points = {}
    for key, gdf in points.items():
        if gdf is None or gdf.empty:
            continue
            
        gdf = gdf.copy()

        if reference_crs is not None and gdf.crs != reference_crs:
            gdf = gdf.to_crs(reference_crs)
            
        if key == 'filling_points':
            gdf['percentage_demand'] = gdf.get('percentage_demand', 0.0)
            gdf['percentage_supply'] = 0.0
        else:
            gdf['percentage_supply'] = gdf.get('percentage', 0.0)
            gdf['percentage_demand'] = 0.0
            
        prepared_points[key] = gdf

    points_temporary = tool.combine_layers(prepared_points, target_keys)
    points_reseller_points = points['reseller_points'].copy()

    # Allocation
    results = alloc.allocation_process(
        primary_storage_gdf=points_temporary,
        friction_path=FRICTION_MOTO_ALIGNED,
        output_excel_path=data_dir / "supply_to_filling_allocation.xlsx"
    )

# %%
tool.export_gpkg(points, data_dir, country)

# %% [markdown]
# # **3. Costs**

# %% [markdown]
# ### 3.0. [reload data] + reload matrixes for propagation

# %%
#TODO DELETE IF NOT NEEDED ANYMORE (IT'S NEEDED WHEN YOU WANT TO RERUN HUFF AGAIN)
"""
supply_chain_layer_path = data_dir / "supply_chain_layer.gpkg"
points = {}
points['ports'] = gpd.read_file(supply_chain_layer_path, layer="ports")
points['refineries'] = gpd.read_file(supply_chain_layer_path, layer="refineries")
points['gas_plants'] = gpd.read_file(supply_chain_layer_path, layer="gas_plants")
points['border_points'] = gpd.read_file(supply_chain_layer_path, layer="border_points")
points['reseller_points'] = gpd.read_file(supply_chain_layer_path, layer="reseller_points")
points['filling_points'] = gpd.read_file(supply_chain_layer_path, layer="filling_points")
points['primary_storage'] = gpd.read_file(supply_chain_layer_path, layer="primary_storage")
"""

# %%
prefix = "storage_allocation" if storage else "supply_to_filling_allocation"
matrix = data_dir / f"{prefix}.xlsx"

df_alloc = pd.read_excel(matrix, sheet_name=f"storage_allocation_percentage", index_col=0).values.astype(np.float64)
df_dist  = pd.read_excel(matrix, sheet_name=f"storage_allocation_distance", index_col=0).values.astype(np.float64)
df_time  = pd.read_excel(matrix, sheet_name=f"storage_allocation_time", index_col=0).values.astype(np.float64)

# %% [markdown]
# ### 3.1. ports, border points

# %%
if not storage:
    points =tool.split_layers(points_temporary)

points['ports'] = cost.ports(points['ports'], lpg_import_cost)
points['border_points'] = cost.border_points(points['border_points'], lpg_import_cost, tanker)


tool.export_gpkg(points, data_dir, country)

# %% [markdown]
# ##### Cost of risk aversion

# %%
if cost_risk_aversion:
    points['ports'] = cost.risk_aversion(points['ports'], aversion_coeff, std_dev)
    points['border_points'] = cost.risk_aversion(points['border_points'], aversion_coeff, std_dev)

# %% [markdown]
# ##### Sensitivity on import price

# %%
if price_variation != 1.0:
    points['ports']['cost_source'] = points['ports']['cost_source'] * price_variation
    points['border_points']['cost_source'] = points['border_points']['cost_source'] * price_variation

# %% [markdown]
# ##### "no storage" case

# %%
if not storage:
    points_temporary = tool.combine_layers(points, ['ports', 'border_points', 'gas_plants', 'refineries', 'filling_points'])

if storage:
    points['primary_storage'] = cost.tracking(
        layer1=tool.combine_layers(points, ['ports', 'border_points', 'gas_plants', 'refineries']),
        layer2=points['primary_storage'],
        layer1_name='supply',
        layer2_name='primary_storage',
        mode='convergence',
        weight1='percentage',
        weight2='percentage_supply',
        id_col='id_supply'
    )
if not storage:
    points['filling_points'] = cost.propagate(
        layer1=points_temporary,
        allocation_matrix=df_alloc,
    )

tool.export_gpkg(points, data_dir, country)


# %% [markdown]
# ### 3.2. transport supply    storage

# %%
if storage:
    points['primary_storage'] = cost.transport(      
        layer1=tool.combine_layers(points, ['ports', 'border_points', 'gas_plants', 'refineries']), 
        layer2=points['primary_storage'],
        layer1_name='supply',
        layer2_name='primary_storage',  
        vehicle=tanker,
        mode='convergence',
        weight1='percentage',
        weight2='percentage_supply',
        id_col='id_supply',
    )

    tool.export_gpkg(points, data_dir, country)


# %% [markdown]
# ### 3.3. storage

# %%
if storage:
    points['primary_storage'] = cost.storage(
        points['primary_storage'],
        points['filling_points'], 
        lpg_household_shares_df,
        use_default_time=USE_DEFAULT_RESIDENCE_TIME,
        use_differentiated=USE_DIFFERENTIATED_STORAGE_COSTS
    )

    tool.export_gpkg(points, data_dir, country)

# %% [markdown]
# ### 3.4. Propagation across rebalancing movements

# %%
if storage: 
    points['primary_storage_allocated'] = cost.propagate(
    layer1=points['primary_storage'],
    allocation_matrix=df_alloc,
    exceptions=['cost_storage_second']
    ) 

    tool.export_gpkg(points, data_dir, country)

# %% [markdown]
# ### 3.5. transport storage    storage

# %%
if storage:
    points['primary_storage_allocated'] = cost.transport(       
    layer1=points['primary_storage_allocated'],
    layer2=points['primary_storage_allocated'],
    layer1_name='primary_storage',
    layer2_name='primary_storage',
    vehicle=tanker,
    mode='cross_exchange', 
    distance_matrix=df_dist,
    time_matrix=df_time,
    weight_matrix=df_alloc,
    id_col='id_supply'
    ) 

    tool.export_gpkg(points, data_dir, country)

# %% [markdown]
# ### 3.6. transport storage    filling

# %%
if storage: 
    points['filling_points'] = cost.tracking(
        layer1=points['primary_storage_allocated'],
        layer2=points['filling_points'],
        layer1_name='primary_storage',
        layer2_name='filling_points',
        mode='divergence',
        id_col='id_supply'
        )
        
    tool.export_gpkg(points, data_dir, country)

# %%
if storage:
    points['filling_points'] = cost.transport(       
        layer1=points['primary_storage_allocated'],
        layer2=points['filling_points'],
        layer1_name='primary_storage',
        layer2_name='filling_points',
        vehicle=tanker,
        mode='divergence', 
        id_col='id_supply' # TODO uniformare gli ID
    ) 

if not storage:
    points['filling_points'] = cost.transport(       
        layer1=points_temporary,
        layer2=points_temporary,
        layer1_name='supply',
        layer2_name='filling_points',
        vehicle=truck,
        mode='cross_exchange',
        distance_matrix=df_dist,
        time_matrix=df_time,
        weight_matrix=df_alloc, 
        id_col='id_supply' # TODO uniformare gli ID
    )
    points =tool.split_layers(points_temporary) 
    points['reseller_points'] = points_reseller_points

tool.export_gpkg(points, data_dir, country)

# %% [markdown]
# ### 3.7. filling

# %%
if storage: # countries without storage don't have a filling cost since LPG is imported pre-filled: in that case, filling plants are just warehouses
    # Calculate the filling plant internal operations
    points['filling_points'] = cost.filling(points['filling_points'])


tool.export_gpkg(points, data_dir, country)

# %% [markdown]
# ### 3.8. transport filling    reseller

# %%
# Propagate accumulated costs from filling points to resellers
points['reseller_points'] = cost.tracking(
    layer1=points['filling_points'],
    layer2=points['reseller_points'],
    layer1_name='filling_points',
    layer2_name='reseller_points',
    mode='divergence',
    id_col='id_res&fil'
)

# Calculate truck transport cost (appends 'cost_transport_filling_to_resell')
points['reseller_points'] = cost.transport(       
    layer1=points['filling_points'],
    layer2=points['reseller_points'],
    layer1_name='filling_points',
    layer2_name='reseller_points',
    vehicle=truck, 
    mode='divergence', 
    id_col='id_res&fil'
)

# %% [markdown]
# ### 3.9. reseller

# %%

points['reseller_points'] = cost.reseller(
    resell_gdf=points['reseller_points'],
    filling_gdf=points['filling_points'],
    income_raster_path=str(INCOME_RASTER_PATH),
    pop_raster_path=str(POPULATION_RASTER_PATH),
    urban_raster_path=str(URBAN_RASTER_PATH)
)

tool.export_gpkg(points, data_dir, country)

# %% [markdown]
# ### 3.10. End-User total

# %%
cost.end_user_multimodal(           # TODO ricontrolla funzione
    points=points,
    huff_raster_path=str(HUFF_PIXEL_RASTER_PATH),
    income_raster_path=str(INCOME_RASTER_PATH),
    pop_raster_path=str(POPULATION_RASTER_PATH),
    urban_raster_path=str(URBAN_RASTER_PATH),
    lpg_share_path=str(LPG_USE_SHARE_PATH),
    output_path=str(END_USER_PRICE_PATH),
    vehicle=van,          
    threshold_min=user_threshold_minutes,
        # set treshold_min to 0 for "pure" CRM (people receive LPG at home)
        # set treshold_min to 9999 for "pure" CCCM (people buy LPG at the store, no delivery)
    use_spatial_vot=True
)

# %%
tool.export_gpkg(points, data_dir, country)


# ============================================================
# MORRIS OUTPUT BLOCK  – append this at the very end of supply.py
# Reads END_USER_PRICE_PATH (band=mean_user_cost) and Population.tif,
# then writes 5 comma-separated metrics to temp_output.txt.
# ============================================================

_out_path   = str(END_USER_PRICE_PATH)          # already defined earlier in supply.py
_pop_path   = str(POPULATION_RASTER_PATH)       # already defined earlier in supply.py
_txt_path   = "temp_output.txt"

try:
    with rasterio.open(_out_path) as _src:
        # Read band 19 (mean_user_cost).  Adjust band index if your raster differs.
        _nodata = _src.nodata
        _price  = _src.read(19).astype(np.float64)

    with rasterio.open(_pop_path) as _src2:
        _pop = _src2.read(1).astype(np.float64)
        _pop_nodata = _src2.nodata

    # ---- build valid mask ----
    _mask = np.ones(_price.shape, dtype=bool)
    if _nodata is not None:
        _mask &= ~np.isclose(_price, _nodata)
    _mask &= np.isfinite(_price) & (_price > 0)

    if _pop_nodata is not None:
        _mask &= ~np.isclose(_pop, _pop_nodata)
    _mask &= np.isfinite(_pop) & (_pop >= 0)

    _p_valid  = _price[_mask]
    _w_valid  = _pop[_mask]

    # ---- population-weighted mean ----
    _w_sum = _w_valid.sum()
    if _w_sum > 0:
        _pw_mean = float(np.average(_p_valid, weights=_w_valid))
    else:
        _pw_mean = float(np.mean(_p_valid))

    # ---- population-weighted percentiles (10th & 90th) ----
    # Sort by price, build cumulative weight
    _sort_idx = np.argsort(_p_valid)
    _p_sorted = _p_valid[_sort_idx]
    _w_sorted = _w_valid[_sort_idx]
    _cdf = np.cumsum(_w_sorted) / (_w_sorted.sum() + 1e-30)

    def _weighted_quantile(q):
        idx = np.searchsorted(_cdf, q)
        idx = min(idx, len(_p_sorted) - 1)
        return float(_p_sorted[idx])

    _pw_p10 = _weighted_quantile(0.10)
    _pw_p90 = _weighted_quantile(0.90)

    # ---- unweighted min / max (useful diagnostic, cheap to add) ----
    _price_min = float(_p_valid.min())
    _price_max = float(_p_valid.max())

    # ---- reseller cost_* weighted averages ----
    try:
        _resell_path = data_dir / f"supply_chain_layer_{country}.gpkg"
        _resell_gdf = gpd.read_file(str(_resell_path), layer="reseller_points")
        
        # weight column, exclude 0 and NaN
        _w_col = "clients_max_ideal"
        _cost_cols = sorted([c for c in _resell_gdf.columns if c.startswith("cost_")])
        
        _resell_metrics = {}
        for _col in _cost_cols:
            _mask_r = (
                _resell_gdf[_w_col].notna() & (_resell_gdf[_w_col] > 0) &
                _resell_gdf[_col].notna() & np.isfinite(_resell_gdf[_col].astype(float))
            )
            _sub = _resell_gdf[_mask_r]
            if len(_sub) > 0:
                _wsum = _sub[_w_col].astype(float).sum()
                _resell_metrics[_col] = float(
                    (_sub[_col].astype(float) * _sub[_w_col].astype(float)).sum() / _wsum
                )
            else:
                _resell_metrics[_col] = float("nan")
    except Exception as _re:
        print(f"[MORRIS OUTPUT] WARNING: could not read reseller metrics: {_re}")
        _cost_cols = []
        _resell_metrics = {}





    # ---- city pixel prices ----
    # Format: (row_0based, col_0based) as read from QGIS
    _CITIES = {
        "city_lagos":   (832, 74), 
        "city_kano":    (217, 652),
        "city_abuja":   (553, 532),
        "city_benincity": (855, 330),
        "port_harcourt": (1022, 482),
    }

    _city_prices = {}
    for _cname, (_r, _c) in _CITIES.items():
        try:
            _val = float(_price[_r, _c])
            # treat nodata / negative as NaN
            if (_nodata is not None and np.isclose(_val, _nodata)) or not np.isfinite(_val) or _val <= 0:
                _val = float("nan")
        except IndexError:
            _val = float("nan")
        _city_prices[_cname] = _val

    # ---- write output ----
    _city_str = ",".join(f"{_city_prices[c]:.6f}" for c in sorted(_city_prices))
    _resell_str = ",".join(f"{_resell_metrics[c]:.6f}" for c in _cost_cols)
    _line = f"{_pw_mean:.6f},{_pw_p10:.6f},{_pw_p90:.6f},{_price_min:.6f},{_price_max:.6f},{_city_str}\n"
    with open(_txt_path, "w") as _f:
        _f.write(_line)

    print(f"[MORRIS OUTPUT] mean={_pw_mean:.4f}  P10={_pw_p10:.4f}  "
          f"P90={_pw_p90:.4f}  min={_price_min:.4f}  max={_price_max:.4f}")
    for _cname, _cv in sorted(_city_prices.items()):
        print(f"[MORRIS OUTPUT]   {_cname}: {_cv:.4f}")
    print(f"[MORRIS OUTPUT] cost_cols order: {_cost_cols}")
    
except Exception as _e:
    with open(_txt_path, "w") as _f:
        _f.write(",".join(["ERROR"] * 22) + "\n")
    raise RuntimeError(f"[MORRIS OUTPUT] Failed to compute metrics: {_e}") from _e
# ============================================================
# END OF MORRIS OUTPUT BLOCK
# ============================================================