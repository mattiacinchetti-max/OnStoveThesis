# Core constants
CRUDE_BARREL_TO_TONS = 7.33  # barrel of crude per ton
BUFFER_KM = 10  # km radius for spatial buffer queries

discount_rate = 0.03 #OnStove uses 0.03 for social perspective, 0.15 for private perspective

# Ports parameters
STS_cost = 0.1  # US$/kg
pre_bottling_cost = 0.2  # 20% TO BE MODIFIED

# Border points parameters
border_point_time_hours = 39
ferry_cost_per_km = 0.78  # US$/ton/straight line km
ferry_time_per_km = 0.071  # hours/straight line km

# Driver parameters
driver_annual_salary_usd = 2604
salary_multiplier = 1.18
hours_per_day = 8 # working hours per day
days_per_year = 330 # working days per year
driver_hourly_cost_usd = (driver_annual_salary_usd * salary_multiplier) / (hours_per_day * days_per_year)

from dataclasses import dataclass

@dataclass
class Vehicle:
    name: str
    capacity_kg: float
    utilization_factor: float
    overnight_cost_usd: float
    life_years: int
    variable_cost_per_km: float
    annual_km: int
    fixed_loading_unloading_hours: float

tanker = Vehicle(
    name="tanker",
    capacity_kg=24000,
    utilization_factor=0.85,
    overnight_cost_usd=197673,
    life_years=35,
    variable_cost_per_km=0.635,
    annual_km=42000,
    fixed_loading_unloading_hours=1.0,
)

truck = Vehicle(
    name="truck",
    capacity_kg=9996,
    utilization_factor=0.85,
    overnight_cost_usd=170200,
    life_years=15,
    variable_cost_per_km=0.54,
    annual_km=42000,
    fixed_loading_unloading_hours=1.0,
)

# TODO car element in vehicle class?

# license paramters
license_cost_usd=418
licence_life_years=5

# Stove parameters (OnStove defaults) (used for storage cost calculation and capacity estimation)
meals_per_day = 1.0
energy_per_meal_mj = 3.64
efficiency = 0.5
energy_content_mj_per_kg = 45.5
density_kg_per_l = 0.5
kg_per_m3 = 1000.0 * density_kg_per_l

# primary storage cost calculation options
USE_DEFAULT_RESIDENCE_TIME = True # Set to True for a fixed 15-day residence time
USE_DIFFERENTIATED_STORAGE_COSTS = False # Set to True for V1 (differentiated), False for V2 (average)

# Storage parameters
storage_overnight_cost_usd_kg = 12.5
storage_installation_fee = 760.0
storage_license_cost_usd = 3952.0
storage_life_years = 50
storage_license_life_years = 3
discount_rate_storage = float(globals().get("discount_rate", 0.03))
storage_cap = 0.25 #US$/kg, maximum price for storage cost, used to cap outliers in the data when USE_DIFFERENTIATED_STORAGE_COSTS is True

#new part
import math

# --- Reseller layer constants ---
RESELLER_ID_COLUMN = "id_res&fil"
RESELLER_LAYER     = "resell"

# Raster nodata values
NODATA = -9999
NODATA_FLOAT = -9999.0
NODATA_INT = -1

# Dijkstra / Huff parameters
LOG_EVERY = 50000                # progress report after every N assigned pixels
ATTR_MIN = 1e-6                  # minimum attractiveness (prevents division by zero)
TIME_LIMIT_FACTOR = 1.2          # early termination factor for exact recomputation
EXACT_MAX_DISTANCE_KM = 300.0    # Maximum distance (km) for exact Dijkstra recomputation; pixels farther keep approximate times.

# 8‑neighbour moves with Euclidean distance factors (pixel units)
NEIGHBORS = (
    (-1,  0, 1.0),
    ( 1,  0, 1.0),
    ( 0, -1, 1.0),
    ( 0,  1, 1.0),
    (-1, -1, math.sqrt(2.0)),
    (-1,  1, math.sqrt(2.0)),
    ( 1, -1, math.sqrt(2.0)),
    ( 1,  1, math.sqrt(2.0)),
)

# --- Demand Calibration & Allocation Parameters ---
WALK_THRESHOLD_MIN = 30
CAR_THRESHOLD_MIN = 45
URBAN_SHARE = 0.25468    # source ONSTOVE
RURAL_SHARE = 0.027905   # source ONSTOVE
URBAN_THRESHOLD = 20
NODATA_INT = -1

# --- Income & Vehicle Allocation Parameters ---

# OnStove AWE Estimation Parameters
INCOME_GINI_VALUE = 0.339
INCOME_GDP_PC_VALUE = 2139
INCOME_PARETO_WEIGHT = 0.32

# Vehicle Access Allocation Parameters
VEHICLE_INCOME_EXPONENT = 1.0
VEHICLE_TARGET_TOTAL = 11600000
VEHICLE_F_URBAN = 6.3   # Users per vehicle (Urban)
VEHICLE_F_RURAL = 7.3   # Users per vehicle (Rural)
URBAN_THRESHOLD = 20    # Urbanisation degree threshold (%)
MIN_VEHICLE_SHARE = 1e-5


# --- Filling Station Cost Parameters ---
fom_annual_usd_fill = 348400 # Gurkan et al. (2026)
vom_usd_per_kg_fill = 0.035 # Gurkan et al. (2026)
op_license_fee_usd_fill = 3952 # Gurkan et al. (2026)
op_license_validity_years_fill = 3 # Gurkan et al. (2026)
reference_filling_capacity_kg = 35_000_000 # Gurkan et al. (2026)
cost_increase_cap_pct = 0.30
loss_factor_fill = 0.005 # 0.5% Based on Hydrocarbon Processing (2010)
overnight_filling_usd = 7_000_000 # Gurkan et al. (2026)
lifetime_filling = 15 # Gurkan et al. (2026)
filling_allocation_fraction = 0.333

# --- Reseller Shop Cost Parameters ---
rent_per_month_usd_res = 30.0
license_annual_usd_res = 100.0
vom_usd_per_kg_res = 0.01
retail_allocation_fraction = 1.0
minimum_wage_usd_per_month = 73.0 # from onstove soc

# --- End User Cost Parameters ---
fixed_time_at_retailer_hours = 0.25
cylinder_weight_kg = 12.5 #onstove choice
car_variable_cost_per_km = 0.136 #source dont remember

# --- Value of Time (VOT) Parameters ---
work_hours_per_month = 240.0  # 30 days * 8 hours
wage_range = (0.2, 0.5) #onstrove choice
wage_fraction_fallback = 0.3
default_vot_usd_per_hour = wage_fraction_fallback * (minimum_wage_usd_per_month / work_hours_per_month)
