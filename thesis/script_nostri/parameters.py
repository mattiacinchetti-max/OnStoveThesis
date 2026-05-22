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

# Tankers parameters
tanker_capacity_kg = 24000
utilization_factor = 0.85
variable_cost_per_km = 0.635
tanker_overnight_cost_usd = 197673 
tanker_life_years = 35
license_cost_usd = 418
licence_life_years = 5
fixed_loading_unloading_hours = 1.0
tanker_annual_km = 42000 

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