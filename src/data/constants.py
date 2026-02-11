"""
Domain constants and probability distributions for Italian motor insurance.
These drive the synthetic data generator to produce realistic claims.
"""

# ============================================================
# VEHICLE MARKET (Italian market shares, roughly)
# ============================================================

VEHICLE_CATALOG: list[dict] = [
    # Make, Model, min_year, max_year, value_range (EUR)
    {"make": "Fiat", "model": "Panda", "min_year": 2012, "max_year": 2024, "value": (4_000, 16_000)},
    {"make": "Fiat", "model": "500", "min_year": 2010, "max_year": 2024, "value": (3_500, 22_000)},
    {"make": "Fiat", "model": "Tipo", "min_year": 2016, "max_year": 2024, "value": (8_000, 25_000)},
    {"make": "Fiat", "model": "Punto", "min_year": 2008, "max_year": 2018, "value": (2_000, 10_000)},
    {"make": "Volkswagen", "model": "Golf", "min_year": 2012, "max_year": 2024, "value": (8_000, 38_000)},
    {"make": "Volkswagen", "model": "Polo", "min_year": 2012, "max_year": 2024, "value": (6_000, 26_000)},
    {"make": "Volkswagen", "model": "T-Roc", "min_year": 2018, "max_year": 2024, "value": (20_000, 40_000)},
    {"make": "Toyota", "model": "Yaris", "min_year": 2012, "max_year": 2024, "value": (6_000, 25_000)},
    {"make": "Toyota", "model": "Corolla", "min_year": 2015, "max_year": 2024, "value": (12_000, 35_000)},
    {"make": "Toyota", "model": "RAV4", "min_year": 2016, "max_year": 2024, "value": (18_000, 45_000)},
    {"make": "Renault", "model": "Clio", "min_year": 2012, "max_year": 2024, "value": (5_000, 22_000)},
    {"make": "Renault", "model": "Captur", "min_year": 2015, "max_year": 2024, "value": (10_000, 30_000)},
    {"make": "Peugeot", "model": "208", "min_year": 2015, "max_year": 2024, "value": (7_000, 28_000)},
    {"make": "Peugeot", "model": "308", "min_year": 2014, "max_year": 2024, "value": (9_000, 35_000)},
    {"make": "Ford", "model": "Fiesta", "min_year": 2010, "max_year": 2023, "value": (4_000, 22_000)},
    {"make": "Ford", "model": "Focus", "min_year": 2012, "max_year": 2024, "value": (7_000, 32_000)},
    {"make": "BMW", "model": "Serie 3", "min_year": 2014, "max_year": 2024, "value": (15_000, 55_000)},
    {"make": "BMW", "model": "X1", "min_year": 2016, "max_year": 2024, "value": (20_000, 52_000)},
    {"make": "Audi", "model": "A3", "min_year": 2014, "max_year": 2024, "value": (14_000, 45_000)},
    {"make": "Audi", "model": "Q3", "min_year": 2016, "max_year": 2024, "value": (22_000, 50_000)},
    {"make": "Mercedes-Benz", "model": "Classe A", "min_year": 2015, "max_year": 2024, "value": (16_000, 48_000)},
    {"make": "Mercedes-Benz", "model": "GLA", "min_year": 2017, "max_year": 2024, "value": (25_000, 55_000)},
    {"make": "Opel", "model": "Corsa", "min_year": 2012, "max_year": 2024, "value": (5_000, 24_000)},
    {"make": "Citroen", "model": "C3", "min_year": 2014, "max_year": 2024, "value": (6_000, 22_000)},
    {"make": "Hyundai", "model": "Tucson", "min_year": 2016, "max_year": 2024, "value": (15_000, 40_000)},
    {"make": "Hyundai", "model": "i20", "min_year": 2014, "max_year": 2024, "value": (6_000, 22_000)},
    {"make": "Dacia", "model": "Sandero", "min_year": 2016, "max_year": 2024, "value": (8_000, 16_000)},
    {"make": "Dacia", "model": "Duster", "min_year": 2014, "max_year": 2024, "value": (10_000, 28_000)},
    {"make": "Jeep", "model": "Renegade", "min_year": 2015, "max_year": 2024, "value": (14_000, 35_000)},
    {"make": "Alfa Romeo", "model": "Giulietta", "min_year": 2012, "max_year": 2020, "value": (6_000, 25_000)},
    {"make": "Alfa Romeo", "model": "Stelvio", "min_year": 2017, "max_year": 2024, "value": (25_000, 60_000)},
    {"make": "Lancia", "model": "Ypsilon", "min_year": 2012, "max_year": 2024, "value": (4_000, 18_000)},
]

# Weights for vehicle selection (Italian market share approximation)
VEHICLE_WEIGHTS: list[float] = [
    0.10, 0.08, 0.04, 0.03,   # Fiat (25%)
    0.05, 0.04, 0.03,          # VW (12%)
    0.04, 0.03, 0.02,          # Toyota (9%)
    0.04, 0.03,                 # Renault (7%)
    0.03, 0.03,                 # Peugeot (6%)
    0.03, 0.02,                 # Ford (5%)
    0.03, 0.02,                 # BMW (5%)
    0.02, 0.02,                 # Audi (4%)
    0.02, 0.02,                 # Mercedes (4%)
    0.03,                       # Opel (3%)
    0.03,                       # Citroen (3%)
    0.02, 0.02,                 # Hyundai (4%)
    0.02, 0.02,                 # Dacia (4%)
    0.02,                       # Jeep (2%)
    0.02, 0.02,                 # Alfa Romeo (4%)
    0.04,                       # Lancia (4%)
]

FUEL_TYPES = ["petrol", "diesel", "electric", "hybrid"]
FUEL_WEIGHTS = [0.35, 0.30, 0.10, 0.25]

# ============================================================
# GEOGRAPHY (Italian cities + provinces)
# ============================================================

CITIES: list[dict] = [
    {"city": "Milano", "province": "MI", "weight": 0.14},
    {"city": "Roma", "province": "RM", "weight": 0.13},
    {"city": "Napoli", "province": "NA", "weight": 0.08},
    {"city": "Torino", "province": "TO", "weight": 0.07},
    {"city": "Bologna", "province": "BO", "weight": 0.05},
    {"city": "Firenze", "province": "FI", "weight": 0.04},
    {"city": "Palermo", "province": "PA", "weight": 0.04},
    {"city": "Genova", "province": "GE", "weight": 0.03},
    {"city": "Bari", "province": "BA", "weight": 0.04},
    {"city": "Catania", "province": "CT", "weight": 0.03},
    {"city": "Verona", "province": "VR", "weight": 0.03},
    {"city": "Padova", "province": "PD", "weight": 0.03},
    {"city": "Brescia", "province": "BS", "weight": 0.03},
    {"city": "Bergamo", "province": "BG", "weight": 0.02},
    {"city": "Modena", "province": "MO", "weight": 0.02},
    {"city": "Reggio Emilia", "province": "RE", "weight": 0.02},
    {"city": "Parma", "province": "PR", "weight": 0.02},
    {"city": "Cagliari", "province": "CA", "weight": 0.02},
    {"city": "Perugia", "province": "PG", "weight": 0.02},
    {"city": "Taranto", "province": "TA", "weight": 0.02},
    {"city": "Messina", "province": "ME", "weight": 0.02},
    {"city": "Salerno", "province": "SA", "weight": 0.02},
    {"city": "Trieste", "province": "TS", "weight": 0.02},
    {"city": "Ravenna", "province": "RA", "weight": 0.01},
    {"city": "Lecce", "province": "LE", "weight": 0.02},
    {"city": "Pescara", "province": "PE", "weight": 0.01},
    {"city": "Sassari", "province": "SS", "weight": 0.01},
    {"city": "Monza", "province": "MB", "weight": 0.02},
]

# ============================================================
# INCIDENT TYPES & DISTRIBUTIONS
# ============================================================

INCIDENT_TYPES = [
    "collision",         # two+ vehicles
    "single_vehicle",    # hit guardrail, tree, ditch
    "hit_and_run",       # other party fled
    "parking",           # damage while parked
    "theft",             # vehicle stolen
    "vandalism",         # intentional damage
    "weather",           # hail, flooding, storm
]

INCIDENT_WEIGHTS = [0.35, 0.15, 0.10, 0.12, 0.10, 0.08, 0.10]

# ============================================================
# POLICY TYPES
# ============================================================

# third_party: RC only; comprehensive: full coverage incl. own damage; fire_theft: incendio/furto.
POLICY_TYPES = ["third_party", "comprehensive", "fire_theft"]
POLICY_WEIGHTS = [0.45, 0.35, 0.20]

# ============================================================
# DOCUMENT TYPES THAT CAN BE ATTACHED
# ============================================================

DOCUMENT_TYPES = [
    "photo_damage",
    "photo_scene",
    "police_report",
    "repair_estimate",
    "medical_report",
    "witness_statement",
    "id_document",
    "cai_form",           # Constatazione Amichevole di Incidente (Italian joint statement)
]

# Which docs are typically expected per incident type
EXPECTED_DOCS: dict[str, list[str]] = {
    "collision": ["photo_damage", "cai_form", "repair_estimate"],
    "single_vehicle": ["photo_damage", "photo_scene", "repair_estimate"],
    "hit_and_run": ["photo_damage", "police_report", "repair_estimate"],
    "parking": ["photo_damage", "repair_estimate"],
    "theft": ["police_report", "id_document"],
    "vandalism": ["photo_damage", "police_report", "repair_estimate"],
    "weather": ["photo_damage", "repair_estimate"],
}

# ============================================================
# FRAUD PATTERNS
# ============================================================

# staged: incident arranged/fabricated; inflated: real incident, claim overstated;
# phantom: incident never happened; repeated: same damage claimed multiple times.
FRAUD_TYPES = ["staged", "inflated", "phantom", "repeated"]

# Fraud signals / red flags that we'll bake into the data
FRAUD_RATE = 0.08  # ~8% of claims have fraud

# ============================================================
# COMPLEXITY DRIVERS
# ============================================================

# These map incident characteristics → complexity
# Used by the generator to assign ground truth labels
COMPLEXITY_RULES = {
    "injuries": {"none": 0, "minor": 1, "moderate": 2, "severe": 3},
    "num_parties": {1: 0, 2: 1, 3: 2},       # 3+ = always complex
    "police_report": {True: 0.5, False: 0},
    "incident_type_base": {
        "parking": -1,
        "weather": 0,
        "vandalism": 0,
        "collision": 1,
        "single_vehicle": 0,
        "hit_and_run": 1.5,
        "theft": 1,
    },
}
