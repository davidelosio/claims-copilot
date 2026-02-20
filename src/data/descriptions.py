from __future__ import annotations

from datetime import date, time

import numpy as np

from src.data.constants import VEHICLE_CATALOG
from src.data.templates import (
    CAUSES_SINGLE_VEHICLE,
    CLAIMED_EXTRAS_FRAUD,
    CROSS_ROADS,
    DAMAGE_AREAS,
    DAMAGE_DESCRIPTIONS,
    DESTINATIONS,
    DIRECTIONS,
    FRAUD_TEMPLATES_BY_TYPE,
    IMPACT_POINTS,
    INJURY_TEXTS,
    LOCATIONS,
    OBSTACLES,
    PARKING_TYPES,
    POLICE_TEXTS,
    ROAD_CONDITIONS,
    ROADS,
    TEMPLATES_BY_TYPE,
    TRAFFIC_CONTROLS,
    WEATHER_EVENTS,
    WITNESS_TEXTS,
)


def generate_description(
    rng: np.random.Generator,
    incident_type: str,
    incident_date: date,
    incident_time: time,
    incident_city: str,
    vehicle: dict,
    injury_severity: str,
    police_report: bool,
    damage_estimate: float,
    is_fraud: bool,
    fraud_type: str | None,
) -> str:
    templates = _pick_templates(incident_type, is_fraud, fraud_type)
    if not templates:
        return f"Claim for {incident_type} incident on {incident_date} in {incident_city}."

    template = templates[int(rng.integers(0, len(templates)))]
    report_num = f"{rng.integers(1000, 9999)}/{incident_date.year}"
    fill = {
        "date": incident_date.strftime("%d/%m/%Y"),
        "time": incident_time.strftime("%H:%M"),
        "vehicle": f"{vehicle['make']} {vehicle['model']} ({vehicle['year']})",
        "city": incident_city,
        "road": rng.choice(ROADS),
        "cross_road": rng.choice(CROSS_ROADS),
        "direction": rng.choice(DIRECTIONS),
        "other_vehicle": _random_other_vehicle(rng),
        "traffic_control": rng.choice(TRAFFIC_CONTROLS),
        "impact_point": rng.choice(IMPACT_POINTS),
        "damage_desc": rng.choice(DAMAGE_DESCRIPTIONS),
        "damage_areas": rng.choice(DAMAGE_AREAS),
        "injury_text": rng.choice(INJURY_TEXTS.get(injury_severity, INJURY_TEXTS["none"])),
        "police_text": rng.choice(POLICE_TEXTS.get(police_report, POLICE_TEXTS[False])).format(
            report_num=report_num,
        ),
        "witness_text": rng.choice(WITNESS_TEXTS),
        "report_num": report_num,
        "driving_action": rng.choice([
            "proceeding normally", "slowing down for traffic",
            "turning left", "exiting a parking spot", "stopped at a red light",
        ]),
        "other_action": rng.choice([
            "ran a red light and hit me", "pulled out without looking",
            "changed lanes suddenly", "rear-ended me", "was speeding and lost control",
        ]),
        "cause": rng.choice(CAUSES_SINGLE_VEHICLE),
        "obstacle": rng.choice(OBSTACLES),
        "road_condition": rng.choice(ROAD_CONDITIONS),
        "weather_event": rng.choice(WEATHER_EVENTS),
        "parking_type": rng.choice(PARKING_TYPES),
        "location": rng.choice(LOCATIONS),
        "plate": vehicle["plate_number"],
        "destination": rng.choice(DESTINATIONS),
        "extra_damage": rng.choice(["broke my side mirror", "smashed the window", "slashed a tire"]),
        "extra_detail": rng.choice([
            "Several other cars in the area were also damaged.",
            "The municipality has declared a state of emergency.",
            "",
        ]),
        "water_level": rng.choice(["the doors", "the windows", "the hood"]),
        "time_parked": rng.choice(["at 20:00", "in the afternoon", "overnight"]),
        "claimed_extras": rng.choice(CLAIMED_EXTRAS_FRAUD),
        "inflated_amount": f"EUR {damage_estimate:,.2f}",
    }

    try:
        return template.template.format_map(_DefaultDict(fill)).strip()
    except Exception:
        text = template.template
        for key, value in fill.items():
            text = text.replace(f"{{{key}}}", str(value))
        return text.strip()


def _pick_templates(incident_type: str, is_fraud: bool, fraud_type: str | None):
    if is_fraud and fraud_type in ("staged", "inflated") and incident_type in FRAUD_TEMPLATES_BY_TYPE:
        return FRAUD_TEMPLATES_BY_TYPE[incident_type]
    if is_fraud and fraud_type == "phantom" and incident_type in FRAUD_TEMPLATES_BY_TYPE:
        return FRAUD_TEMPLATES_BY_TYPE.get(incident_type, TEMPLATES_BY_TYPE.get(incident_type, []))
    return TEMPLATES_BY_TYPE.get(incident_type, [])


def _random_other_vehicle(rng: np.random.Generator) -> str:
    spec = rng.choice(VEHICLE_CATALOG)
    return f"{spec['make']} {spec['model']}"


class _DefaultDict(dict):
    def __missing__(self, key: str) -> str:
        return f"[{key}]"
