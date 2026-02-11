"""
Templates for generating realistic free-text claim descriptions.

Each template has placeholders that get filled by the generator.
Templates vary in:
- detail level (sparse → detailed)
- writing quality (formal → messy)
- completeness (some omit key facts)

For fraud cases, we have special templates with subtle inconsistencies.
"""

from typing import NamedTuple


class DescriptionTemplate(NamedTuple):
    template: str
    detail_level: str  # sparse / normal / detailed
    needs: list[str]   # which placeholders it requires


# ============================================================
# COLLISION templates
# ============================================================

COLLISION_TEMPLATES = [
    # Detailed, well-written
    DescriptionTemplate(
        "On {date} at approximately {time}, I was driving my {vehicle} along {road} "
        "in {city} heading towards {direction}. At the intersection with {cross_road}, "
        "the other vehicle ({other_vehicle}) failed to yield at the {traffic_control} "
        "and hit {impact_point}. The impact caused {damage_desc}. "
        "{injury_text} {witness_text} {police_text}",
        "detailed",
        ["date", "time", "vehicle", "road", "city", "direction", "cross_road",
         "other_vehicle", "traffic_control", "impact_point", "damage_desc",
         "injury_text", "witness_text", "police_text"],
    ),
    DescriptionTemplate(
        "Collision occurred on {date} around {time} on {road}, {city}. "
        "I was {driving_action} when {other_vehicle} {other_action}. "
        "Damage to {damage_areas}. {damage_desc}. "
        "{injury_text} {police_text}",
        "normal",
        ["date", "time", "road", "city", "driving_action", "other_vehicle",
         "other_action", "damage_areas", "damage_desc", "injury_text", "police_text"],
    ),
    # Sparse
    DescriptionTemplate(
        "Car accident on {date} in {city}. Other car hit me. {damage_desc}.",
        "sparse",
        ["date", "city", "damage_desc"],
    ),
    DescriptionTemplate(
        "Accident {date}, {road} {city}. The other driver rear-ended me at the traffic light. "
        "Bumper and taillight damaged. {police_text}",
        "normal",
        ["date", "road", "city", "police_text"],
    ),
    # Messy / informal
    DescriptionTemplate(
        "hi so basically on {date} i was driving on {road} and this guy just came out of "
        "nowhere and hit my car on the {impact_point}.. the damage is pretty bad, "
        "{damage_desc}. {injury_text} i have photos",
        "normal",
        ["date", "road", "impact_point", "damage_desc", "injury_text"],
    ),
    DescriptionTemplate(
        "Incidente del {date} ore {time} circa. Stavo percorrendo {road} a {city} "
        "quando un veicolo {other_vehicle} mi ha tamponato al semaforo. "
        "Danni: {damage_desc}. {injury_text} {police_text}",
        "detailed",
        ["date", "time", "road", "city", "other_vehicle", "damage_desc",
         "injury_text", "police_text"],
    ),
]

# ============================================================
# SINGLE VEHICLE templates
# ============================================================

SINGLE_VEHICLE_TEMPLATES = [
    DescriptionTemplate(
        "On {date} at {time}, while driving on {road} near {city}, I lost control "
        "of my {vehicle} due to {cause}. The car hit {obstacle}. {damage_desc}. "
        "{injury_text} {police_text}",
        "detailed",
        ["date", "time", "road", "city", "vehicle", "cause", "obstacle",
         "damage_desc", "injury_text", "police_text"],
    ),
    DescriptionTemplate(
        "Single car incident, {date}, {road}. {cause}. Hit {obstacle}. {damage_desc}.",
        "sparse",
        ["date", "road", "cause", "obstacle", "damage_desc"],
    ),
    DescriptionTemplate(
        "Driving home on {date} around {time}, the road was {road_condition} and "
        "I skidded into {obstacle}. {damage_desc}. {injury_text}",
        "normal",
        ["date", "time", "road_condition", "obstacle", "damage_desc", "injury_text"],
    ),
]

# ============================================================
# HIT AND RUN templates
# ============================================================

HIT_AND_RUN_TEMPLATES = [
    DescriptionTemplate(
        "On {date} at {time}, my {vehicle} was hit by an unknown vehicle on {road} "
        "in {city}. The other driver did not stop. {witness_text} {damage_desc}. "
        "I filed a police report. {police_text}",
        "detailed",
        ["date", "time", "vehicle", "road", "city", "witness_text",
         "damage_desc", "police_text"],
    ),
    DescriptionTemplate(
        "Hit and run, {date}, {city}. Someone hit my car and drove off. "
        "Damage to {damage_areas}. {police_text}",
        "sparse",
        ["date", "city", "damage_areas", "police_text"],
    ),
    DescriptionTemplate(
        "I was at a red light on {road} when someone rear-ended me and immediately "
        "drove away ({date}, around {time}). I couldn't get the plate number. "
        "{damage_desc}. {injury_text}",
        "normal",
        ["road", "date", "time", "damage_desc", "injury_text"],
    ),
]

# ============================================================
# PARKING templates
# ============================================================

PARKING_TEMPLATES = [
    DescriptionTemplate(
        "Found my {vehicle} damaged on {date} in the parking {parking_type} "
        "near {location}, {city}. {damage_desc}. No note left by the responsible party.",
        "normal",
        ["vehicle", "date", "parking_type", "location", "city", "damage_desc"],
    ),
    DescriptionTemplate(
        "Car was parked on {road}, {city}. Came back on {date} and found {damage_desc}. "
        "No witnesses.",
        "sparse",
        ["road", "city", "date", "damage_desc"],
    ),
]

# ============================================================
# THEFT templates
# ============================================================

THEFT_TEMPLATES = [
    DescriptionTemplate(
        "My {vehicle} (plate {plate}) was stolen on {date} from {location}, {city}. "
        "I noticed it was missing around {time}. I have filed a police report "
        "(report number {report_num}). The car was locked and the alarm was active.",
        "detailed",
        ["vehicle", "plate", "date", "location", "city", "time", "report_num"],
    ),
    DescriptionTemplate(
        "Vehicle stolen, {date}, {city}. Plate: {plate}. Police report filed. {police_text}",
        "sparse",
        ["date", "city", "plate", "police_text"],
    ),
]

# ============================================================
# VANDALISM templates
# ============================================================

VANDALISM_TEMPLATES = [
    DescriptionTemplate(
        "On {date}, I found my {vehicle} vandalized while parked on {road}, {city}. "
        "{damage_desc}. {police_text}",
        "normal",
        ["date", "vehicle", "road", "city", "damage_desc", "police_text"],
    ),
    DescriptionTemplate(
        "Someone keyed my car and {extra_damage} on {date}. Parked near {location}, "
        "{city}. {police_text}",
        "sparse",
        ["extra_damage", "date", "location", "city", "police_text"],
    ),
]

# ============================================================
# WEATHER templates
# ============================================================

WEATHER_TEMPLATES = [
    DescriptionTemplate(
        "During the {weather_event} on {date}, my {vehicle} parked on {road} in {city} "
        "suffered {damage_desc}. {extra_detail}",
        "normal",
        ["weather_event", "date", "vehicle", "road", "city", "damage_desc", "extra_detail"],
    ),
    DescriptionTemplate(
        "Hailstorm on {date}, {city}. Multiple dents on roof and hood of my {vehicle}. "
        "{damage_desc}.",
        "sparse",
        ["date", "city", "vehicle", "damage_desc"],
    ),
    DescriptionTemplate(
        "On {date} there was severe flooding on {road}, {city}. My {vehicle} was "
        "submerged up to {water_level}. Engine and electronics damaged. {damage_desc}.",
        "detailed",
        ["date", "road", "city", "vehicle", "water_level", "damage_desc"],
    ),
]

# ============================================================
# FRAUD-SPECIFIC templates (with subtle inconsistencies)
# ============================================================

FRAUD_COLLISION_TEMPLATES = [
    # Vague on details, very specific on damage amount
    DescriptionTemplate(
        "Accident on {date}. The other car hit me somewhere on {road}. "
        "I think it was around {time} but I'm not sure. The mechanic says repairs "
        "will cost exactly {inflated_amount}. {damage_desc}.",
        "normal",
        ["date", "road", "time", "inflated_amount", "damage_desc"],
    ),
    # Date/time inconsistency potential
    DescriptionTemplate(
        "On {date} at {time} I was driving to {destination}. The collision happened "
        "on {road}, {city}. The other driver and I exchanged information. "
        "My car has extensive damage: {damage_desc}. Repair estimate attached.",
        "detailed",
        ["date", "time", "destination", "road", "city", "damage_desc"],
    ),
]

FRAUD_THEFT_TEMPLATES = [
    DescriptionTemplate(
        "My {vehicle} disappeared from {location}, {city} on {date}. "
        "I'm sure I parked it there {time_parked}. "
        "It had {claimed_extras} installed. Police report filed.",
        "normal",
        ["vehicle", "location", "city", "date", "time_parked", "claimed_extras"],
    ),
]

# ============================================================
# TEMPLATE REGISTRY
# ============================================================

TEMPLATES_BY_TYPE: dict[str, list[DescriptionTemplate]] = {
    "collision": COLLISION_TEMPLATES,
    "single_vehicle": SINGLE_VEHICLE_TEMPLATES,
    "hit_and_run": HIT_AND_RUN_TEMPLATES,
    "parking": PARKING_TEMPLATES,
    "theft": THEFT_TEMPLATES,
    "vandalism": VANDALISM_TEMPLATES,
    "weather": WEATHER_TEMPLATES,
}

FRAUD_TEMPLATES_BY_TYPE: dict[str, list[DescriptionTemplate]] = {
    "collision": FRAUD_COLLISION_TEMPLATES,
    "theft": FRAUD_THEFT_TEMPLATES,
}

# ============================================================
# FILL VALUES (used by the generator to populate placeholders)
# ============================================================

ROADS = [
    "Via Roma", "Via Garibaldi", "Via Mazzini", "Corso Italia", "Via Dante",
    "Via Nazionale", "Viale Europa", "Via XX Settembre", "Via Cavour", "Via Verdi",
    "Via Marconi", "Corso Vittorio Emanuele", "Via della Repubblica", "Via Gramsci",
    "Via Matteotti", "Viale della Libertà", "Via Togliatti", "Via Kennedy",
    "Tangenziale Est", "Tangenziale Ovest", "Via Emilia", "SS36", "A4 Milano-Venezia",
    "A1 del Sole", "Via Cristoforo Colombo", "Viale Monza",
]

CROSS_ROADS = [
    "Via Verdi", "Via Manzoni", "Piazza Garibaldi", "Via dei Mille",
    "Viale Trento", "Via Pascoli", "Via Leopardi", "Corso Buenos Aires",
]

DIRECTIONS = [
    "the city center", "the highway", "home", "work", "the hospital",
    "north", "south", "the station",
]

TRAFFIC_CONTROLS = [
    "stop sign", "traffic light", "yield sign", "roundabout", "uncontrolled intersection",
]

IMPACT_POINTS = [
    "the front left side", "the rear bumper", "the passenger side",
    "the driver side door", "the front right fender", "my rear left quarter panel",
]

DAMAGE_DESCRIPTIONS = [
    "dented bumper and scratched paint",
    "broken headlight and crumpled fender",
    "significant body damage on the driver side",
    "rear bumper completely detached, taillight broken",
    "deep scratches along the entire right side",
    "front bumper pushed in, radiator possibly damaged",
    "minor dent on the door panel",
    "cracked windshield and roof dent",
    "hood and grille severely damaged",
    "wheel arch crushed, tire blown",
]

DAMAGE_AREAS = [
    "front bumper and hood", "rear bumper", "left side", "right side",
    "driver door", "passenger side", "roof and hood", "all panels",
]

OBSTACLES = [
    "a guardrail", "a tree", "a concrete barrier", "a parked car",
    "a ditch", "a road sign", "a wall", "a lamp post",
]

CAUSES_SINGLE_VEHICLE = [
    "wet road conditions", "a tire blowout", "an animal crossing the road",
    "poor visibility due to fog", "ice on the road", "a sudden swerve to avoid debris",
    "distraction", "mechanical failure",
]

ROAD_CONDITIONS = [
    "very wet from rain", "covered in ice", "slippery due to oil",
    "flooded", "foggy with poor visibility",
]

WEATHER_EVENTS = [
    "severe hailstorm", "flooding", "storm with strong winds",
    "heavy snowfall", "tornado-like winds",
]

PARKING_TYPES = ["lot", "area", "garage", "street"]

LOCATIONS = [
    "the train station", "the shopping center", "my workplace",
    "the supermarket", "Piazza del Duomo", "the hospital",
    "my apartment building", "the gym", "the airport parking",
]

CLAIMED_EXTRAS_FRAUD = [
    "a premium sound system and custom alloy wheels",
    "aftermarket navigation and leather seats",
    "recently installed new tires and a dashcam",
    "sport suspension and performance exhaust",
]

DESTINATIONS = [
    "work", "the gym", "a friend's house", "the supermarket",
    "my parents' place", "the airport", "a client meeting",
]

INJURY_TEXTS = {
    "none": [
        "No injuries.",
        "Fortunately, no one was injured.",
        "",
    ],
    "minor": [
        "I experienced some neck pain afterwards.",
        "Minor whiplash, went to the ER as a precaution.",
        "Slight pain in my back, will see a doctor.",
        "The other driver complained of neck pain.",
    ],
    "moderate": [
        "I was taken to the hospital with back pain and a sprained wrist.",
        "The passenger in the other car was injured and taken by ambulance.",
        "I have a fractured rib from the impact.",
    ],
    "severe": [
        "Multiple injuries, I was hospitalized for two days.",
        "Serious injuries to my leg, currently in recovery.",
        "The other driver was severely injured and airlifted.",
    ],
}

POLICE_TEXTS = {
    True: [
        "Police were called to the scene.",
        "I filed a report at the local police station.",
        "The Carabinieri arrived and filed a report.",
        "Report number {report_num} was filed.",
    ],
    False: [
        "We did not call the police.",
        "No police report was filed.",
        "",
    ],
}

WITNESS_TEXTS = [
    "A pedestrian witnessed the accident and left their contact info.",
    "No witnesses were present.",
    "There were several witnesses at the scene.",
    "",
    "",  # often no mention
]
