"""
Synthetic claims data generator for Italian motor insurance.

Generates:
- Policyholders, vehicles, policies
- Claims with free-text descriptions
- Ground-truth labels (complexity, fraud, handling time)
- Realistic event sequences
- Document metadata (present / missing)

Usage:
    gen = ClaimsGenerator(seed=42)
    data = gen.generate(n_claims=5000)
    gen.to_postgres(data, dsn="dbname=claims_copilot")
    # or
    gen.to_csv(data, output_dir="data/")
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, time, timedelta

import numpy as np
from faker import Faker

from src.data.constants import (
    CITIES,
    COMPLEXITY_RULES,
    DOCUMENT_TYPES,
    EXPECTED_DOCS,
    FRAUD_RATE,
    FUEL_TYPES,
    FUEL_WEIGHTS,
    INCIDENT_TYPES,
    INCIDENT_WEIGHTS,
    POLICY_TYPES,
    POLICY_WEIGHTS,
    VEHICLE_CATALOG,
    VEHICLE_WEIGHTS,
)
from src.data.descriptions import generate_description
from src.data.events import generate_claim_events
from src.data.io import to_csv, to_postgres

fake = Faker("it_IT")


class ClaimsGenerator:
    """Generate synthetic motor insurance claims synthetic data."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        Faker.seed(seed)
        self._policyholder_cache: dict[str, dict] = {}
        self._vehicle_cache: dict[str, dict] = {}
        self._policy_cache: dict[str, dict] = {}
        # For repeat-fraud pattern: track some policyholders who will file many claims
        self._repeat_fraudsters: set[str] = set()

    def _weighted_choice(self, items: list[dict], weights: list[float]) -> dict:
        w = np.asarray(weights, dtype=float)
        w /= w.sum()
        idx = self.rng.choice(len(items), p=w)
        return items[idx]

    def generate(self, n_claims: int = 5000, claims_per_ph_ratio: float = 1.3) -> dict[str, list[dict]]:
        """Generate a full dataset.

        Returns dict with keys:
            policyholders, vehicles, policies, claims,
            claim_labels, claim_documents, claim_events
        """
        # Step 1: decide how many policyholders we need
        # avg ~1.3 claims per policyholder, some have more
        n_policyholders = int(n_claims / claims_per_ph_ratio)

        # Step 2: generate policyholders + vehicles + policies
        policyholders = [self._gen_policyholder() for _ in range(n_policyholders)]
        vehicles = []
        policies = []
        for ph in policyholders:
            v = self._gen_vehicle(ph["policyholder_id"])
            vehicles.append(v)
            p = self._gen_policy(ph["policyholder_id"], v["vehicle_id"])
            policies.append(p)

        # Step 3: pick some repeat fraudsters (~2% of policyholders)
        n_fraudsters = max(1, int(n_policyholders * 0.02))
        fraudster_ids = self.rng.choice(
            [ph["policyholder_id"] for ph in policyholders],
            size=n_fraudsters,
            replace=False,
        )
        self._repeat_fraudsters = set(fraudster_ids)

        # Step 4: generate claims
        claims = []
        claim_labels = []
        claim_documents = []
        claim_events = []

        policy_list = list(policies)  # all available policies

        for i in range(n_claims):
            # Pick a policy (with bias: repeat fraudsters get more claims)
            policy = self._pick_policy(policy_list)
            ph_id = policy["policyholder_id"]

            # Decide if this is a fraud claim
            if ph_id in self._repeat_fraudsters:
                is_fraud = self.rng.random() < 0.6  # 60% of their claims are repeated fraud
                fraud_type = "repeated" if is_fraud else None
            else:
                is_fraud = self.rng.random() < FRAUD_RATE
                fraud_type = self.rng.choice(["staged", "inflated", "phantom"]) if is_fraud else None

            # Generate claim
            claim, label, docs, events = self._gen_claim(
                policy=policy,
                claim_index=i,
                is_fraud=is_fraud,
                fraud_type=fraud_type,
            )

            claims.append(claim)
            claim_labels.append(label)
            claim_documents.extend(docs)
            claim_events.extend(events)

        return {
            "policyholders": policyholders,
            "vehicles": vehicles,
            "policies": policies,
            "claims": claims,
            "claim_labels": claim_labels,
            "claim_documents": claim_documents,
            "claim_events": claim_events,
        }

    def to_csv(self, data: dict[str, list[dict]], output_dir: str = "data") -> None:
        """Write all tables to CSV files."""
        to_csv(data, output_dir=output_dir)

    def to_postgres(self, data: dict[str, list[dict]], dsn: str) -> None:
        """Insert all data into PostgreSQL."""
        to_postgres(data, dsn=dsn)

    def _gen_policyholder(self) -> dict:
        gender = self.rng.choice(["M", "F"], p=[0.55, 0.45])
        if gender == "M":
            first = fake.first_name_male()
            last = fake.last_name_male()
        else:
            first = fake.first_name_female()
            last = fake.last_name_female()

        city_info = self._pick_city()
        dob_year = int(self.rng.integers(1955, 2004))
        license_year = max(dob_year + 18, int(self.rng.integers(dob_year + 18, min(dob_year + 30, 2024))))

        ph_id = f"PH-{uuid.uuid4().hex[:8].upper()}"
        ph = {
            "policyholder_id": ph_id,
            "first_name": first,
            "last_name": last,
            "date_of_birth": date(dob_year, int(self.rng.integers(1, 13)), int(self.rng.integers(1, 29))),
            "gender": gender,
            "city": city_info["city"],
            "province": city_info["province"],
            "driving_license_year": license_year,
        }
        self._policyholder_cache[ph_id] = ph
        return ph

    def _gen_vehicle(self, ph_id: str) -> dict:
        spec = self._weighted_choice(VEHICLE_CATALOG, VEHICLE_WEIGHTS)
        year = int(self.rng.integers(spec["min_year"], spec["max_year"] + 1))
        # Value depreciates with age
        base_val = self.rng.uniform(*spec["value"])
        age = 2024 - year
        depreciation = max(0.3, 1 - age * 0.07)
        value = round(base_val * depreciation, 2)

        fuel = self.rng.choice(FUEL_TYPES, p=FUEL_WEIGHTS)
        plate = fake.license_plate()

        v_id = f"VH-{uuid.uuid4().hex[:8].upper()}"
        v = {
            "vehicle_id": v_id,
            "policyholder_id": ph_id,
            "make": spec["make"],
            "model": spec["model"],
            "year": year,
            "fuel_type": fuel,
            "estimated_value": value,
            "plate_number": plate,
        }
        self._vehicle_cache[v_id] = v
        return v

    def _gen_policy(self, ph_id: str, v_id: str) -> dict:
        policy_type = self.rng.choice(POLICY_TYPES, p=POLICY_WEIGHTS)
        vehicle = self._vehicle_cache[v_id]

        # Premium depends on vehicle value, type, and some noise
        base_premium = vehicle["estimated_value"] * 0.03
        type_mult = {"third_party": 0.6, "fire_theft": 0.8, "comprehensive": 1.2}[policy_type]
        premium = round(base_premium * type_mult * self.rng.uniform(0.8, 1.3), 2)

        # Policy inception: random date in last 3 years
        inception = date(2024, 1, 1) - timedelta(days=int(self.rng.integers(0, 1095)))
        expiry = inception + timedelta(days=365)

        coverage_limit = round(vehicle["estimated_value"] * self.rng.uniform(2, 5), 2)
        deductible = round(self.rng.choice([0, 250, 500, 750, 1000]), 2)

        p_id = f"POL-{uuid.uuid4().hex[:8].upper()}"
        p = {
            "policy_id": p_id,
            "policyholder_id": ph_id,
            "vehicle_id": v_id,
            "policy_type": policy_type,
            "inception_date": inception,
            "expiry_date": expiry,
            "annual_premium": premium,
            "coverage_limit": coverage_limit,
            "deductible": deductible,
        }
        self._policy_cache[p_id] = p
        return p

    def _gen_claim(
        self,
        policy: dict,
        claim_index: int,
        is_fraud: bool,
        fraud_type: str | None,
    ) -> tuple[dict, dict, list[dict], list[dict]]:
        """Generate a single claim with labels, docs, and events."""
        claim_id = f"CLM-{claim_index:05d}"

        # Incident type
        incident_type = self.rng.choice(INCIDENT_TYPES, p=INCIDENT_WEIGHTS)
        # Fraud staged accidents are always collisions
        if is_fraud and fraud_type == "staged":
            incident_type = "collision"
        if is_fraud and fraud_type == "phantom":
            incident_type = self.rng.choice(["theft", "vandalism"])

        # Incident date: within policy period, biased toward recent
        policy_start = policy["inception_date"]
        policy_end = policy["expiry_date"]
        days_range = (policy_end - policy_start).days
        if is_fraud and fraud_type in ("staged", "phantom"):
            # Fraud: often shortly after inception (red flag)
            incident_offset = int(self.rng.integers(5, min(90, days_range)))
        else:
            incident_offset = int(self.rng.integers(0, days_range))
        incident_date = policy_start + timedelta(days=incident_offset)

        # Incident time (some patterns)
        if is_fraud and fraud_type == "staged":
            # Staged: often late night
            hour = int(self.rng.choice([22, 23, 0, 1, 2, 3]))
        else:
            # Normal: peaks at rush hours
            hour_weights = np.array([
                1, 0.5, 0.3, 0.2, 0.2, 0.3,   # 0-5
                1, 3, 5, 3, 2, 2,                # 6-11
                3, 3, 2, 2, 3, 5,                # 12-17
                5, 4, 3, 2, 1.5, 1,              # 18-23
            ], dtype=float)
            hour_weights /= hour_weights.sum()
            hour = int(self.rng.choice(24, p=hour_weights))
        minute = int(self.rng.integers(0, 60))
        incident_time = time(hour, minute)

        # Incident location
        if self.rng.random() < 0.7:
            # Same city as policyholder
            ph = self._policyholder_cache[policy["policyholder_id"]]
            incident_city = ph["city"]
            incident_province = ph["province"]
        else:
            city_info = self._pick_city()
            incident_city = city_info["city"]
            incident_province = city_info["province"]

        # Parties, injuries, police
        num_parties = self._gen_num_parties(incident_type)
        injuries, injury_severity = self._gen_injuries(incident_type)
        police_report = self._gen_police_report(incident_type, injuries)

        # Damage estimate
        vehicle = self._vehicle_cache[policy["vehicle_id"]]
        damage_estimate = self._gen_damage_estimate(
            incident_type, vehicle["estimated_value"], is_fraud, fraud_type,
        )

        # Free-text description
        description = generate_description(
            rng=self.rng,
            incident_type=incident_type,
            incident_date=incident_date,
            incident_time=incident_time,
            incident_city=incident_city,
            vehicle=vehicle,
            injury_severity=injury_severity,
            police_report=police_report,
            damage_estimate=damage_estimate,
            is_fraud=is_fraud,
            fraud_type=fraud_type,
        )

        # Complexity + handling time (ground truth)
        complexity, handling_days, num_touches = self._compute_complexity(
            incident_type, num_parties, injuries, injury_severity,
            police_report, damage_estimate, vehicle["estimated_value"],
            is_fraud,
        )

        # Status and timestamps
        created_at = datetime.combine(incident_date, time(9, 0)) + timedelta(
            days=int(self.rng.integers(0, 4)),
            hours=int(self.rng.integers(0, 10)),
        )
        first_contact_at = created_at + timedelta(
            hours=int(self.rng.integers(1, 48)),
        )
        settled_at = created_at + timedelta(days=handling_days)
        status = self.rng.choice(
            ["settled", "in_progress", "pending_docs", "denied"],
            p=[0.60, 0.15, 0.15, 0.10],
        )
        was_reopened = bool(self.rng.random() < 0.05)
        if was_reopened:
            status = "reopened"

        settled_amount = self._gen_settled_amount(
            damage_estimate, status, is_fraud,
        )

        # Handler assignment
        handler_queue = {
            "simple": "fast_lane",
            "medium": "standard",
            "complex": "specialist",
        }[complexity]
        if is_fraud and self.rng.random() < 0.3:
            handler_queue = "fraud_review"  # some fraud gets flagged
        handler_id = f"HDL-{int(self.rng.integers(1, 51)):03d}"

        # Build claim record
        claim = {
            "claim_id": claim_id,
            "policy_id": policy["policy_id"],
            "incident_date": incident_date,
            "incident_time": incident_time,
            "incident_city": incident_city,
            "incident_province": incident_province,
            "incident_type": incident_type,
            "description": description,
            "num_parties": num_parties,
            "injuries": injuries,
            "injury_severity": injury_severity,
            "police_report": police_report,
            "damage_estimate": damage_estimate,
            "status": status,
            "created_at": created_at,
            "first_contact_at": first_contact_at,
            "settled_at": settled_at if status in ("settled", "denied") else None,
            "assigned_handler": handler_id,
            "handler_queue": handler_queue,
        }

        # Ground truth labels
        label = {
            "claim_id": claim_id,
            "complexity": complexity,
            "handling_days": handling_days,
            "num_touches": num_touches,
            "is_fraud": is_fraud,
            "fraud_type": fraud_type,
            "settled_amount": settled_amount,
            "was_reopened": was_reopened,
        }

        # Documents
        docs = self._gen_documents(claim_id, incident_type, is_fraud)

        # Events
        events = generate_claim_events(
            rng=self.rng,
            claim_id=claim_id,
            created_at=created_at,
            handling_days=handling_days,
            status=status,
            handler_id=handler_id,
        )

        return claim, label, docs, events

    def _pick_city(self) -> dict:
        weights = [c["weight"] for c in CITIES]
        return self._weighted_choice(CITIES, weights)

    def _pick_policy(self, policies: list[dict]) -> dict:
        """Pick a policy, with bias toward repeat fraudsters."""
        if self._repeat_fraudsters and self.rng.random() < 0.15:
            # 15% chance to pick a fraudster's policy
            fraudster_policies = [
                p for p in policies
                if p["policyholder_id"] in self._repeat_fraudsters
            ]
            if fraudster_policies:
                return self.rng.choice(fraudster_policies)
        return self.rng.choice(policies)

    def _gen_num_parties(self, incident_type: str) -> int:
        if incident_type in ("theft", "vandalism", "weather", "parking"):
            return 1
        if incident_type == "collision":
            return int(self.rng.choice([2, 2, 2, 3, 3, 4]))
        if incident_type == "hit_and_run":
            return 2
        return 1  # single_vehicle

    def _gen_injuries(self, incident_type: str) -> tuple[bool, str]:
        if incident_type in ("theft", "vandalism", "weather", "parking"):
            return False, "none"
        # Probability of injury by type
        injury_prob = {
            "collision": 0.25,
            "single_vehicle": 0.30,
            "hit_and_run": 0.20,
        }.get(incident_type, 0.05)

        if self.rng.random() < injury_prob:
            severity = self.rng.choice(
                ["minor", "moderate", "severe"],
                p=[0.60, 0.30, 0.10],
            )
            return True, severity
        return False, "none"

    def _gen_police_report(self, incident_type: str, injuries: bool) -> bool:
        if incident_type in ("theft", "hit_and_run"):
            return True  # almost always
        if injuries:
            return self.rng.random() < 0.85
        return self.rng.random() < 0.30

    def _gen_damage_estimate(
        self,
        incident_type: str,
        vehicle_value: float,
        is_fraud: bool,
        fraud_type: str | None,
    ) -> float:
        if incident_type == "theft":
            return round(vehicle_value * self.rng.uniform(0.7, 1.0), 2)

        # Base damage as fraction of vehicle value
        base_fracs = {
            "collision": (0.05, 0.40),
            "single_vehicle": (0.05, 0.50),
            "hit_and_run": (0.03, 0.25),
            "parking": (0.01, 0.08),
            "vandalism": (0.01, 0.10),
            "weather": (0.02, 0.30),
        }
        lo, hi = base_fracs.get(incident_type, (0.05, 0.30))
        frac = self.rng.uniform(lo, hi)
        estimate = vehicle_value * frac

        # Fraud: inflated estimates
        if is_fraud and fraud_type == "inflated":
            estimate *= self.rng.uniform(1.8, 3.5)
        elif is_fraud and fraud_type == "staged":
            estimate *= self.rng.uniform(1.3, 2.0)

        return round(min(estimate, vehicle_value * 0.95), 2)

    def _gen_settled_amount(
        self, damage_estimate: float, status: str, is_fraud: bool,
    ) -> float | None:
        if status == "denied":
            return 0.0
        if status not in ("settled", "reopened"):
            return None
        if is_fraud and self.rng.random() < 0.3:
            # Some fraud gets partially paid before detection
            return round(damage_estimate * self.rng.uniform(0.2, 0.6), 2)
        # Normal: settled around the estimate minus deductible / negotiation
        return round(damage_estimate * self.rng.uniform(0.7, 1.0), 2)

    def _compute_complexity(
        self,
        incident_type: str,
        num_parties: int,
        injuries: bool,
        injury_severity: str,
        police_report: bool,
        damage_estimate: float,
        vehicle_value: float,
        is_fraud: bool,
    ) -> tuple[str, int, int]:
        """Compute complexity label, handling days, and number of touches."""
        score = 0.0
        score += COMPLEXITY_RULES["incident_type_base"].get(incident_type, 0)
        score += COMPLEXITY_RULES["injuries"].get(injury_severity, 0)
        party_key = min(num_parties, 3)
        score += COMPLEXITY_RULES["num_parties"].get(party_key, 2)
        score += COMPLEXITY_RULES["police_report"].get(police_report, 0)

        # High damage ratio adds complexity
        if vehicle_value > 0:
            damage_ratio = damage_estimate / vehicle_value
            if damage_ratio > 0.5:
                score += 1.5
            elif damage_ratio > 0.25:
                score += 0.5

        # Fraud cases tend to take longer
        if is_fraud:
            score += 1.0

        # Add noise
        score += self.rng.normal(0, 0.5)

        if score < 1.5:
            complexity = "simple"
            handling_days = max(1, int(self.rng.normal(5, 2)))
            num_touches = max(1, int(self.rng.normal(2, 1)))
        elif score < 3.5:
            complexity = "medium"
            handling_days = max(3, int(self.rng.normal(15, 5)))
            num_touches = max(2, int(self.rng.normal(5, 2)))
        else:
            complexity = "complex"
            handling_days = max(7, int(self.rng.normal(35, 12)))
            num_touches = max(4, int(self.rng.normal(10, 3)))

        return complexity, handling_days, num_touches

    def _gen_documents(
        self, claim_id: str, incident_type: str, is_fraud: bool,
    ) -> list[dict]:
        """Generate document metadata (which docs exist, which are missing)."""
        expected = EXPECTED_DOCS.get(incident_type, ["photo_damage"])
        docs = []

        for doc_type in expected:
            # Most docs are present, but some are missing → drives next-best-action
            if is_fraud:
                present = self.rng.random() < 0.6  # fraud claims often have missing docs
            else:
                present = self.rng.random() < 0.85

            docs.append({
                "document_id": f"DOC-{uuid.uuid4().hex[:8].upper()}",
                "claim_id": claim_id,
                "doc_type": doc_type,
                "present": present,
            })

        # Sometimes extra docs are uploaded
        if self.rng.random() < 0.2:
            extra = self.rng.choice([d for d in DOCUMENT_TYPES if d not in expected])
            docs.append({
                "document_id": f"DOC-{uuid.uuid4().hex[:8].upper()}",
                "claim_id": claim_id,
                "doc_type": extra,
                "present": True,
            })

        return docs
