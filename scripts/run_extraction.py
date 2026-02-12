#!/usr/bin/env python3
"""
Run LLM extraction on claims and optionally evaluate against ground truth.

Usage:
    # Extract a sample of 20 claims (for testing / cost control)
    uv run python scripts/run_extraction.py --n-sample 20 --output-dir data/extractions

    # Extract with evaluation against ground truth
    uv run python scripts/run_extraction.py --n-sample 50 --eval

    # Full run (all claims)
    uv run python scripts/run_extraction.py --all --output-dir data/extractions
"""

import json
from pathlib import Path

import click
import pandas as pd

from src.extraction.pipeline import ClaimExtractor


def load_claims(csv_dir: str, n_sample: int | None = None, seed: int = 42) -> list[dict]:
    """Load claims with policy context for extraction."""
    claims = pd.read_csv(Path(csv_dir) / "claims.csv")
    policies = pd.read_csv(Path(csv_dir) / "policies.csv")
    vehicles = pd.read_csv(Path(csv_dir) / "vehicles.csv")
    policyholders = pd.read_csv(Path(csv_dir) / "policyholders.csv")

    # Join for context
    merged = (
        claims
        .merge(policies[["policy_id", "policyholder_id", "vehicle_id", "policy_type"]], on="policy_id")
        .merge(vehicles[["vehicle_id", "make", "model", "year"]], on="vehicle_id")
        .merge(policyholders[["policyholder_id", "city"]], on="policyholder_id", suffixes=("", "_ph"))
    )

    if n_sample and n_sample < len(merged):
        merged = merged.sample(n=n_sample, random_state=seed)

    result = []
    for _, row in merged.iterrows():
        result.append({
            "claim_id": row["claim_id"],
            "description": row["description"],
            "policy_context": {
                "policy_type": row["policy_type"],
                "vehicle": {
                    "make": row["make"],
                    "model": row["model"],
                    "year": int(row["year"]),
                },
                "policyholder_city": row["city"],
            },
        })
    return result


def evaluate(extractions: list[dict], labels_path: str, claims_path: str) -> dict:
    """Compare extractions against ground truth labels and structured claim fields."""
    claims = pd.read_csv(claims_path).set_index("claim_id")

    results = {
        "incident_type_accuracy": 0,
        "injury_detection_accuracy": 0,
        "police_report_accuracy": 0,
        "city_accuracy": 0,
        "total": 0,
        "details": [],
    }

    for ext in extractions:
        cid = ext["claim_id"]
        if cid not in claims.index:
            continue

        gt_claim = claims.loc[cid]
        results["total"] += 1
        detail = {"claim_id": cid}

        # Incident type match
        gt_type = gt_claim.get("incident_type", "")
        pred_type = ext.get("facts", {}).get("incident_type", "unknown")
        type_match = gt_type == pred_type
        results["incident_type_accuracy"] += int(type_match)
        detail["incident_type"] = {"gt": gt_type, "pred": pred_type, "match": type_match}

        # Injury detection
        gt_injuries = gt_claim.get("injuries", False)
        pred_injuries = ext.get("facts", {}).get("injuries_reported", False)
        inj_match = bool(gt_injuries) == bool(pred_injuries)
        results["injury_detection_accuracy"] += int(inj_match)
        detail["injuries"] = {"gt": bool(gt_injuries), "pred": pred_injuries, "match": inj_match}

        # Police report
        gt_police = gt_claim.get("police_report", False)
        pred_police_field = ext.get("facts", {}).get("police_report_mentioned", {})
        pred_police_val = pred_police_field.get("value", "")
        pred_police = pred_police_val is not None and "yes" in str(pred_police_val).lower() or pred_police_val == "true"
        police_match = bool(gt_police) == pred_police
        results["police_report_accuracy"] += int(police_match)
        detail["police_report"] = {"gt": bool(gt_police), "pred": pred_police, "match": police_match}

        # City
        gt_city = str(gt_claim.get("incident_city", "")).lower().strip()
        pred_city_field = ext.get("facts", {}).get("incident_city", {})
        pred_city = str(pred_city_field.get("value", "")).lower().strip()
        city_match = gt_city in pred_city or pred_city in gt_city
        results["city_accuracy"] += int(city_match)
        detail["city"] = {"gt": gt_city, "pred": pred_city, "match": city_match}

        results["details"].append(detail)

    n = results["total"]
    if n > 0:
        for key in ["incident_type_accuracy", "injury_detection_accuracy",
                     "police_report_accuracy", "city_accuracy"]:
            results[f"{key}_pct"] = round(100 * results[key] / n, 1)

    return results


@click.command(help="Run claim extraction pipeline")
@click.option("--csv-dir", default="data", type=click.Path(file_okay=False), show_default=True,
              help="Directory with generated CSVs")
@click.option("--n-sample", type=int, default=20, show_default=True,
              help="Number of claims to extract")
@click.option("--all", "extract_all", is_flag=True, help="Extract all claims")
@click.option("--output-dir", default="data/extractions", type=click.Path(file_okay=False),
              show_default=True, help="Output directory")
@click.option("--eval", "run_eval", is_flag=True, help="Run evaluation against ground truth")
@click.option("--model", default="mistral:7b-instruct-q4_K_M", show_default=True,
              help="Model to use")
@click.option("--delay", type=float, default=0.5, show_default=True,
              help="Delay between API calls (s)")
@click.option("--timeout", type=float, default=120.0, show_default=True,
              help="Ollama request timeout (s)")
def main(csv_dir: str, n_sample: int, extract_all: bool, output_dir: str,
         run_eval: bool, model: str, delay: float, timeout: float):
    n = None if extract_all else n_sample
    print(f"\nLoading claims from {csv_dir}/ (sample={n or 'all'})...")
    claims = load_claims(csv_dir, n_sample=n)
    print(f"   Loaded {len(claims)} claims")

    print(f"\nRunning extraction with {model}...")
    extractor = ClaimExtractor(model=model, timeout=timeout)
    results = extractor.extract_batch(claims, delay=delay)

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    extractions_raw = [r.model_dump(mode="json") for r in results]
    output_file = output_path / "extractions.json"
    with open(output_file, "w") as f:
        json.dump(extractions_raw, f, indent=2, default=str)
    print(f"\nSaved to {output_file}")

    # Show a few examples
    print("\n" + "=" * 70)
    print("SAMPLE EXTRACTIONS")
    print("=" * 70)
    for r in results[:3]:
        print(f"\n--- {r.claim_id} ---")
        print(f"Summary: {r.summary}")
        print(f"Type: {r.facts.incident_type.value}")
        print(f"City: {r.facts.incident_city.value} ({r.facts.incident_city.confidence.value})")
        print(f"Injuries: {r.facts.injuries_reported} ({r.facts.injury_severity.value})")
        if r.missing_info:
            print(f"Missing: {', '.join(m.field for m in r.missing_info)}")
        if r.extraction_notes:
            print(f"Notes: {r.extraction_notes}")

    # Evaluation
    if run_eval:
        print("\n" + "=" * 70)
        print("EVALUATION vs GROUND TRUTH")
        print("=" * 70)
        eval_results = evaluate(
            extractions_raw,
            labels_path=str(Path(csv_dir) / "claim_labels.csv"),
            claims_path=str(Path(csv_dir) / "claims.csv"),
        )
        print(f"\nTotal evaluated: {eval_results['total']}")
        print(f"Incident type accuracy: {eval_results.get('incident_type_accuracy_pct', 0)}%")
        print(f"Injury detection accuracy: {eval_results.get('injury_detection_accuracy_pct', 0)}%")
        print(f"Police report accuracy: {eval_results.get('police_report_accuracy_pct', 0)}%")
        print(f"City extraction accuracy: {eval_results.get('city_accuracy_pct', 0)}%")

        eval_file = output_path / "eval_results.json"
        with open(eval_file, "w") as f:
            json.dump(eval_results, f, indent=2)
        print(f"\nSaved eval to {eval_file}")


if __name__ == "__main__":
    main()
