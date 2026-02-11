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

import argparse
import json
import time
from pathlib import Path

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
    labels = pd.read_csv(labels_path).set_index("claim_id")
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


def main():
    parser = argparse.ArgumentParser(description="Run claim extraction pipeline")
    parser.add_argument("--csv-dir", default="data", help="Directory with generated CSVs")
    parser.add_argument("--n-sample", type=int, default=20, help="Number of claims to extract")
    parser.add_argument("--all", action="store_true", help="Extract all claims")
    parser.add_argument("--output-dir", default="data/extractions", help="Output directory")
    parser.add_argument("--eval", action="store_true", help="Run evaluation against ground truth")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Claude model to use")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API calls (s)")
    args = parser.parse_args()

    n = None if args.all else args.n_sample
    print(f"\n📋 Loading claims from {args.csv_dir}/ (sample={n or 'all'})...")
    claims = load_claims(args.csv_dir, n_sample=n)
    print(f"   Loaded {len(claims)} claims")

    print(f"\n🤖 Running extraction with {args.model}...")
    extractor = ClaimExtractor(model=args.model)
    t0 = time.time()
    results = extractor.extract_batch(claims, delay=args.delay)
    elapsed = time.time() - t0
    print(f"\n✅ Extracted {len(results)} claims in {elapsed:.1f}s ({elapsed/len(results):.1f}s/claim)")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extractions_raw = [r.model_dump(mode="json") for r in results]
    output_file = output_dir / "extractions.json"
    with open(output_file, "w") as f:
        json.dump(extractions_raw, f, indent=2, default=str)
    print(f"\n💾 Saved to {output_file}")

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
    if args.eval:
        print("\n" + "=" * 70)
        print("EVALUATION vs GROUND TRUTH")
        print("=" * 70)
        eval_results = evaluate(
            extractions_raw,
            labels_path=str(Path(args.csv_dir) / "claim_labels.csv"),
            claims_path=str(Path(args.csv_dir) / "claims.csv"),
        )
        print(f"\nTotal evaluated: {eval_results['total']}")
        print(f"Incident type accuracy: {eval_results.get('incident_type_accuracy_pct', 0)}%")
        print(f"Injury detection accuracy: {eval_results.get('injury_detection_accuracy_pct', 0)}%")
        print(f"Police report accuracy: {eval_results.get('police_report_accuracy_pct', 0)}%")
        print(f"City extraction accuracy: {eval_results.get('city_accuracy_pct', 0)}%")

        eval_file = output_dir / "eval_results.json"
        with open(eval_file, "w") as f:
            json.dump(eval_results, f, indent=2)
        print(f"\nSaved eval to {eval_file}")


if __name__ == "__main__":
    main()
