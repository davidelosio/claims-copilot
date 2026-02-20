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

from src.extraction.dataset import load_claims
from src.extraction.evaluation import evaluate
from src.extraction.pipeline import ClaimExtractor


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
@click.option("--timeout", type=float, default=240.0, show_default=True,
              help="Ollama request timeout (s)")
@click.option("--no-batch", is_flag=True,
              help="Force Ollama num_batch=1 for lower memory pressure")
def main(csv_dir: str, n_sample: int, extract_all: bool, output_dir: str,
         run_eval: bool, model: str, delay: float, timeout: float, no_batch: bool):
    n = None if extract_all else n_sample
    print(f"\nLoading claims from {csv_dir}/ (sample={n or 'all'})...")
    claims = load_claims(csv_dir, n_sample=n)
    print(f"   Loaded {len(claims)} claims")
    if not claims:
        print("\nNo claims to process. Exiting.")
        return

    print(f"\nRunning extraction with {model}...")
    num_batch = 1 if no_batch else None
    extractor = ClaimExtractor(model=model, timeout=timeout, num_batch=num_batch)
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
