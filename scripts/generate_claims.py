#!/usr/bin/env python3
"""
Generate synthetic motor insurance claims data.

Usage:
    python scripts/generate_claims.py -n 5000 -o csv
    python scripts/generate_claims.py -n 5000 -o postgres -d "dbname=claims_copilot"
"""

from collections import Counter

import click

from src.data.generator import ClaimsGenerator


@click.command(context_settings={"help_option_names": ["-h"]})
@click.option("-n", "n_claims", type=int, default=5000, show_default=True, help="Number of claims to generate")
@click.option("-s", "seed", type=int, default=42, show_default=True, help="Random seed")
@click.option(
    "-o",
    "output",
    type=click.Choice(["csv", "postgres", "both"], case_sensitive=False),
    default="csv",
    show_default=True,
    help="Output format",
)
@click.option("-c", "csv_dir", default="data", show_default=True, help="CSV output directory")
@click.option("-d", "dsn", default="dbname=claims_copilot", show_default=True, help="PostgreSQL DSN")
def main(n_claims: int, seed: int, output: str, csv_dir: str, dsn: str) -> None:
    print(f"\nGenerating {n_claims} synthetic claims (seed={seed})...\n")

    gen = ClaimsGenerator(seed=seed)
    data = gen.generate(n_claims=n_claims)

    for table, rows in data.items():
        print(f"   {table}: {len(rows)} rows")

    if output in ("csv", "both"):
        print(f"\nWriting CSV to {csv_dir}/")
        gen.to_csv(data, output_dir=csv_dir)

    if output in ("postgres", "both"):
        print(f"\nWriting to PostgreSQL ({dsn})")
        gen.to_postgres(data, dsn=dsn)

    print("\nDone.\n")

    # Print some stats
    labels = data["claim_labels"]
    n_fraud = sum(1 for label in labels if label["is_fraud"])
    complexity_counts = Counter(label["complexity"] for label in labels)

    print("Dataset stats:")
    print(f"   Fraud rate: {n_fraud}/{len(labels)} ({100*n_fraud/len(labels):.1f}%)")
    print(f"   Complexity: {complexity_counts}")
    avg_days = sum(label["handling_days"] for label in labels) / len(labels)
    print(f"   Avg handling days: {avg_days:.1f}")


if __name__ == "__main__":
    main()
