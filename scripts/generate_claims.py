#!/usr/bin/env python3
"""
Generate synthetic motor insurance claims data.

Usage:
    python scripts/generate_claims.py --n-claims 5000 --output csv
    python scripts/generate_claims.py --n-claims 5000 --output postgres --dsn "dbname=claims_copilot"
"""

import argparse
import time

from src.data.generator import ClaimsGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic claims data")
    parser.add_argument("--n-claims", type=int, default=5000, help="Number of claims to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        choices=["csv", "postgres", "both"],
        default="csv",
        help="Output format",
    )
    parser.add_argument("--csv-dir", default="data", help="CSV output directory")
    parser.add_argument("--dsn", default="dbname=claims_copilot", help="PostgreSQL DSN")
    args = parser.parse_args()

    print(f"\n🏗️  Generating {args.n_claims} synthetic claims (seed={args.seed})...\n")
    t0 = time.time()

    gen = ClaimsGenerator(seed=args.seed)
    data = gen.generate(n_claims=args.n_claims)

    elapsed = time.time() - t0
    print(f"\n✅ Generated in {elapsed:.1f}s:")
    for table, rows in data.items():
        print(f"   {table}: {len(rows)} rows")

    if args.output in ("csv", "both"):
        print(f"\n📁 Writing CSV to {args.csv_dir}/")
        gen.to_csv(data, output_dir=args.csv_dir)

    if args.output in ("postgres", "both"):
        print(f"\n🐘 Writing to PostgreSQL ({args.dsn})")
        gen.to_postgres(data, dsn=args.dsn)

    print("\n🎉 Done!\n")

    # Print some stats
    labels = data["claim_labels"]
    n_fraud = sum(1 for l in labels if l["is_fraud"])
    complexity_counts = {}
    for l in labels:
        complexity_counts[l["complexity"]] = complexity_counts.get(l["complexity"], 0) + 1

    print("📊 Dataset stats:")
    print(f"   Fraud rate: {n_fraud}/{len(labels)} ({100*n_fraud/len(labels):.1f}%)")
    print(f"   Complexity: {complexity_counts}")
    avg_days = sum(l["handling_days"] for l in labels) / len(labels)
    print(f"   Avg handling days: {avg_days:.1f}")


if __name__ == "__main__":
    main()
