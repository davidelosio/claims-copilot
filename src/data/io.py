from __future__ import annotations

from pathlib import Path

import pandas as pd

TABLE_ORDER = (
    "policyholders",
    "vehicles",
    "policies",
    "claims",
    "claim_labels",
    "claim_documents",
    "claim_events",
)


def to_csv(data: dict[str, list[dict]], output_dir: str = "data") -> None:
    """Write all generated tables to CSV files."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for table_name, rows in data.items():
        if not rows:
            continue
        path = out_dir / f"{table_name}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"  {table_name}: {len(rows)} rows written to {path}")


def to_postgres(data: dict[str, list[dict]], dsn: str) -> None:
    """Insert all generated tables into PostgreSQL."""
    import psycopg
    from psycopg.types.json import Json

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            for table_name in TABLE_ORDER:
                rows = data.get(table_name, [])
                if not rows:
                    continue
                cols = list(rows[0].keys())
                placeholders = ", ".join([f"%({c})s" for c in cols])
                col_names = ", ".join(cols)
                sql = f"INSERT INTO {table_name} ({col_names}) VALUES ({placeholders})"
                adapted_rows = []
                for row in rows:
                    adapted = {}
                    for key, value in row.items():
                        adapted[key] = Json(value) if isinstance(value, (dict, list)) else value
                    adapted_rows.append(adapted)
                cur.executemany(sql, adapted_rows)
                print(f"  {table_name}: {len(rows)} rows inserted")
        conn.commit()
