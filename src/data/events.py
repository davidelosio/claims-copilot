from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np


def generate_claim_events(
    rng: np.random.Generator,
    claim_id: str,
    created_at: datetime,
    handling_days: int,
    status: str,
    handler_id: str,
) -> list[dict]:
    """Generate a realistic event timeline for a claim."""
    events = []
    ts = created_at

    events.append({
        "claim_id": claim_id,
        "event_type": "created",
        "event_timestamp": ts,
        "event_data": None,
        "actor": "system",
    })

    ts += timedelta(hours=int(rng.integers(1, 5)))
    events.append({
        "claim_id": claim_id,
        "event_type": "assigned",
        "event_timestamp": ts,
        "event_data": {"handler_id": handler_id},
        "actor": "system",
    })

    n_intermediate = max(0, int(rng.normal(handling_days / 5, 1)))
    for _ in range(min(n_intermediate, 10)):
        ts += timedelta(
            days=int(rng.integers(1, max(2, handling_days // 4))),
            hours=int(rng.integers(0, 8)),
        )
        if ts > created_at + timedelta(days=handling_days):
            break
        event_type = rng.choice(
            ["doc_requested", "doc_uploaded", "contacted_customer", "note_added", "escalated"],
            p=[0.25, 0.25, 0.20, 0.20, 0.10],
        )
        events.append({
            "claim_id": claim_id,
            "event_type": event_type,
            "event_timestamp": ts,
            "event_data": None,
            "actor": handler_id,
        })

    final_ts = created_at + timedelta(days=handling_days)
    final_type = "settled" if status in ("settled", "reopened") else "denied"
    if status in ("in_progress", "pending_docs"):
        final_type = "note_added"
    events.append({
        "claim_id": claim_id,
        "event_type": final_type,
        "event_timestamp": final_ts,
        "event_data": None,
        "actor": handler_id,
    })

    return events
