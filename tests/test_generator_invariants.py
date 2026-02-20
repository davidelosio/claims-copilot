from __future__ import annotations

from src.data.generator import ClaimsGenerator


def test_generated_data_has_basic_relational_consistency():
    gen = ClaimsGenerator(seed=42)
    data = gen.generate(n_claims=10)

    expected_keys = {
        "policyholders",
        "vehicles",
        "policies",
        "claims",
        "claim_labels",
        "claim_documents",
        "claim_events",
    }
    assert set(data.keys()) == expected_keys
    assert len(data["claims"]) == 10
    assert len(data["claim_labels"]) == 10

    policy_ids = {p["policy_id"] for p in data["policies"]}
    claim_ids = {c["claim_id"] for c in data["claims"]}
    label_claim_ids = {l["claim_id"] for l in data["claim_labels"]}
    doc_claim_ids = {d["claim_id"] for d in data["claim_documents"]}
    event_claim_ids = {e["claim_id"] for e in data["claim_events"]}

    assert all(claim["policy_id"] in policy_ids for claim in data["claims"])
    assert label_claim_ids == claim_ids
    assert doc_claim_ids.issubset(claim_ids)
    assert event_claim_ids.issubset(claim_ids)
    assert len(data["claim_documents"]) > 0
    assert len(data["claim_events"]) > 0
