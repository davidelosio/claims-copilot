from __future__ import annotations

import pandas as pd

try:
    from sklearn.metrics import accuracy_score as _sk_accuracy_score
except ImportError:
    _sk_accuracy_score = None


def _coerce_yes_no(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"yes", "true", "1"}


def _norm_text(value: object) -> str:
    return str(value or "").strip().lower()


def _city_match(gt_city: str, pred_city: str) -> bool:
    return gt_city in pred_city or pred_city in gt_city


def _accuracy_score(y_true: list, y_pred: list) -> float:
    if not y_true:
        return 0.0
    if _sk_accuracy_score is not None:
        return float(_sk_accuracy_score(y_true, y_pred))
    matches = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    return matches / len(y_true)


def evaluate(extractions: list[dict], claims_path: str) -> dict:
    """Compare extracted facts against structured claim ground truth."""
    claims = pd.read_csv(claims_path).set_index("claim_id")
    rows: list[dict] = []

    for ext in extractions:
        cid = ext["claim_id"]
        if cid not in claims.index:
            continue

        gt_claim = claims.loc[cid]
        gt_type = gt_claim.get("incident_type", "")
        pred_type = ext.get("facts", {}).get("incident_type", "unknown")

        gt_injuries = gt_claim.get("injuries", False)
        pred_injuries = ext.get("facts", {}).get("injuries_reported", False)

        gt_police = gt_claim.get("police_report", False)
        pred_police_field = ext.get("facts", {}).get("police_report_mentioned", {})
        pred_police = _coerce_yes_no(pred_police_field.get("value"))

        gt_city = _norm_text(gt_claim.get("incident_city", ""))
        pred_city_field = ext.get("facts", {}).get("incident_city", {})
        pred_city = _norm_text(pred_city_field.get("value", ""))
        rows.append({
            "claim_id": cid,
            "gt_type": gt_type,
            "pred_type": pred_type,
            "gt_injuries": bool(gt_injuries),
            "pred_injuries": bool(pred_injuries),
            "gt_police": bool(gt_police),
            "pred_police": pred_police,
            "gt_city": gt_city,
            "pred_city": pred_city,
        })

    if not rows:
        return {
            "incident_type_accuracy": 0,
            "injury_detection_accuracy": 0,
            "police_report_accuracy": 0,
            "city_accuracy": 0,
            "total": 0,
            "details": [],
        }

    frame = pd.DataFrame(rows)
    frame["type_match"] = frame["gt_type"] == frame["pred_type"]
    frame["injuries_match"] = frame["gt_injuries"] == frame["pred_injuries"]
    frame["police_match"] = frame["gt_police"] == frame["pred_police"]
    frame["city_match"] = [
        _city_match(gt_city, pred_city)
        for gt_city, pred_city in zip(frame["gt_city"], frame["pred_city"])
    ]

    total = len(frame)
    incident_type_acc = _accuracy_score(frame["gt_type"].tolist(), frame["pred_type"].tolist())
    injuries_acc = _accuracy_score(frame["gt_injuries"].tolist(), frame["pred_injuries"].tolist())
    police_acc = _accuracy_score(frame["gt_police"].tolist(), frame["pred_police"].tolist())
    city_acc = _accuracy_score([True] * total, frame["city_match"].tolist())

    details = []
    for row in frame.itertuples(index=False):
        details.append({
            "claim_id": row.claim_id,
            "incident_type": {"gt": row.gt_type, "pred": row.pred_type, "match": bool(row.type_match)},
            "injuries": {
                "gt": bool(row.gt_injuries),
                "pred": bool(row.pred_injuries),
                "match": bool(row.injuries_match),
            },
            "police_report": {
                "gt": bool(row.gt_police),
                "pred": bool(row.pred_police),
                "match": bool(row.police_match),
            },
            "city": {"gt": row.gt_city, "pred": row.pred_city, "match": bool(row.city_match)},
        })

    results = {
        "incident_type_accuracy": int(frame["type_match"].sum()),
        "injury_detection_accuracy": int(frame["injuries_match"].sum()),
        "police_report_accuracy": int(frame["police_match"].sum()),
        "city_accuracy": int(frame["city_match"].sum()),
        "total": total,
        "details": details,
        "incident_type_accuracy_pct": round(100 * incident_type_acc, 1),
        "injury_detection_accuracy_pct": round(100 * injuries_acc, 1),
        "police_report_accuracy_pct": round(100 * police_acc, 1),
        "city_accuracy_pct": round(100 * city_acc, 1),
    }
    return results
