from __future__ import annotations

from pathlib import Path

from src.serving.next_best_action import ActionCategory, ActionPriority

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
EXTRACTIONS_PATH = DATA_DIR / "extractions" / "extractions.json"

VIEW_OPTIONS = ["🔍 Claim Detail", "📊 Dashboard"]
DASHBOARD_VIEW = "📊 Dashboard"

PRIORITY_COLORS = {
    ActionPriority.URGENT: "#dc2626",
    ActionPriority.HIGH: "#ea580c",
    ActionPriority.MEDIUM: "#ca8a04",
    ActionPriority.LOW: "#65a30d",
}

PRIORITY_EMOJI = {
    ActionPriority.URGENT: "🔴",
    ActionPriority.HIGH: "🟠",
    ActionPriority.MEDIUM: "🟡",
    ActionPriority.LOW: "🟢",
}

CATEGORY_EMOJI = {
    ActionCategory.DOCUMENT: "📄",
    ActionCategory.CONTACT: "📞",
    ActionCategory.VERIFICATION: "🔍",
    ActionCategory.ESCALATION: "⚠️",
    ActionCategory.INTERNAL: "🏢",
    ActionCategory.SETTLEMENT: "💰",
}

QUEUE_CONFIG = {
    "fast_lane": {"label": "Fast Lane", "color": "#16a34a", "icon": "⚡"},
    "standard": {"label": "Standard", "color": "#2563eb", "icon": "📋"},
    "specialist": {"label": "Specialist", "color": "#9333ea", "icon": "🔧"},
    "fraud_review": {"label": "Fraud Review", "color": "#dc2626", "icon": "🚨"},
}

FACT_FIELDS = [
    ("Incident Date", "incident_date"),
    ("Incident Time", "incident_time"),
    ("Location", "incident_location"),
    ("City", "incident_city"),
    ("Type", "incident_type"),
    ("Parties", "num_parties"),
    ("Other Vehicle", "other_vehicle"),
    ("Damage", "damage_description"),
    ("Damage Areas", "damage_areas"),
    ("Injuries", "injuries_reported"),
    ("Injury Severity", "injury_severity"),
    ("Police Report", "police_report_mentioned"),
    ("Report Number", "police_report_number"),
    ("Witnesses", "witnesses_mentioned"),
]

CONFIDENCE_BADGES = {
    "high": "🟢",
    "medium": "🟡",
    "low": "🔴",
    "unknown": "⚫",
}

HEADER_STYLE = """
<style>
    .main-header {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 0.9rem;
        opacity: 0.6;
        margin-top: -0.5rem;
    }
    .metric-card {
        border: 1px solid rgba(128,128,128,0.2);
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.6;
    }
    .action-item {
        border-left: 3px solid;
        padding: 0.5rem 0.75rem;
        margin-bottom: 0.5rem;
        border-radius: 0 6px 6px 0;
        background: rgba(128,128,128,0.05);
    }
    .fact-row {
        display: flex;
        justify-content: space-between;
        padding: 0.3rem 0;
        border-bottom: 1px solid rgba(128,128,128,0.1);
    }
    .confidence-high { color: #16a34a; }
    .confidence-medium { color: #ca8a04; }
    .confidence-low { color: #dc2626; }
    .confidence-unknown { color: #6b7280; }
    .queue-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-weight: 600;
        font-size: 0.85rem;
        color: white;
    }
    .fraud-gauge {
        height: 8px;
        border-radius: 4px;
        background: linear-gradient(to right, #16a34a, #ca8a04, #dc2626);
        position: relative;
        margin: 0.5rem 0;
    }
    .fraud-marker {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: white;
        border: 2px solid #1f2937;
        position: absolute;
        top: -2px;
        transform: translateX(-50%);
    }
</style>
"""
