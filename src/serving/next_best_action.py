"""
Next-Best-Action engine for claims handlers.

Takes:
- Claim data (structured fields)
- Model outputs (complexity, fraud score, handling days)
- LLM extraction output (missing info, extraction notes)
- Document metadata (present / missing)

Produces:
- Prioritized action checklist
- Suggested templates/messages
- Routing recommendation with justification

Architecture:
- v1: deterministic rules + model thresholds (ships fast, easy to audit)
- v2 (future): learn ranking from historical sequences
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ============================================================
# DATA STRUCTURES
# ============================================================

class ActionPriority(str, Enum):
    URGENT = "urgent"       # must do now
    HIGH = "high"           # do today
    MEDIUM = "medium"       # do this week
    LOW = "low"             # nice to have


class ActionCategory(str, Enum):
    DOCUMENT = "document"           # request / verify a document
    CONTACT = "contact"             # reach out to party
    VERIFICATION = "verification"   # check / validate something
    ESCALATION = "escalation"       # route to specialist / fraud
    INTERNAL = "internal"           # internal process step
    SETTLEMENT = "settlement"       # proceed toward resolution


@dataclass
class Action:
    """A single recommended action for the handler."""
    action: str                                  # what to do
    priority: ActionPriority                     # how urgent
    category: ActionCategory                     # type of action
    reason: str                                  # why (shown to handler)
    template_key: Optional[str] = None           # link to message template
    auto_completable: bool = False               # could be automated in v2


@dataclass
class NextBestActionResult:
    """Complete output of the NBA engine for one claim."""
    claim_id: str
    actions: list[Action] = field(default_factory=list)
    routing: Optional[RoutingRecommendation] = None
    summary_note: str = ""                       # one-liner for the handler


@dataclass
class RoutingRecommendation:
    """Where this claim should go."""
    queue: str                          # fast_lane / standard / specialist / fraud_review
    confidence: float                   # model confidence
    reasons: list[str] = field(default_factory=list)
    override_warning: Optional[str] = None  # if handler should consider overriding


# ============================================================
# DOCUMENT RULES
# ============================================================

# Which documents are required (must-have) vs recommended per incident type
REQUIRED_DOCS: dict[str, list[str]] = {
    "collision": ["photo_damage", "cai_form"],
    "single_vehicle": ["photo_damage"],
    "hit_and_run": ["photo_damage", "police_report"],
    "parking": ["photo_damage"],
    "theft": ["police_report", "id_document"],
    "vandalism": ["photo_damage", "police_report"],
    "weather": ["photo_damage"],
}

RECOMMENDED_DOCS: dict[str, list[str]] = {
    "collision": ["repair_estimate", "photo_scene", "witness_statement"],
    "single_vehicle": ["repair_estimate", "photo_scene"],
    "hit_and_run": ["repair_estimate", "witness_statement"],
    "parking": ["repair_estimate"],
    "theft": [],
    "vandalism": ["repair_estimate"],
    "weather": ["repair_estimate"],
}

INJURY_DOCS = ["medical_report"]

# ============================================================
# MESSAGE TEMPLATES
# ============================================================

TEMPLATES: dict[str, str] = {
    "request_photo_damage": (
        "Gentile cliente, per procedere con la gestione del sinistro {claim_id}, "
        "Le chiediamo di inviarci le fotografie dei danni al veicolo. "
        "Le foto devono mostrare chiaramente l'entità del danno."
    ),
    "request_police_report": (
        "Gentile cliente, per il sinistro {claim_id} è necessario il verbale "
        "delle forze dell'ordine. La preghiamo di inviarcelo al più presto."
    ),
    "request_repair_estimate": (
        "Gentile cliente, La preghiamo di inviarci un preventivo di riparazione "
        "da un'officina autorizzata per il sinistro {claim_id}."
    ),
    "request_cai_form": (
        "Gentile cliente, per il sinistro {claim_id} è necessaria la Constatazione "
        "Amichevole di Incidente (modulo CAI/CID) compilata e firmata da entrambe le parti."
    ),
    "request_medical_report": (
        "Gentile cliente, avendo segnalato lesioni nel sinistro {claim_id}, "
        "Le chiediamo di inviarci il certificato medico e la relativa documentazione sanitaria."
    ),
    "request_id_document": (
        "Gentile cliente, per completare la pratica del sinistro {claim_id}, "
        "Le chiediamo copia di un documento d'identità in corso di validità."
    ),
    "request_witness_statement": (
        "Gentile cliente, se sono presenti testimoni del sinistro {claim_id}, "
        "La preghiamo di fornirci i loro contatti e, se possibile, una dichiarazione scritta."
    ),
    "confirm_receipt": (
        "Gentile cliente, confermiamo la ricezione della Sua denuncia di sinistro {claim_id}. "
        "Un nostro operatore La contatterà entro {sla_hours} ore lavorative."
    ),
    "fraud_review_escalation": (
        "Sinistro {claim_id} segnalato per revisione antifrode. "
        "Anomalie rilevate: {reasons}. Verificare prima di procedere."
    ),
}

DOC_TYPE_TO_TEMPLATE: dict[str, str] = {
    "photo_damage": "request_photo_damage",
    "police_report": "request_police_report",
    "repair_estimate": "request_repair_estimate",
    "cai_form": "request_cai_form",
    "medical_report": "request_medical_report",
    "id_document": "request_id_document",
    "witness_statement": "request_witness_statement",
}

DOC_TYPE_LABELS: dict[str, str] = {
    "photo_damage": "Damage photos",
    "photo_scene": "Scene photos",
    "police_report": "Police report",
    "repair_estimate": "Repair estimate",
    "medical_report": "Medical report",
    "witness_statement": "Witness statement",
    "id_document": "ID document",
    "cai_form": "CAI form (joint statement)",
}


# ============================================================
# ENGINE
# ============================================================

class NextBestActionEngine:
    """Generate prioritized actions for a claim handler.

    Args:
        fraud_high_threshold: fraud score above this → escalate
        fraud_medium_threshold: fraud score above this → flag for review
        damage_high_threshold: damage ratio above this → require extra checks
    """

    def __init__(
        self,
        fraud_high_threshold: float = 0.5,
        fraud_medium_threshold: float = 0.2,
        damage_high_threshold: float = 0.5,
    ):
        self.fraud_high = fraud_high_threshold
        self.fraud_medium = fraud_medium_threshold
        self.damage_high = damage_high_threshold

    def generate(
        self,
        claim_id: str,
        claim: dict,
        model_output: dict,
        documents: list[dict],
        extraction: Optional[dict] = None,
    ) -> NextBestActionResult:
        """Generate next-best-actions for a claim.

        Args:
            claim_id: Claim identifier
            claim: Dict with claim fields (incident_type, injuries, etc.)
            model_output: Dict from ComplexityPredictor.predict()
            documents: List of doc dicts with {doc_type, present}
            extraction: Optional LLM extraction result dict
        """
        actions: list[Action] = []
        incident_type = claim.get("incident_type", "unknown")

        present_docs = _present_doc_types(documents)
        fraud_score = model_output.get("fraud_score", 0)

        self._add_document_actions(actions, claim, incident_type, present_docs)
        self._add_fraud_actions(actions, claim, fraud_score)
        self._add_extraction_actions(actions, extraction)

        complexity, confidence = self._add_complexity_actions(actions, model_output, fraud_score)
        self._add_contact_actions(actions, claim, incident_type, complexity)
        self._add_coverage_actions(actions, claim, incident_type)

        actions = _sort_actions(_deduplicate_actions(actions))
        routing = self._build_routing(model_output, fraud_score, complexity, confidence)
        summary = _build_summary_note(actions, fraud_score, self.fraud_medium)

        return NextBestActionResult(
            claim_id=claim_id,
            actions=actions,
            routing=routing,
            summary_note=summary,
        )

    def _add_document_actions(
        self,
        actions: list[Action],
        claim: dict,
        incident_type: str,
        present_docs: set[str],
    ) -> None:
        for doc_type in REQUIRED_DOCS.get(incident_type, []):
            if doc_type not in present_docs:
                actions.append(_build_document_action(doc_type, incident_type, required=True))

        for doc_type in RECOMMENDED_DOCS.get(incident_type, []):
            if doc_type not in present_docs:
                actions.append(_build_document_action(doc_type, incident_type, required=False))

        if claim.get("injuries") and "medical_report" not in present_docs:
            actions.append(Action(
                action="Request medical report",
                priority=ActionPriority.HIGH,
                category=ActionCategory.DOCUMENT,
                reason=f"Injuries reported ({claim.get('injury_severity', 'unknown')} severity)",
                template_key="request_medical_report",
            ))

    def _add_fraud_actions(
        self,
        actions: list[Action],
        claim: dict,
        fraud_score: float,
    ) -> None:
        if fraud_score >= self.fraud_high:
            actions.append(Action(
                action="Escalate to fraud review team",
                priority=ActionPriority.URGENT,
                category=ActionCategory.ESCALATION,
                reason=f"Fraud risk score is {fraud_score:.0%}",
                template_key="fraud_review_escalation",
            ))
        elif fraud_score >= self.fraud_medium:
            actions.append(Action(
                action="Flag for additional verification before settlement",
                priority=ActionPriority.HIGH,
                category=ActionCategory.VERIFICATION,
                reason=f"Fraud risk score is {fraud_score:.0%} — verify details before proceeding",
            ))

        damage_ratio = claim.get("damage_estimate", 0) / max(claim.get("vehicle_value", 1), 1)
        if damage_ratio > self.damage_high:
            actions.append(Action(
                action="Verify damage estimate with independent assessment",
                priority=ActionPriority.HIGH,
                category=ActionCategory.VERIFICATION,
                reason=f"Damage estimate is {damage_ratio:.0%} of vehicle value",
            ))

    def _add_extraction_actions(
        self,
        actions: list[Action],
        extraction: Optional[dict],
    ) -> None:
        if not extraction:
            return

        for missing in extraction.get("missing_info", []):
            actions.append(Action(
                action=f"Clarify: {missing.get('field', 'missing detail')}",
                priority=_priority_from_importance(missing.get("importance", "medium")),
                category=ActionCategory.CONTACT,
                reason=missing.get("reason", "Information needed for processing"),
            ))

        notes = extraction.get("extraction_notes")
        if notes:
            actions.append(Action(
                action="Review flagged inconsistencies in claim description",
                priority=ActionPriority.HIGH,
                category=ActionCategory.VERIFICATION,
                reason=f"LLM extraction flagged: {notes[:200]}",
            ))

    def _add_complexity_actions(
        self,
        actions: list[Action],
        model_output: dict,
        fraud_score: float,
    ) -> tuple[str, float]:
        complexity = model_output.get("complexity_label", "medium")
        confidence = model_output.get("complexity_confidence", 0)

        if complexity == "complex":
            actions.append(Action(
                action="Consider assigning to senior handler",
                priority=ActionPriority.HIGH,
                category=ActionCategory.INTERNAL,
                reason=(
                    f"Predicted complex claim "
                    f"(~{model_output.get('expected_handling_days', '?')} days expected)"
                ),
            ))

        if complexity == "simple" and confidence > 0.85 and fraud_score < self.fraud_medium:
            actions.append(Action(
                action="Eligible for fast-lane processing",
                priority=ActionPriority.MEDIUM,
                category=ActionCategory.INTERNAL,
                reason=f"High confidence simple claim ({confidence:.0%}), low fraud risk",
                auto_completable=True,
            ))

        return complexity, confidence

    def _add_contact_actions(
        self,
        actions: list[Action],
        claim: dict,
        incident_type: str,
        complexity: str,
    ) -> None:
        if claim.get("status") == "new":
            sla_hours = {"simple": 24, "medium": 12, "complex": 4}.get(complexity, 12)
            actions.append(Action(
                action="Send acknowledgment to policyholder",
                priority=ActionPriority.HIGH,
                category=ActionCategory.CONTACT,
                reason=f"New claim — SLA: contact within {sla_hours} hours",
                template_key="confirm_receipt",
                auto_completable=True,
            ))

        if claim.get("num_parties", 1) > 1 and incident_type == "collision":
            actions.append(Action(
                action="Contact other party/parties for their version",
                priority=ActionPriority.HIGH,
                category=ActionCategory.CONTACT,
                reason=f"{claim.get('num_parties')} parties involved in collision",
            ))

    def _add_coverage_actions(
        self,
        actions: list[Action],
        claim: dict,
        incident_type: str,
    ) -> None:
        if incident_type in ("theft", "weather", "vandalism") and claim.get("policy_type") == "third_party":
            actions.append(Action(
                action="Verify coverage — policy may not cover this incident type",
                priority=ActionPriority.URGENT,
                category=ActionCategory.VERIFICATION,
                reason=f"Third-party policy may not cover {incident_type}",
            ))

    def _build_routing(
        self,
        model_output: dict,
        fraud_score: float,
        complexity: str,
        confidence: float,
    ) -> RoutingRecommendation:
        queue = model_output.get("recommended_queue", "standard")
        routing_reasons = [
            f"Complexity: {complexity} ({confidence:.0%} confidence)",
            f"Expected handling: ~{model_output.get('expected_handling_days', '?')} days",
        ]

        if fraud_score >= self.fraud_high:
            queue = "fraud_review"
            routing_reasons.append(f"Fraud score: {fraud_score:.0%}")

        override_warning = None
        if confidence < 0.6:
            override_warning = "Low model confidence — handler should review routing decision"

        return RoutingRecommendation(
            queue=queue,
            confidence=confidence,
            reasons=routing_reasons,
            override_warning=override_warning,
        )

    def render_template(self, template_key: str, **kwargs) -> Optional[str]:
        """Render a message template with given values."""
        template = TEMPLATES.get(template_key)
        if template is None:
            return None
        try:
            return template.format(**kwargs)
        except KeyError:
            return template


def _deduplicate_actions(actions: list[Action]) -> list[Action]:
    """Remove duplicate actions, keeping highest priority."""
    seen: dict[str, Action] = {}
    priority_rank = {
        ActionPriority.URGENT: 0,
        ActionPriority.HIGH: 1,
        ActionPriority.MEDIUM: 2,
        ActionPriority.LOW: 3,
    }
    for a in actions:
        key = a.action.lower()
        if key not in seen or priority_rank[a.priority] < priority_rank[seen[key].priority]:
            seen[key] = a
    return list(seen.values())


def _present_doc_types(documents: list[dict]) -> set[str]:
    return {document["doc_type"] for document in documents if document.get("present")}


def _build_document_action(doc_type: str, incident_type: str, required: bool) -> Action:
    label = DOC_TYPE_LABELS.get(doc_type, doc_type)
    return Action(
        action=f"Request missing {label}" if required else f"Request {label}",
        priority=ActionPriority.URGENT if required else ActionPriority.MEDIUM,
        category=ActionCategory.DOCUMENT,
        reason=(
            f"{label} is required for {incident_type} claims"
            if required
            else f"{label} helps expedite {incident_type} claims"
        ),
        template_key=DOC_TYPE_TO_TEMPLATE.get(doc_type),
    )


def _priority_from_importance(importance: str) -> ActionPriority:
    return {
        "high": ActionPriority.HIGH,
        "medium": ActionPriority.MEDIUM,
        "low": ActionPriority.LOW,
    }.get(importance, ActionPriority.MEDIUM)


def _sort_actions(actions: list[Action]) -> list[Action]:
    priority_order = {
        ActionPriority.URGENT: 0,
        ActionPriority.HIGH: 1,
        ActionPriority.MEDIUM: 2,
        ActionPriority.LOW: 3,
    }
    return sorted(actions, key=lambda action: priority_order[action.priority])


def _build_summary_note(actions: list[Action], fraud_score: float, fraud_medium: float) -> str:
    n_urgent = sum(1 for action in actions if action.priority == ActionPriority.URGENT)
    n_high = sum(1 for action in actions if action.priority == ActionPriority.HIGH)
    summary = f"{len(actions)} actions ({n_urgent} urgent, {n_high} high priority)"
    if fraud_score >= fraud_medium:
        summary += f" | ⚠️ Fraud risk {fraud_score:.0%}"
    return summary
