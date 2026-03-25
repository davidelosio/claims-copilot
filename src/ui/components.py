from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from src.serving.next_best_action import TEMPLATES
from src.ui.constants import (
    CATEGORY_EMOJI,
    CONFIDENCE_BADGES,
    FACT_FIELDS,
    HEADER_STYLE,
    PRIORITY_COLORS,
    PRIORITY_EMOJI,
    QUEUE_CONFIG,
)
from src.ui.data import load_metrics
from src.ui.logic import collect_fraud_signals, filter_claims


def render_header() -> None:
    st.markdown(HEADER_STYLE, unsafe_allow_html=True)
    st.markdown('<p class="main-header">🛡️ Claims Copilot</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered claims operations assistant</p>', unsafe_allow_html=True)


def render_claim_selector(df: pd.DataFrame) -> Optional[str]:
    """Sidebar claim selector with filters."""
    st.sidebar.markdown("### 📂 Select Claim")

    incident_filter = st.sidebar.multiselect(
        "Incident type",
        options=sorted(df["incident_type"].unique()),
        default=[],
    )
    complexity_filter = st.sidebar.multiselect(
        "Complexity",
        options=["simple", "medium", "complex"],
        default=[],
    )
    fraud_only = st.sidebar.checkbox("Fraud cases only", value=False)

    filtered = filter_claims(df, incident_filter, complexity_filter, fraud_only)
    st.sidebar.caption(f"{len(filtered)} claims match filters")

    if filtered.empty:
        st.sidebar.warning("No claims match filters")
        return None

    return st.sidebar.selectbox("Claim ID", options=filtered["claim_id"].tolist(), index=0)


def render_claim_overview(row: pd.Series, model_output: Optional[dict]) -> None:
    """Top section: key facts at a glance."""
    cols = st.columns(5)

    with cols[0]:
        st.markdown(f"**{row['claim_id']}**")
        st.caption(row["incident_type"].replace("_", " ").title())

    with cols[1]:
        st.metric("Damage", f"€{row['damage_estimate']:,.0f}")

    with cols[2]:
        if model_output:
            st.metric(
                "Complexity",
                model_output["complexity_label"].title(),
                f"{model_output['complexity_confidence']:.0%} confidence",
            )
        else:
            st.metric("Complexity", row.get("complexity", "—").title())

    with cols[3]:
        if model_output:
            st.metric(
                "Fraud Risk",
                f"{model_output['fraud_score']:.0%}",
                model_output["fraud_label"].title(),
            )
        else:
            st.metric("Fraud", "Yes" if row.get("is_fraud") else "No")

    with cols[4]:
        if model_output:
            queue = model_output["recommended_queue"]
            config = QUEUE_CONFIG.get(queue, QUEUE_CONFIG["standard"])
            st.markdown(
                f'<span class="queue-badge" style="background:{config["color"]}">'
                f'{config["icon"]} {config["label"]}</span>',
                unsafe_allow_html=True,
            )
            st.caption(f"~{model_output['expected_handling_days']:.0f} days expected")
        else:
            st.metric("Queue", row.get("handler_queue", "—"))


def render_description_and_summary(row: pd.Series, extraction: Optional[dict]) -> None:
    """Claim description + AI summary side by side."""
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("##### 📝 Customer Description")
        st.text_area(
            "description",
            value=row["description"],
            height=180,
            disabled=True,
            label_visibility="collapsed",
        )

    with col_right:
        st.markdown("##### 🤖 AI Summary")
        if extraction and extraction.get("summary"):
            st.info(extraction["summary"])
            if extraction.get("extraction_notes"):
                st.warning(f"⚠️ **Note:** {extraction['extraction_notes']}")
        else:
            st.caption("Run LLM extraction to generate summary")


def render_extracted_facts(extraction: Optional[dict]) -> None:
    """Show extracted facts with confidence indicators."""
    st.markdown("##### 📋 Extracted Facts")

    if not extraction:
        st.caption("No extraction available — run the extraction pipeline first")
        return

    facts = extraction.get("facts", {})
    for label, key in FACT_FIELDS:
        value = facts.get(key)
        if isinstance(value, dict):
            display = value.get("value", "—") or "—"
            confidence = value.get("confidence", "unknown")
            badge = CONFIDENCE_BADGES.get(confidence, "⚫")
            snippet = value.get("source_snippet", "")
            tooltip = f' title="{snippet}"' if snippet else ""
            st.markdown(
                f'<div class="fact-row">'
                f"<span>{label}</span>"
                f"<span{tooltip}>{badge} {display}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
        elif isinstance(value, bool):
            st.markdown(
                f'<div class="fact-row"><span>{label}</span><span>{"Yes" if value else "No"}</span></div>',
                unsafe_allow_html=True,
            )
        elif value is not None:
            st.markdown(
                f'<div class="fact-row"><span>{label}</span><span>{value}</span></div>',
                unsafe_allow_html=True,
            )


def render_next_actions(nba_result) -> None:
    """Render the next-best-action checklist."""
    st.markdown("##### ✅ Next Actions")
    st.caption(nba_result.summary_note)

    for action in nba_result.actions:
        color = PRIORITY_COLORS.get(action.priority, "#6b7280")
        priority_emoji = PRIORITY_EMOJI.get(action.priority, "⚪")
        category_emoji = CATEGORY_EMOJI.get(action.category, "")

        st.markdown(
            f'<div class="action-item" style="border-left-color: {color};">'
            f"<strong>{priority_emoji} {category_emoji} {action.action}</strong><br>"
            f'<span style="font-size:0.85rem;opacity:0.7">{action.reason}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

        if action.template_key:
            with st.expander(f"📧 View template: {action.template_key}", expanded=False):
                st.code(TEMPLATES.get(action.template_key, "Template not found"), language=None)


def render_fraud_detail(model_output: dict, row: pd.Series) -> None:
    """Detailed fraud analysis panel."""
    st.markdown("##### 🔍 Fraud Analysis")
    fraud_score = model_output.get("fraud_score", 0)
    marker_pos = fraud_score * 100

    st.markdown(
        f'<div class="fraud-gauge"><div class="fraud-marker" style="left:{marker_pos}%"></div></div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(3)
    cols[0].markdown('<span style="color:#16a34a;font-size:0.75rem">Low</span>', unsafe_allow_html=True)
    cols[1].markdown(
        '<span style="color:#ca8a04;font-size:0.75rem;text-align:center;display:block">Medium</span>',
        unsafe_allow_html=True,
    )
    cols[2].markdown(
        '<span style="color:#dc2626;font-size:0.75rem;text-align:right;display:block">High</span>',
        unsafe_allow_html=True,
    )

    signals = collect_fraud_signals(row)
    if not signals:
        st.caption("No specific risk signals detected")
        return

    for signal in signals:
        st.markdown(f"- {signal}")


def render_documents(claim_id: str, documents: pd.DataFrame) -> None:
    """Document status panel."""
    st.markdown("##### 📎 Documents")

    claim_docs = documents[documents["claim_id"] == claim_id]
    if claim_docs.empty:
        st.caption("No documents on file")
        return

    for _, document in claim_docs.iterrows():
        icon = "✅" if document["present"] else "❌"
        label = document["doc_type"].replace("_", " ").title()
        st.markdown(f"{icon} {label}")


def render_claim_context(row: pd.Series) -> None:
    """Policy, vehicle, and policyholder context."""
    st.markdown("##### 📊 Context")

    with st.expander("Policy & Vehicle", expanded=False):
        st.markdown(f"""
| | |
|---|---|
| **Policy** | {row.get('policy_id', '—')} ({row.get('policy_type', '—')}) |
| **Premium** | EUR {row.get('annual_premium', 0):,.0f}/yr |
| **Deductible** | EUR {row.get('deductible', 0):,.0f} |
| **Coverage Limit** | EUR {row.get('coverage_limit', 0):,.0f} |
| **Vehicle** | {row.get('make', '')} {row.get('model', '')} ({row.get('year', '')}) |
| **Value** | EUR {row.get('estimated_value', 0):,.0f} |
| **Plate** | {row.get('plate_number', '—')} |
| **Fuel** | {row.get('fuel_type', '—')} |
        """)

    with st.expander("Policyholder", expanded=False):
        st.markdown(f"""
| | |
|---|---|
| **Name** | {row.get('first_name', '')} {row.get('last_name', '')} |
| **City** | {row.get('city', '')} ({row.get('province', '')}) |
| **DOB** | {row.get('date_of_birth', '—')} |
| **License since** | {row.get('driving_license_year', '—')} |
        """)


def render_model_performance() -> None:
    """Sidebar model performance metrics."""
    metrics = load_metrics()
    if not metrics:
        return

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Model Performance")

    complexity = metrics.get("complexity", {})
    if complexity:
        st.sidebar.metric("Complexity Accuracy", f"{complexity.get('accuracy', 0):.1%}")
        st.sidebar.metric("F1 (macro)", f"{complexity.get('f1_macro', 0):.3f}")

    fraud = metrics.get("fraud", {})
    if fraud:
        st.sidebar.metric("Fraud AUC", f"{fraud.get('auc_roc', 0):.3f}")
        st.sidebar.metric("Precision@10%", f"{fraud.get('precision_at_10pct', 0):.1%}")

    handling_days = metrics.get("handling_days", {})
    if handling_days:
        st.sidebar.metric("Handling MAE", f"{handling_days.get('mae', 0):.1f} days")
def render_feedback_form(claim_id: str) -> Optional[dict]:
    """Render feedback form and return a payload descriptor when submitted."""
    st.markdown("---")
    st.markdown("##### 💬 Feedback")
    st.caption("Submitting feedback will save the current on-screen output first.")

    with st.form(key=f"feedback_form_{claim_id}"):
        feedback_type = st.radio(
            "How useful was this output?",
            options=["accepted", "edited", "rejected"],
            horizontal=True,
            format_func=lambda value: {
                "accepted": "Helpful",
                "edited": "Edited",
                "rejected": "Not helpful",
            }[value],
        )
        detail_text = st.text_area(
            "What changed or why?",
            height=100,
            help="Required for edited feedback. Optional for helpful/not helpful.",
        )
        submitted = st.form_submit_button("Send feedback", use_container_width=True)

    if not submitted:
        return None

    if feedback_type == "edited" and not detail_text.strip():
        st.warning("Describe what you changed so the feedback is useful.")
        return None

    return {
        "claim_id": claim_id,
        "feedback_type": feedback_type,
        "detail_text": detail_text,
    }


def render_dashboard(df: pd.DataFrame) -> None:
    """Overview dashboard with portfolio-level stats."""
    st.markdown("##### 📊 Claims Dashboard")

    cols = st.columns(4)
    cols[0].metric("Total Claims", f"{len(df):,}")
    cols[1].metric("Fraud Rate", f"{df['is_fraud'].mean():.1%}")
    cols[2].metric("Avg Handling", f"{df['handling_days'].mean():.0f} days")
    cols[3].metric("Avg Damage", f"€{df['damage_estimate'].mean():,.0f}")

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("###### By Incident Type")
        st.bar_chart(df["incident_type"].value_counts())
    with col_right:
        st.markdown("###### By Complexity")
        st.bar_chart(df["complexity"].value_counts().reindex(["simple", "medium", "complex"]))

    col_left2, col_right2 = st.columns(2)
    with col_left2:
        st.markdown("###### Handling Days Distribution")
        st.bar_chart(
            pd.cut(df["handling_days"], bins=[0, 5, 10, 20, 30, 50, 100])
            .value_counts()
            .sort_index()
        )
    with col_right2:
        st.markdown("###### Fraud by Incident Type")
        st.bar_chart(df.groupby("incident_type")["is_fraud"].mean().sort_values(ascending=False))
