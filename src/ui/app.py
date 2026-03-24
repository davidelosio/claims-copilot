"""
Claims Copilot — Handler UI

The panel a claims handler sees when they open a claim.
Ties together all layers: extraction, complexity, fraud, next-best-action.

Run:
    uv run streamlit run src/ui/app.py
"""

from __future__ import annotations

from urllib.error import HTTPError, URLError

import streamlit as st

from src.serving.next_best_action import NextBestActionEngine
from src.ui.components import (
    render_claim_context,
    render_claim_overview,
    render_claim_selector,
    render_dashboard,
    render_description_and_summary,
    render_documents,
    render_extracted_facts,
    render_feedback_form,
    render_fraud_detail,
    render_header,
    render_model_performance,
    render_next_actions,
    render_output_persistence,
)
from src.ui.constants import DASHBOARD_VIEW, VIEW_OPTIONS
from src.ui.data import (
    get_claim_prediction,
    get_missing_model_artifacts,
    load_all_data,
    load_extractions,
    load_predictor,
)
from src.ui.logic import build_nba_claim
from src.ui.persistence import (
    ClaimsCopilotApiClient,
    build_copilot_output_payload,
    build_feedback_payload,
)


def main() -> None:
    st.set_page_config(
        page_title="Claims Copilot",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    render_header()

    try:
        df, documents = load_all_data()
    except FileNotFoundError:
        st.error(
            "Data not found. Generate it first:\n\n"
            "```\nuv run python scripts/generate_claims.py --n-claims 5000 --output csv\n```"
        )
        return

    predictor = load_predictor()
    extractions = load_extractions()
    missing_model_artifacts = get_missing_model_artifacts()
    nba_engine = NextBestActionEngine()
    api_client = ClaimsCopilotApiClient()

    page = st.sidebar.radio("View", VIEW_OPTIONS, index=0)
    if missing_model_artifacts:
        st.sidebar.info(
            "Predictions are disabled until model artifacts are present. "
            f"Missing: {', '.join(missing_model_artifacts)}"
        )

    if page == DASHBOARD_VIEW:
        render_dashboard(df)
        render_model_performance()
        return

    selected_id = render_claim_selector(df)
    if not selected_id:
        return

    render_model_performance()

    row = df[df["claim_id"] == selected_id].iloc[0]
    claim_docs = documents[documents["claim_id"] == selected_id]
    extraction = extractions.get(selected_id)
    api_error = None
    saved_output = None

    model_output = None
    if predictor:
        try:
            model_output = get_claim_prediction(selected_id, predictor)
        except Exception as exc:
            st.sidebar.warning(f"Model prediction failed: {exc}")

    try:
        saved_output = api_client.get_latest_output(selected_id)
    except (HTTPError, URLError) as exc:
        api_error = str(exc)
        st.sidebar.info("Persistence API unavailable — output saving and feedback are disabled.")

    render_claim_overview(row, model_output)
    st.markdown("---")

    nba_result = None
    col_main, col_side = st.columns([3, 2])
    with col_main:
        render_description_and_summary(row, extraction)
        st.markdown("---")

        if model_output:
            nba_result = nba_engine.generate(
                claim_id=selected_id,
                claim=build_nba_claim(row),
                model_output=model_output,
                documents=claim_docs.to_dict("records"),
                extraction=extraction,
            )
            render_next_actions(nba_result)
        else:
            st.caption("Train models to see next-best-action recommendations")

        if model_output and nba_result:
            save_clicked = render_output_persistence(selected_id, saved_output, api_error)
            if save_clicked:
                try:
                    saved_output = api_client.create_output(
                        build_copilot_output_payload(
                            claim_id=selected_id,
                            extraction=extraction,
                            model_output=model_output,
                            nba_result=nba_result,
                        )
                    )
                    st.success(f"Saved copilot snapshot #{saved_output['output_id']}.")
                except (HTTPError, URLError) as exc:
                    st.error(str(exc))

    with col_side:
        render_extracted_facts(extraction)
        st.markdown("---")

        if model_output:
            render_fraud_detail(model_output, row)

        st.markdown("---")
        render_documents(selected_id, documents)
        st.markdown("---")
        render_claim_context(row)

    feedback_request = render_feedback_form(selected_id, saved_output, api_error)
    if feedback_request:
        try:
            feedback = api_client.create_feedback(
                build_feedback_payload(
                    claim_id=feedback_request["claim_id"],
                    output_id=feedback_request["output_id"],
                    feedback_type=feedback_request["feedback_type"],
                    detail_text=feedback_request["detail_text"],
                )
            )
            st.success(
                f"Feedback recorded as {feedback['feedback_type']} for snapshot "
                f"#{feedback['output_id']}."
            )
        except (HTTPError, URLError) as exc:
            st.error(str(exc))

    with st.expander("🔬 Ground Truth (dev only)", expanded=False):
        gt_cols = st.columns(4)
        gt_cols[0].metric("Complexity", row.get("complexity", "—").title())
        gt_cols[1].metric("Handling Days", f"{row.get('handling_days', '—')}")
        gt_cols[2].metric("Fraud", "Yes" if row.get("is_fraud") else "No")
        gt_cols[3].metric("Fraud Type", row.get("fraud_type", "—"))
        if row.get("settled_amount"):
            st.metric("Settled Amount", f"€{row['settled_amount']:,.0f}")


if __name__ == "__main__":
    main()
