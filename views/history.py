"""Prediction history: officers see their own, admins see everyone's."""
import streamlit as st

from src import analytics, ui


def render(user: dict):
    is_admin = user["role"] == "admin"
    ui.hero("Audit trail", "Every prediction.", "One clear history.")
    st.caption(
        "Predictions made by all users." if is_admin else "Your past predictions."
    )

    preds = analytics.load_predictions(None if is_admin else user["id"])
    if preds.empty:
        st.info("No predictions yet. Open the Assess page and run one.")
        return

    # Filters
    left, right = st.columns(2)
    with left:
        decisions = st.multiselect(
            "Decision", ["Approved", "Rejected"], default=["Approved", "Rejected"]
        )
    chosen_users = None
    if is_admin:
        with right:
            all_users = sorted(preds["username"].unique())
            chosen_users = st.multiselect("User", all_users, default=all_users)

    view = preds[preds["decision"].isin(decisions)]
    if chosen_users is not None:
        view = view[view["username"].isin(chosen_users)]

    # Summary cards
    approved = (view["decision"] == "Approved").mean() if len(view) else None
    avg_prob = view["probability"].mean() if len(view) else None
    cards = [
        ("Showing", f"{len(view)}"),
        ("Approval rate", f"{approved:.0%}" if approved is not None else "—"),
        ("Average probability", f"{avg_prob:.2f}" if avg_prob is not None else "—"),
    ]
    for col, (label, value) in zip(st.columns(3), cards):
        with col, st.container(border=True):
            st.metric(label, value)

    if view.empty:
        st.info("Nothing matches these filters.")
        return

    # Table
    order = ["created_at", "username", "loan_intent", "loan_amnt",
             "person_income", "credit_score", "probability", "decision"]
    if not is_admin:
        order.remove("username")

    st.dataframe(
        view,
        hide_index=True,
        use_container_width=True,
        column_order=order,
        column_config={
            "created_at": st.column_config.DatetimeColumn(
                "Time (UTC)", format="DD MMM YYYY, HH:mm"
            ),
            "username": "User",
            "loan_intent": "Purpose",
            "loan_amnt": st.column_config.NumberColumn("Loan amount", format="%d"),
            "person_income": st.column_config.NumberColumn("Income", format="%d"),
            "credit_score": "Credit score",
            "probability": st.column_config.ProgressColumn(
                "Approval probability", min_value=0.0, max_value=1.0, format="%.2f"
            ),
            "decision": "Decision",
        },
    )

    st.download_button(
        "Download as CSV",
        data=view[order].to_csv(index=False).encode("utf-8"),
        file_name="prediction_history.csv",
        mime="text/csv",
    )