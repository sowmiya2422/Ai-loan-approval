"""Loan application form, decision and SHAP explanation."""
import plotly.express as px
import streamlit as st

import config
from src import anomaly, explain, ui, predict as scoring


def render(user: dict):
    st.title("Loan Application Assessment")
    ui.hero(
        "AI loan assessment",
        "Assess an application.",
        "Understand the decision.",
        "Enter the applicant's details to get an approval estimate and a clear explanation.",
    )
    st.caption(config.DISCLAIMER)
    opts = config.CATEGORY_OPTIONS

    with st.form("application_form"):
        left, right = st.columns(2)
        with left:
            age = st.number_input("Age", 18, 100, 30)
            income = st.number_input("Annual income", 1000, 10_000_000, 60_000, step=1000)
            emp_exp = st.number_input("Years of employment experience", 0, 60, 5)
            education = st.selectbox("Education", opts["person_education"])
            home = st.selectbox("Home ownership", opts["person_home_ownership"])
            prev_default = st.selectbox(
                "Previous loan default on file", opts["previous_loan_defaults_on_file"]
            )
        with right:
            loan_amnt = st.number_input("Loan amount", 500, 5_000_000, 10_000, step=500)
            int_rate = st.number_input("Interest rate (%)", 5.0, 25.0, 11.0, step=0.1)
            intent = st.selectbox("Loan purpose", opts["loan_intent"])
            hist = st.number_input("Credit history length (years)", 0, 50, 5)
            score = st.number_input("Credit score", 300, 850, 690)
        submitted = st.form_submit_button("Assess application")

    if not submitted:
        return

    inputs = {
        "person_age": age,
        "person_income": income,
        "person_emp_exp": emp_exp,
        "loan_amnt": loan_amnt,
        "loan_int_rate": int_rate,
        "loan_percent_income": round(loan_amnt / income, 2),  # calculated, not typed
        "cb_person_cred_hist_length": hist,
        "credit_score": score,
        "person_education": education,
        "person_home_ownership": home,
        "loan_intent": intent,
        "previous_loan_defaults_on_file": prev_default,
    }
    unusual = anomaly.check(inputs)
    if unusual["is_unusual"] or inputs["loan_percent_income"] > 0.7:
        st.warning(
            "**Unusual application.** These values look very different from the "
            "applications the model was trained on, so the result may be unreliable. "
            "Please double-check the inputs and have an officer review this case."
        )
    try:
        result = scoring.predict(inputs)
        drivers = explain.explain(inputs)
    except FileNotFoundError as e:
        st.error(str(e))
        return

    scoring.save_prediction(user["id"], inputs, result)

    st.subheader("Result")
    col_a, col_b = st.columns(2)
    col_a.metric("Approval probability", f"{result['probability']:.1%}")
    col_b.metric("Model decision", result["decision"])
    if result["decision"] == "Approved":
        st.success("The model leans towards approval.")
    else:
        st.error("The model leans towards rejection.")

    st.subheader("Why? Top factors for this application")
    st.caption("Green = pushes towards approval, red = pushes towards rejection.")
    chart_df = drivers.assign(
        value=drivers["value"].astype(str),
        direction=drivers["impact"].apply(
            lambda v: "Towards approval" if v > 0 else "Towards rejection"
        ),
    ).iloc[::-1]  # reversed so the strongest factor appears at the top
    fig = px.bar(
        chart_df,
        x="impact",
        y="feature",
        orientation="h",
        color="direction",
        color_discrete_map={
            "Towards approval": "#2e9e5b",
            "Towards rejection": "#d64545",
        },
        hover_data=["value"],
    )
    fig.update_layout(
        xaxis_title="Impact on decision (SHAP)", yaxis_title="", legend_title=""
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(chart_df.iloc[::-1][["feature", "value", "impact"]], hide_index=True)