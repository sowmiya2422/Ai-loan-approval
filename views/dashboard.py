"""Admin dashboard: KPIs, model performance, fairness, dataset insights, activity."""
import pandas as pd
import plotly.express as px
import streamlit as st

import config
from src import analytics, ui

ACCENT = "#0E9AA7"
GREEN = "#12A150"
RED = "#E5484D"
MUTED = "#CBD5E1"


@st.cache_data(show_spinner=False)
def _approval_rate_by(column: str) -> pd.DataFrame:
    return analytics.approval_rate_by(column)


def _style(fig, height: int = 330):
    """Shared look for every chart: transparent background, subtle grid."""
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
        font=dict(color="#0B1F3A"),
    )
    fig.update_xaxes(gridcolor="rgba(11,31,58,0.08)")
    fig.update_yaxes(gridcolor="rgba(11,31,58,0.08)")
    return fig


def _kpis(metrics, preds):
    approved = (preds["decision"] == "Approved").mean() if len(preds) else None
    items = [
        ("Predictions made", f"{len(preds)}"),
        ("Approval rate (app)", f"{approved:.0%}" if approved is not None else "—"),
        ("Model accuracy", f"{metrics['test']['accuracy']:.1%}" if metrics else "—"),
        ("Model ROC-AUC", f"{metrics['test']['roc_auc']:.3f}" if metrics else "—"),
    ]
    for col, (label, value) in zip(st.columns(4), items):
        with col:
            st.metric(label, value)


def _model_tab(metrics):
    if metrics is None:
        st.warning("No trained model found. Run:  python scripts/train.py")
        return
    left, right = st.columns(2)

    with left:
        st.subheader("Model comparison")
        cv = pd.DataFrame({
            "model": list(metrics["cv_roc_auc"]),
            "roc_auc": list(metrics["cv_roc_auc"].values()),
        })
        cv["role"] = cv["model"].map(
            lambda m: "Selected" if m == metrics["best_model"] else "Other"
        )
        fig = px.bar(
            cv, x="model", y="roc_auc", color="role",
            text=cv["roc_auc"].map("{:.3f}".format),
            color_discrete_map={"Selected": ACCENT, "Other": MUTED},
        )
        fig.update_yaxes(range=[0.9, 1.0], title="Cross-validated ROC-AUC")
        fig.update_xaxes(title="")
        st.plotly_chart(_style(fig), use_container_width=True)
        st.caption("Y-axis starts at 0.90 so the small differences are visible.")

    with right:
        st.subheader("Test-set performance")
        test = pd.DataFrame(list(metrics["test"].items()), columns=["metric", "value"])
        test["metric"] = test["metric"].str.replace("_", " ").str.title()
        fig = px.bar(test, x="metric", y="value", text=test["value"].map("{:.3f}".format))
        fig.update_traces(marker_color=ACCENT)
        fig.update_yaxes(range=[0, 1.05], title="")
        fig.update_xaxes(title="")
        st.plotly_chart(_style(fig), use_container_width=True)
        st.caption(f"Best model: **{metrics['best_model']}**, evaluated once on held-out data.")


def _fairness_tab(metrics):
    if metrics is None:
        st.warning("No trained model found. Run:  python scripts/train.py")
        return
    groups = metrics["fairness_approval_rate_by_gender"]
    df = pd.DataFrame([
        {"gender": g.title(), "approval_rate": v["approval_rate"], "applicants": v["n"]}
        for g, v in groups.items()
    ])
    top, bottom = df["approval_rate"].max(), df["approval_rate"].min()
    gap = (top - bottom) * 100
    ratio = bottom / top if top > 0 else 1.0

    left, right = st.columns([2, 1])
    with left:
        st.subheader("Predicted approval rate by gender")
        fig = px.bar(
            df, x="gender", y="approval_rate",
            text=df["approval_rate"].map("{:.1%}".format), hover_data=["applicants"],
        )
        fig.update_traces(marker_color=ACCENT)
        fig.update_yaxes(tickformat=".0%", range=[0, top * 1.4], title="")
        fig.update_xaxes(title="")
        st.plotly_chart(_style(fig), use_container_width=True)
    with right:
        st.subheader("Fairness checks")
        with st.container():
            st.metric("Approval-rate gap", f"{gap:.1f} pts")
        with st.container():
            st.metric("Disparate impact ratio", f"{ratio:.2f}")
        st.caption(
            "Gender is **not** a model input; it is only used here to audit the "
            "results on the test set. A ratio of 0.80 or higher is a common rule of thumb."
        )


def _insights_tab():
    st.caption("Computed from the training dataset, not from app predictions.")
    left, right = st.columns(2)
    charts = [
        (left, "loan_intent", "Approval rate by loan purpose"),
        (right, "person_home_ownership", "Approval rate by home ownership"),
    ]
    for col, column, title in charts:
        with col:
            st.subheader(title)
            df = _approval_rate_by(column).sort_values("approval_rate")
            fig = px.bar(
                df, x="approval_rate", y=column, orientation="h",
                text=df["approval_rate"].map("{:.0%}".format),
                hover_data=["applications"],
            )
            fig.update_traces(marker_color=ACCENT)
            fig.update_xaxes(tickformat=".0%", title="")
            fig.update_yaxes(title="")
            st.plotly_chart(_style(fig), use_container_width=True)


def _activity_tab(preds):
    if preds.empty:
        st.info("No predictions yet. Assess an application first and it will show up here.")
        return
    left, right = st.columns(2)
    with left:
        st.subheader("Decisions")
        fig = px.pie(
            preds, names="decision", hole=0.55, color="decision",
            color_discrete_map={"Approved": GREEN, "Rejected": RED},
        )
        st.plotly_chart(_style(fig), use_container_width=True)
    with right:
        st.subheader("Predictions per day")
        daily = (
            preds.assign(day=preds["created_at"].dt.date)
            .groupby("day").size().reset_index(name="predictions")
        )
        fig = px.bar(daily, x="day", y="predictions")
        fig.update_traces(marker_color=ACCENT)
        fig.update_xaxes(title="")
        fig.update_yaxes(title="", dtick=1)
        st.plotly_chart(_style(fig), use_container_width=True)

    st.subheader("Recent predictions")
    recent = preds.head(10)[[
        "created_at", "username", "loan_intent", "loan_amnt",
        "credit_score", "probability", "decision",
    ]]
    st.dataframe(
        recent, hide_index=True, use_container_width=True,
        column_config={
            "probability": st.column_config.ProgressColumn(
                "Approval probability", min_value=0.0, max_value=1.0, format="%.2f"
            ),
        },
    )


def render(user: dict):
    if user["role"] != "admin":
        st.error("The dashboard is available to administrators only.")
        return

    ui.hero("Admin overview", "Model health,", "at a glance.")
    st.caption(config.DISCLAIMER)

    metrics = analytics.load_metrics()
    preds = analytics.load_predictions()

    _kpis(metrics, preds)

    model_tab, fairness_tab, insights_tab, activity_tab = st.tabs(
        ["Model", "Fairness", "Dataset insights", "Activity"]
    )
    with model_tab:
        _model_tab(metrics)
    with fairness_tab:
        _fairness_tab(metrics)
    with insights_tab:
        _insights_tab()
    with activity_tab:
        _activity_tab(preds)