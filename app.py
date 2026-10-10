"""Streamlit entry point: login gate, then sidebar navigation."""
import streamlit as st

from src import db, ui
from views import dashboard, history, login
from views import predict as predict_page

st.set_page_config(page_title="AI Loan Approval", page_icon="🏦", layout="wide")
db.init_db()
ui.apply_theme()

if "user" not in st.session_state:
    st.session_state["user"] = None

user = st.session_state["user"]

if user is None:
    login.render()
else:
    pages = [
        st.Page(lambda: predict_page.render(user), title="Assess",
                icon="📝", url_path="assess", default=True),
        st.Page(lambda: history.render(user), title="History",
                icon="🗂️", url_path="history"),
    ]
    if user["role"] == "admin":
        pages.append(
            st.Page(lambda: dashboard.render(user), title="Dashboard",
                    icon="📊", url_path="dashboard")
        )

    nav = st.navigation(pages)

    st.sidebar.divider()
    st.sidebar.write(f"Logged in as **{user['username']}** ({user['role']})")
    if st.sidebar.button("Log out"):
        st.session_state["user"] = None
        st.rerun()

    nav.run()