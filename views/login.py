"""Login / register screen."""
import streamlit as st

from src import auth, ui


def render():
    st.title("AI Loan Approval Assistant")
    ui.hero(
        "Trust the decision",
        "Smarter loans.",
        "Clearer decisions.",
        "Estimate approval chances, see exactly why, and review every case with confidence.",
    )
    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in")
        if submitted:
            user = auth.authenticate(username, password)
            if user:
                st.session_state["user"] = user
                st.rerun()
            else:
                st.error("Invalid username or password.")

    with register_tab:
        with st.form("register_form"):
            new_user = st.text_input("Choose a username")
            new_pass = st.text_input("Choose a password", type="password")
            confirm = st.text_input("Confirm password", type="password")
            created = st.form_submit_button("Create account")
        if created:
            if new_pass != confirm:
                st.error("Passwords do not match.")
            else:
                try:
                    auth.create_user(new_user, new_pass, role="officer")
                    st.success("Account created. You can log in now.")
                except ValueError as e:
                    st.error(str(e))