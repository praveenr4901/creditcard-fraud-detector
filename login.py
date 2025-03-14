import streamlit as st

# Dummy user credentials
users = {"admin": "admin123", "user": "user123"}

def login():
    """Handles user login."""
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.rerun()  # Refresh after login
        else:
            st.error("Invalid username or password")

def check_authentication():
    """Ensures user is authenticated before accessing the dashboard."""
    if not st.session_state.get("authenticated", False):
        login()
        st.stop()

def logout():
    """Handles user logout."""
    if st.sidebar.button("Logout"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        st.rerun()

def get_user_role():
    """Returns the role of the currently logged-in user."""
    if st.session_state.get("username") == "admin":
        return "admin"
    return "user"

