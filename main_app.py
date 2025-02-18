import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import yolo_app  
from databaseUser import add_user, authenticate_user
from auth import auth_main

st.set_page_config(
    page_title="Vehicle Counting System",
    page_icon="",
    layout="wide"
)

def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate_user(username, password):
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.session_state['current_page'] = "main"  
            st.rerun()
        else:
            st.error("Invalid username or password.")

def register_page():
    st.title("Register")
    username = st.text_input("Choose a username")
    password = st.text_input("Choose a password", type="password")
    if st.button("Register"):
        if add_user(username, password):
            st.success("Account created successfully! Please login.")
        else:
            st.error("Username already exists or an error occurred.")

def logout():
    st.session_state.clear()
    st.success("Logged out successfully.")


def render_sidebar():
    with st.sidebar:
        st.image("images/logo2.png", use_container_width=True)
        
        if st.session_state.get("logged_in"):
            selected = option_menu(
                menu_title="Algorithm",
                options=["YOLOv8"],
                icons=["camera"],
                menu_icon="cast",
                default_index=0,
                styles={
                    "container": {"padding": "5px"},
                    "icon": {"font-size": "25px"},
                    "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"},
                    "nav-link-selected": {"background-color": "#808080"},
                }
            )
            st.markdown("---")
            if st.button("Logout"):
                st.session_state["logged_in"] = False
                st.session_state["current_page"] = "main"
                st.rerun()
            
            return selected
        
        else:
            selected = option_menu(
                menu_title="MENU",
                options=["Login", "Register"],
                icons=["person", "person-plus"],
                menu_icon="cast",
                default_index=0,
                styles={
                    "container": {"padding": "5px"},
                    "icon": {"font-size": "25px"},
                    "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"},
                    "nav-link-selected": {"background-color": "#808080"},
                }
            )
            return selected

def run_yolo_page():
    st.title("YOLOv8 Configuration")
    st.markdown("""
    Use this page to configure and launch YOLOv8 for real-time object detection.
    """)

    if st.button("🚀 Launch YOLOv8"):
        st.session_state["current_page"] = "yolo_app"
        st.rerun()


def main():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "main"

    if not st.session_state.get("logged_in"):
        auth_main()  
    else:
        if st.session_state["current_page"] == "yolo_app":
            yolo_app.main()  
        else:
            selected = render_sidebar()
            
            if selected == "YOLOv8":
                run_yolo_page()

if __name__ == "__main__":
    main()