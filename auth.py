import streamlit as st
from databaseUser import add_user, authenticate_user, update_user_password

def reset_password_form():
    """
    Render a password reset form with actual password reset functionality
    """
    col1, col2, col3 = st.columns([3, 2, 3])
    
    with col2:
        with st.container(border=True):
            st.markdown("<h2 style='text-align: center; color: #333;'>Change Password</h2>", unsafe_allow_html=True)
            
            # Username input
            username = st.text_input("Username", placeholder="Enter your username")
            
            # New password inputs
            new_password = st.text_input("New Password", 
                                         type="password", 
                                         placeholder="Create a new password",
                                         help="Password must be at least 8 characters with uppercase, lowercase, number, and special character")
            
            confirm_new_password = st.text_input("Confirm New Password", 
                                                 type="password", 
                                                 placeholder="Repeat your new password")
            
            # Reset button
            if st.button("Change Password", use_container_width=True):
                # Validation checks
                if not username.strip():
                    st.error("Username is required.")
                elif not new_password:
                    st.error("New password is required.")
                elif len(new_password) < 8:
                    st.error("Password must be at least 8 characters long.")
                elif new_password != confirm_new_password:
                    st.error("Passwords do not match.")
                else:
                    try:
                        # Attempt to update user password
                        reset_success = update_user_password(username.strip(), new_password)
                        
                        if reset_success:
                            st.success("Password reset successfully!")
                            # Optional: Automatically switch to login
                            st.session_state['auth_mode'] = 'login'
                            st.rerun()
                        else:
                            st.error("Failed to reset password. Username may not exist.")
                    
                    except Exception as e:
                        st.error(f"An error occurred during password reset: {str(e)}")
            
            # Back to login link
            st.markdown("---")
            col1, col2 = st.columns([3, 1])  # Adjust column proportions as needed

        # Text in the first column
        with col1:
            st.markdown(
        "<p style='text-align: left; color: #666;'>Remember your password?</p>",
        unsafe_allow_html=True
    )
            with col2:
                if st.markdown("").button("Login"):
                    st.session_state['auth_mode'] = 'login'
                    st.rerun()

def login_form():
    """
    Render a centered, container-based login form
    """
    # Create a container to center the login form
    col1, col2, col3 = st.columns([3, 2, 3])
    
    with col2:
        # Create a card-like container
        with st.container(border=True):
            st.markdown("<h2 style='text-align: center; color: #333;'>Login</h2>", unsafe_allow_html=True)
            
            # Username input
            username = st.text_input("Username", placeholder="Enter your username")
            
            # Password input
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            # Login button with custom styling
            login_col1, login_col2 = st.columns([1, 1])
            with login_col1:
                if st.button("Login", use_container_width=True):
                    # Additional validation for login
                    if not username or not password:
                        st.error("Username and password are required.")
                    else:
                        if authenticate_user(username, password):
                            st.session_state['logged_in'] = True
                            st.session_state['username'] = username
                            st.success("Logged in successfully!")
                            return True
                        else:
                            st.error("Invalid username or password.")
            
            with login_col2:
                # Forgot password link 
                if st.button("Forgot Password?", use_container_width=True):
                    # Set session state to show reset password form
                    st.session_state['auth_mode'] = 'reset_password'
                    st.rerun()
            
            # Divider
            st.markdown("---")
            col1, col2 = st.columns([3, 1])  # Adjust column proportions as needed

        # Text in the first column
        with col1:
            st.markdown(
        "<p style='text-align: left; color: #666;'>Don't have an account?</p>",
        unsafe_allow_html=True
    )
            with col2:
             if st.button("Register"):
                    st.session_state['auth_mode'] = 'register'
                    st.rerun()


def registration_form():
    """
    Render a centered, container-based registration form
    """
    # Create a container to center the registration form
    col1, col2, col3 = st.columns([3, 2, 3])
    
    with col2:
        # Create a card-like container
        with st.container(border=True):
            st.markdown("<h2 style='text-align: center; color: #333;'>Create Account</h2>", unsafe_allow_html=True)
            
            # Username input with hint
            username = st.text_input("Username", 
                                     placeholder="Choose a username (3-20 characters)",
                                     help="Username must start with a letter, can contain letters, numbers, and underscores")
            
            # Password inputs
            password = st.text_input("Password", 
                                     type="password", 
                                     placeholder="Create a strong password",
                                     help="Password must be at least 8 characters with uppercase, lowercase, number, and special character")
            
            confirm_password = st.text_input("Confirm Password", 
                                              type="password", 
                                              placeholder="Repeat your password")
            
            # Registration button with columns for layout
            reg_col1, reg_col2 = st.columns([1, 1])
            with reg_col1:
                if st.button("Register", use_container_width=True):
                    # Comprehensive validation checks
                    if not username.strip():
                        st.error("Username cannot be empty.")
                    elif not password:
                        st.error("Password is required.")
                    elif password != confirm_password:
                        st.error("Passwords do not match.")
                    elif len(password) < 8:
                        st.error("Password must be at least 8 characters long.")
                    else:
                        # Attempt to add user with more robust error handling
                        try:
                            result = add_user(username.strip(), password)
                            if result:
                                st.success("Account created successfully! Please login.")
                                # Optionally, switch back to login
                                st.session_state['auth_mode'] = 'login'
                                st.rerun()
                            else:
                                st.error("Failed to create account. Username may already exist.")
                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")
            
            with reg_col2:
                # Reset form button
                if st.button("Reset", use_container_width=True):
                    st.rerun()
            
            # Divider
            st.markdown("---")
            col1, col2 = st.columns([3, 1])  # Adjust column proportions as needed
            with col1:
                st.markdown(
                "<p style='text-align: left; color: #666;'>Already have an account?</p>",
        unsafe_allow_html=True
    )
            with col2:
                if st.markdown("").button("Login"):
                    st.session_state['auth_mode'] = 'login'
                    st.rerun()

def logout():
    """
    Logout function to clear session state
    """
    st.session_state.clear()
    st.success("Logged out successfully.")
    return True

def auth_main():
    """
    Main authentication routing function
    """
    # Initialize auth mode if not set
    if 'auth_mode' not in st.session_state:
        st.session_state['auth_mode'] = 'login'
    
    # Routing based on auth_mode
    if st.session_state['auth_mode'] == 'login':
        login_form()
    elif st.session_state['auth_mode'] == 'register':
        registration_form()
    elif st.session_state['auth_mode'] == 'reset_password':
        reset_password_form()