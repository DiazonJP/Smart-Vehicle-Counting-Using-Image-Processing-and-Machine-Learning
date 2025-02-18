from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from werkzeug.security import generate_password_hash, check_password_hash
import re

# Database connection
DATABASE_URL = "postgresql://postgres:12345@localhost/users"

# SQLAlchemy setup
engine = create_engine(DATABASE_URL)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

# User model
class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)

# Create tables
def create_user_table():
    Base.metadata.create_all(engine)

def validate_username(username):
    """
    Validate username based on these criteria:
    - 3-20 characters long
    - Only contains alphanumeric characters and underscores
    - Starts with a letter
    """
    print(f"Debug: Validating username: {username}")
    
    if not username:
        print("Debug: Username is empty")
        return False
    
    # Check length
    if len(username) < 3 or len(username) > 20:
        print(f"Debug: Username length invalid. Length: {len(username)}")
        return False
    
    # Check if starts with a letter and contains only alphanumeric and underscores
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_]*$', username):
        print("Debug: Username does not match regex pattern")
        return False
    
    print("Debug: Username validation passed")
    return True

def validate_password(password):
    """
    Validate password based on these criteria:
    - At least 8 characters long
    """
    if not password:
        return False
    
    # Check length
    if len(password) < 8:
        return False
    
    return True

def add_user(username, password):
    """
    Adds a new user to the database.
    Performs comprehensive validation before adding user.
    
    Returns:
    - True if user added successfully
    - False with specific error conditions:
      - Invalid username
      - Invalid password
      - Username already exists
      - Database error
    """
    # Validate username and password
    if not validate_username(username):
        print("Debug: Invalid username validation")
        return False
    
    if not validate_password(password):
        print("Debug: Invalid password validation")
        return False
    
    # Create a new session
    session = SessionLocal()
    try:
        # Check if username already exists
        existing_user = session.query(User).filter_by(username=username).first()
        if existing_user:
            print(f"Debug: Username '{username}' already exists.")
            return False
        
        # Hash the password
        hashed_password = generate_password_hash(password)
        
        # Create new user
        new_user = User(username=username, password_hash=hashed_password)
        
        # Add and commit the new user
        session.add(new_user)
        session.commit()
        
        print(f"Debug: User '{username}' successfully added.")
        return True
    
    except Exception as e:
        # Rollback the session in case of any database error
        session.rollback()
        print(f"Debug: Error adding user: {e}")
        return False
    
    finally:
        # Always close the session
        session.close()

def authenticate_user(username, password):
    """
    Authenticates a user by verifying the password.
    """
    session = SessionLocal()
    try:
        user = session.query(User).filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            return True
        return False
    except Exception as e:
        print(f"Error authenticating user: {e}")
        return False
    finally:
        session.close()

def update_user_password(username, new_password):
    """
    Update a user's password in the database
    
    Args:
        username (str): Username of the account to update
        new_password (str): New password to set
    
    Returns:
        bool: True if password was successfully updated, False otherwise
    """
    # Validate the new password
    if not validate_password(new_password):
        print("Debug: Invalid new password")
        return False

    # Create a new session
    session = SessionLocal()
    try:
        # Find the user
        user = session.query(User).filter_by(username=username).first()
        
        # Check if user exists
        if not user:
            print(f"Debug: User '{username}' not found.")
            return False
        
        # Hash the new password
        new_password_hash = generate_password_hash(new_password)
        
        # Update the password
        user.password_hash = new_password_hash
        
        # Commit the changes
        session.commit()
        
        print(f"Debug: Password for user '{username}' successfully updated.")
        return True
    
    except Exception as e:
        # Rollback the session in case of any database error
        session.rollback()
        print(f"Debug: Error updating password: {e}")
        return False
    
    finally:
        # Always close the session
        session.close()

def fetch_all_users():
    """
    Fetches all users for admin purposes.
    """
    session = SessionLocal()
    try:
        users = session.query(User).all()
        return [{"id": user.id, "username": user.username} for user in users]
    finally:
        session.close()

# Initialize the database
create_user_table()