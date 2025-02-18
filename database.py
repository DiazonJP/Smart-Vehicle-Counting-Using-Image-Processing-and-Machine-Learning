from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json


# Database connection parameters
DATABASE_URL = "postgresql://postgres:12345@localhost/yolo_db"

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create a base class for declarative models
Base = declarative_base()


# Vehicle Entry/Exit Model
class VehicleLog(Base):
    __tablename__ = 'vehicle_logs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    vehicle_type = Column(String)
    direction = Column(String)  # 'entry' or 'exit'
    roi_id = Column(String)
    
    # Additional metrics
    vehicle_count = Column(Integer)
    density = Column(Float)
    status = Column(String)
    

# Time Series Data Model
class TimeSeriesData(Base):
    __tablename__ = 'time_series_data'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    vehicle_type = Column(String)
    direction = Column(String)
    count = Column(Integer)

# ROI Tracking Model
class ROITracking(Base):
    __tablename__ = 'roi_tracking'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    roi_id = Column(String)
    vehicle_count = Column(Integer)
    density = Column(Float)
    status = Column(String)

    

# Create database tables
def create_tables():
    Base.metadata.create_all(engine)

# Database Session
SessionLocal = sessionmaker(bind=engine)

# Function to add vehicle log
def add_vehicle_log(vehicle_type, direction, roi_id, vehicle_count, density, status, timestamp=None
                     ):
    session = SessionLocal()
    try:
        new_log = VehicleLog(
            vehicle_type=vehicle_type,
            direction=direction,
            roi_id=roi_id,
            vehicle_count=vehicle_count,
            density=density,
            status=status,
            timestamp=timestamp or datetime.utcnow() 
            
           
        )
        session.add(new_log)
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Error adding vehicle log: {e}")
    finally:
        session.close()

# Function to add time series data
def add_time_series_data(vehicle_type, direction, count, timestamp=None):
    session = SessionLocal()
    try:
        new_data = TimeSeriesData(
            vehicle_type=vehicle_type,
            direction=direction,
            count=count,
            timestamp=timestamp or datetime.utcnow()
        )
        session.add(new_data)
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Error adding time series data: {e}")
    finally:
        session.close()

# Function to add ROI tracking data
def add_roi_tracking_data(roi_id, vehicle_count, density, status, timestamp=None
                           ):
    session = SessionLocal()
    try:
        new_tracking = ROITracking(
            roi_id=roi_id,
            vehicle_count=vehicle_count,
            density=density,
            status=status,
            timestamp=timestamp or datetime.utcnow()
           
        )
        session.add(new_tracking)
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Error adding ROI tracking data: {e}")
    finally:
        session.close()

# Initialize database
def init_database():
    create_tables()

# Before running your application
init_database()