#user_dashboard.py
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, func
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
from database import Base, ROITracking, engine
from enum import Enum
import plotly.graph_objs as go
import time  # For periodic refresh

class TrafficStatus(Enum):
    FREE_FLOW = "Free Flow"
    LIGHT_TRAFFIC = "Light Traffic"
    MODERATE_TRAFFIC = "Moderate Traffic"
    HEAVY_TRAFFIC = "Heavy Traffic"

class TrafficStatusConfig:
    CONFIG = {
        TrafficStatus.FREE_FLOW: {
            "color": "green",
            "icon": "🟢",
            "description": "Smooth flowing traffic with minimal congestion"
        },
        TrafficStatus.LIGHT_TRAFFIC: {
            "color": "yellow",
            "icon": "🟡",
            "description": "Slight slowdown, but still moving"
        },
        TrafficStatus.MODERATE_TRAFFIC: {
            "color": "orange",
            "icon": "🟠",
            "description": "Noticeable congestion and slower movement"
        },
        TrafficStatus.HEAVY_TRAFFIC: {
            "color": "red",
            "icon": "🔴",
            "description": "Significant congestion and potential standstill"
        }
    }

class TrafficStatusApp:
    def __init__(self):
        # Create a session factory
        self.SessionLocal = sessionmaker(bind=engine)

    def get_roi_tracking_data(self, hours=24):
        """Retrieve ROI tracking data from the last specified hours"""
        session = self.SessionLocal()
        try:
            # Calculate the timestamp for 'hours' ago
            time_threshold = datetime.utcnow() - timedelta(hours=hours)
            
            # Fetch data points from the last 'hours'
            query = (session.query(ROITracking)
                     .filter(ROITracking.timestamp >= time_threshold)
                     .order_by(ROITracking.timestamp.desc()))
            
            data = query.all()
            
            # Convert to DataFrame
            df = pd.DataFrame([
                {
                    'timestamp': entry.timestamp, 
                    'roi_id': entry.roi_id, 
                    'vehicle_count': entry.vehicle_count, 
                    'density': entry.density, 
                    'status': entry.status
                } for entry in data
            ])
            
            return df
        except Exception as e:
            st.error(f"Error retrieving data: {e}")
            return pd.DataFrame()
        finally:
            session.close()

    def render_roi_status(self, roi_id, status):
        """Render individual ROI status"""
        # Map ROI IDs to lane names
        lane_names = {
            'ROI ENTERING_ROI': 'Left Lane', 
            'ROI LEAVING_ROI': 'Right Lane'
        }
        lane_name = lane_names.get(roi_id, roi_id)

        # Convert string status to TrafficStatus enum
        try:
            traffic_status = TrafficStatus(status)
        except ValueError:
            # Default to Free Flow if status is not recognized
            traffic_status = TrafficStatus.FREE_FLOW

        config = TrafficStatusConfig.CONFIG[traffic_status]
        
        st.markdown(f"""
        <div style="
            background-color: {config['color']}20;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            border: 2px solid {config['color']};">
            <h3>{lane_name}</h3>
            <h1>{config['icon']}</h1>
            <p><strong>{traffic_status.value}</strong></p>
            <small>{config['description']}</small>
        </div>
        """, unsafe_allow_html=True)

    def create_traffic_trends_chart(self, df):
        """Create a line chart showing traffic trends"""
        # Group by timestamp and ROI, aggregate vehicle count and density
        grouped = df.groupby(['timestamp', 'roi_id']).agg({
            'vehicle_count': 'mean',
            'density': 'mean'
        }).reset_index()

        # Create a Plotly figure
        fig = go.Figure()

        # Add vehicle count traces for each ROI
        for roi in grouped['roi_id'].unique():
            roi_data = grouped[grouped['roi_id'] == roi]
            fig.add_trace(go.Scatter(
                x=roi_data['timestamp'],
                y=roi_data['vehicle_count'],
                mode='lines+markers',
                name=f'{roi} Vehicle Count'
            ))

        # Update layout
        fig.update_layout(
            title='Vehicle Count Trends by ROI',
            xaxis_title='Timestamp',
            yaxis_title='Vehicle Count',
            height=400
        )

        return fig

    def run(self):
        """Main Streamlit app"""
        st.set_page_config(page_title="Traffic Status", layout="wide")
        
        # Title and header
        st.title("🚦 Real-Time Traffic Status Dashboard")
        
        # Time range selector
        hours = st.sidebar.selectbox(
            "Select Data Range",
            [1, 6, 12, 24, 48],
            index=3,
            format_func=lambda x: f"Last {x} Hours"
        )
        
        # Fetch the latest data
        df = self.get_roi_tracking_data(hours)
        
        if df.empty:
            st.warning("No data available")
            return
        
        # Get latest status for each ROI
        latest_roi_status = df.groupby('roi_id').last().reset_index()
        
        # Create columns for ROI statuses
        st.subheader("Current Lane Statuses")
        columns = st.columns(len(latest_roi_status))
        
        for idx, (_, row) in enumerate(latest_roi_status.iterrows()):
            with columns[idx]:
                self.render_roi_status(
                    row['roi_id'], 
                    row['status']
                )
        
        # Update time display
        latest_timestamp = df['timestamp'].max()
        st.markdown(f"🕒 Last Updated: {latest_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Refresh button
        if st.button("Refresh Traffic Status"):
            st.rerun()

def main():
    app = TrafficStatusApp()
    app.run()

if __name__ == "__main__":
    main()