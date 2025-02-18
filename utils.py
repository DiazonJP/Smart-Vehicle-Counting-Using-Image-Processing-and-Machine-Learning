#utils.py
from ultralytics import YOLO
import streamlit as st
import cv2
from PIL import Image
import tempfile
import config
from pathlib import Path
import plotly.graph_objs as plt
from datetime import datetime, timedelta
import plotly.graph_objects as go
import pandas as pd
import uuid  
from collections import defaultdict
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

def create_vehicle_count_line_graphs(vehicle_entry_series, vehicle_exit_series):
    """
    Create line graphs for vehicle entry and exit counts over time.
    
    :param vehicle_entry_series: Dictionary containing entry counts for each vehicle type
    :param vehicle_exit_series: Dictionary containing exit counts for each vehicle type
    :return: Two Plotly figures - one for entry and one for exit
    """
    # Colors for different vehicle types
    colors = {
        'car': '#1f77b4',
        'motorcycle': '#ff7f0e',
        'bus': '#2ca02c',
        'truck': '#d62728',
        'tricycle': '#9467bd',
        'mini-bus': '#e377c2'
    }
    
    # Create figure for entry counts
    entry_fig = go.Figure()
    for vehicle_type, data in vehicle_entry_series.items():
        if data and len(data.get('counts', [])) > 0:  # Check if data exists and has counts
            entry_fig.add_trace(go.Scatter(
                x=data['timestamps'],
                y=data['counts'],
                mode='lines+markers',
                name=f'{vehicle_type}',
                line=dict(color=colors.get(vehicle_type, '#636EFA'))
            ))

    entry_fig.update_layout(
        title='Vehicle Entry Counts Over Time',
        xaxis_title='Time',
        yaxis_title='Entry Count',
        height=400,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    # Create figure for exit counts
    exit_fig = go.Figure()
    for vehicle_type, data in vehicle_exit_series.items():
        if data and len(data.get('counts', [])) > 0:  # Check if data exists and has counts
            exit_fig.add_trace(go.Scatter(
                x=data['timestamps'],
                y=data['counts'],
                mode='lines+markers',
                name=f'{vehicle_type}',
                line=dict(color=colors.get(vehicle_type, '#636EFA'))
            ))

    exit_fig.update_layout(
        title='Vehicle Exit Counts Over Time',
        xaxis_title='Time',
        yaxis_title='Exit Count',
        height=400,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return entry_fig, exit_fig

def calculate_roi_vehicle_count(bbox, rois, image_shape):
    """
    Calculate the number of vehicles within each ROI with improved accuracy.
    
    :param bbox: List of bounding boxes of detected vehicles (x1, y1, x2, y2, confidence, class)
    :param rois: List of ROIs with their properties
    :param image_shape: Shape of the input image (height, width)
    :return: Dictionary with ROI IDs as keys and vehicle counts as values
    """
    roi_vehicle_counts = {roi['id']: 0 for roi in rois}
    height, width = image_shape[:2]

    for box in bbox:
        x1, y1, x2, y2 = [int(i) for i in box[:4]]
        
        # Calculate intersection ratio
        def calculate_intersection_ratio(box_coords, roi_vertices):
            # Box coordinates
            box_x1, box_y1, box_x2, box_y2 = box_coords
            box_area = max(1, (box_x2 - box_x1) * (box_y2 - box_y1))
            
            # ROI coordinates
            roi_x1 = min(v[0] for v in roi_vertices)
            roi_y1 = min(v[1] for v in roi_vertices)
            roi_x2 = max(v[0] for v in roi_vertices)
            roi_y2 = max(v[1] for v in roi_vertices)
            
            # Calculate intersection coordinates
            inter_x1 = max(box_x1, roi_x1)
            inter_y1 = max(box_y1, roi_y1)
            inter_x2 = min(box_x2, roi_x2)
            inter_y2 = min(box_y2, roi_y2)
            
            # Calculate intersection area
            inter_width = max(0, inter_x2 - inter_x1)
            inter_height = max(0, inter_y2 - inter_y1)
            inter_area = inter_width * inter_height
            
            # Calculate intersection ratio
            intersection_ratio = inter_area / box_area
            return intersection_ratio

        for roi in rois:
            # Convert ROI vertices to numpy array for OpenCV processing
            roi_vertices = np.array(roi['vertices'], np.int32)
            
            # Improved detection methods
            # 1. Check vehicle center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Use cv2.pointPolygonTest for more robust point-in-polygon test
            point_test = cv2.pointPolygonTest(roi_vertices, (center_x, center_y), False)
            
            # 2. Check intersection ratio
            intersection_ratio = calculate_intersection_ratio(
                (x1, y1, x2, y2), roi['vertices']
            )

            # Detection criteria:
            # - Point is inside the polygon (returns positive value) OR
            # - Intersection ratio is significant (more than 30%)
            if point_test >= 0 or intersection_ratio > 0.3:
                roi_vehicle_counts[roi['id']] += 1

    return roi_vehicle_counts

def update_lane_tracking_data(bbox, rois, image):
    """
    Update LANE_TRACKING_DATA with more accurate vehicle counts and status for each ROI.
    
    :param bbox: List of bounding boxes of detected vehicles
    :param rois: List of ROIs with their properties
    :param image: Current frame/image for context
    """
    roi_counts = calculate_roi_vehicle_count(bbox, rois, image.shape)

    for roi in rois:
        roi_id = roi['id']
        vehicle_count = roi_counts[roi_id]
        
        # Calculate ROI area more precisely
        roi_vertices = np.array(roi['vertices'], np.int32)
        roi_area = cv2.contourArea(roi_vertices)
        
        # Calculate density
        density = vehicle_count / (roi_area / 1000)
        
        # Enhanced status determination with vehicle movement analysis
        status = 'Free Flow'
        avg_speed = _calculate_vehicle_speed(bbox)
        movement_variation = _analyze_vehicle_movement(bbox)
        
        # More sophisticated traffic status determination
        if vehicle_count > 0:
            if vehicle_count <= 5 and avg_speed > 0.7:
                status = 'Light Traffic'
            elif (4 <= vehicle_count <= 8) and (0.3 < avg_speed <= 0.7):
                status = 'Moderate Traffic'
            elif (vehicle_count > 5) and movement_variation < 0.2:
                status = 'Heavy Traffic'
            elif (vehicle_count > 5) and movement_variation >= 0.2:
                status = 'Congested'
        
        config.LANE_TRACKING_DATA[roi_id] = {
            'vehicle_count': vehicle_count,
            'density': density,
            'status': status,
            'avg_speed': avg_speed,
            'movement_variation': movement_variation,
            'raw_area': roi_area,
            'log_density': round(density, 2)
        }

        print(f"ROI: {roi_id}, Status: {status}, Vehicles: {vehicle_count}, "
              f"Avg Speed: {avg_speed:.2f}, Movement Variation: {movement_variation:.2f}")

def _calculate_vehicle_speed(bbox):
    """
    Estimate vehicle speeds based on bounding box changes.
    
    :param bbox: List of bounding boxes
    :return: Normalized average speed (0-1 scale)
    """
    if len(bbox) < 2:
        return 1.0  # Assume free flow if few vehicles
    
    speeds = []
    for i in range(len(bbox) - 1):
        # Calculate relative movement between consecutive frames
        x1, y1, x2, y2 = [int(bbox[i][j]) for j in range(4)]
        x1_next, y1_next, x2_next, y2_next = [int(bbox[i+1][j]) for j in range(4)]
        
        # Calculate center points
        center_x1 = (x1 + x2) / 2
        center_y1 = (y1 + y2) / 2
        center_x2 = (x1_next + x2_next) / 2
        center_y2 = (y1_next + y2_next) / 2
        
        # Calculate relative movement magnitude
        movement = np.sqrt((center_x2 - center_x1)**2 + (center_y2 - center_y1)**2)
        speeds.append(movement)
    
    # Normalize speed (assuming larger movement means faster)
    if speeds:
        max_possible_speed = max(speeds)
        avg_speed = np.mean(speeds) / max_possible_speed
        return min(max(avg_speed, 0), 1)
    
    return 1.0  # Default to free flow

def _analyze_vehicle_movement(bbox):
    """
    Analyze variation in vehicle movements.
    
    :param bbox: List of bounding boxes
    :return: Movement variation (0-1 scale)
    """
    if len(bbox) < 2:
        return 0.0  # Minimal variation with few vehicles
    
    movements = []
    for i in range(len(bbox) - 1):
        x1, y1, x2, y2 = [int(bbox[i][j]) for j in range(4)]
        x1_next, y1_next, x2_next, y2_next = [int(bbox[i+1][j]) for j in range(4)]
        
        # Calculate relative movement
        movement_x = abs((x1_next + x2_next)/2 - (x1 + x2)/2)
        movement_y = abs((y1_next + y2_next)/2 - (y1 + y2)/2)
        
        movements.append(np.sqrt(movement_x**2 + movement_y**2))
    
    # Calculate variation in movements
    if movements:
        variation = np.std(movements) / np.mean(movements) if np.mean(movements) > 0 else 0
        return min(max(variation, 0), 1)
    
    return 0.0

def _display_detected_frames(conf, model, st_count, st_video, image):
    # Predict objects in the frame
    res = model.predict(image, conf=conf)

     # Draw ROI and entry/exit lines
    if hasattr(config, 'RECTANGLE_ROIS'):
        for roi in config.RECTANGLE_ROIS:
            pts = np.array(roi['vertices'], np.int32)
            cv2.polylines(image, [pts], isClosed=True,
                          color=roi['color'],
                          thickness=roi['thickness'])

    
    # Draw entry and exit polygonal zones
    cv2.polylines(image, [config.VEHICLE_ENTRY_ZONE], True, (0, 255, 0), 3)  # Entry zone (Green)
    cv2.polylines(image, [config.VEHICLE_EXIT_ZONE], True, (0, 0, 255), 3)    # Exit zone (Red)

    # Update ROI data
    update_lane_tracking_data(
    res[0].boxes.xyxy.tolist(), 
    config.RECTANGLE_ROIS, 
    image  # Pass the current image frame
)

    # Ensure all classes are displayed with counts initialized to 0 if not present
    all_classes = ['car', 'motorcycle', 'bus', 'truck', 'tricycle', 'mini-bus']
    in_text = "Vehicles Entering:\n"
    out_text = "Vehicles Leaving:\n"

    for vehicle_type in all_classes:
        entry_count = config.OBJECT_COUNTER_ENTRY.get(vehicle_type, 0)
        exit_count = config.OBJECT_COUNTER_EXIT.get(vehicle_type, 0)
        in_text += f" {vehicle_type}: {entry_count}\n"
        out_text += f" {vehicle_type}: {exit_count}\n"

    # Add ROI traffic status with new format
    roi_status_text = "REGION OF INTEREST TRAFFIC STATUS\n"
    if hasattr(config, 'LANE_TRACKING_DATA'):
        for roi_id, tracking in config.LANE_TRACKING_DATA.items():
            # Swap the labels to match the correct direction
            display_id = 'Entering ROI' if roi_id == 'Leaving_ROI' else 'Leaving ROI'
            roi_status_text += (
                f"{display_id}: {tracking['status']} - {tracking['vehicle_count']} vehicles\n"
            )
    # Display video frame in the left column
    res_plotted = res[0].plot()
    st_video.image(
        res_plotted,
        caption="Processed Video Frame",
        channels="BGR",
        use_container_width=True,
    )

    # Display counting information in the right column
    st_count.text(f"{in_text}\n{out_text}\n{roi_status_text}")

    # Update graphs and pie charts





@st.cache_resource
def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def infer_uploaded_video(conf, model):
    """
    Execute inference for uploaded video with vehicle count graph.
    :param conf: Confidence of YOLOv8 model.
    :param model: An instance of the YOLOv8 class containing the YOLOv8 model.
    :return: None.
    """
    source_video = st.sidebar.file_uploader(
        label="Choose a video...",
        type=["mp4", "avi", "mov"]
    )

    if source_video:
        st.video(source_video)

    if source_video:
        if st.button("Execution"):
            with st.spinner("Running..."):
                try:
                    config.OBJECT_COUNTER_ENTRY = defaultdict(int)
                    config.OBJECT_COUNTER_EXIT = defaultdict(int)
                    
                    # Initialize vehicle time series tracking
                    config.VEHICLE_TIME_SERIES = {
                        'entry': {
                            'car': {'counts': [], 'timestamps': []},
                            'motorcycle': {'counts': [], 'timestamps': []},
                            'bus': {'counts': [], 'timestamps': []},
                            'truck': {'counts': [], 'timestamps': []},
                            'tricycle': {'counts': [], 'timestamps': []},
                            'mini-bus': {'counts': [], 'timestamps': []}
                        },
                        'exit': {
                            'car': {'counts': [], 'timestamps': []},
                            'motorcycle': {'counts': [], 'timestamps': []},
                            'bus': {'counts': [], 'timestamps': []},
                            'truck': {'counts': [], 'timestamps': []},
                            'tricycle': {'counts': [], 'timestamps': []},
                            'mini-bus': {'counts': [], 'timestamps': []}
                        }
                    }
                    
                    # Reset lane tracking data
                    config.LANE_TRACKING_DATA = {
                        roi['id']: {
                            'vehicle_count': 0,
                            'density': 0,
                            'status': 'Free Flow'
                        } for roi in config.RECTANGLE_ROIS
                    }

                    # Temporary file to process the uploaded video
                    tfile = tempfile.NamedTemporaryFile(delete=False)
                    tfile.write(source_video.read())
                    vid_cap = cv2.VideoCapture(tfile.name)

                    # Get video properties
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))

                    # Create columns: wider for video, smaller for counting
                    col1, col2 = st.columns([3, 1])  # Adjust column widths (75% video, 25% counting)

                    with col1:
                        st_video = st.empty()  # Placeholder for video
                    with col2:
                        st_count = st.empty()  # Placeholder for counting information

                    # Create placeholders for graphs
                    st.subheader("Vehicle Count Over Time")
                    st_entry_graph = st.empty()
                    st_exit_graph = st.empty()

                    # Process frames
                    processed_frames = 0
                    start_time = datetime.now()
                    last_minute_timestamp = start_time
                    one_minute_frame_limit = int(fps * 60)  # Frames in one minute

                    while vid_cap.isOpened():
                        success, image = vid_cap.read()
                        if not success:
                            break
                        
                        current_time = start_time + timedelta(minutes=processed_frames/fps)
                        timestamp_str = current_time.strftime('%H:%M:%S')

                        # Update time series data for all vehicle types
                        for vehicle_type in ['car', 'motorcycle', 'bus', 'truck', 'tricycle', 'mini-bus']:
                            entry_count = config.OBJECT_COUNTER_ENTRY.get(vehicle_type, 0)
                            exit_count = config.OBJECT_COUNTER_EXIT.get(vehicle_type, 0)
                            
                            config.VEHICLE_TIME_SERIES['entry'][vehicle_type]['counts'].append(entry_count)
                            config.VEHICLE_TIME_SERIES['entry'][vehicle_type]['timestamps'].append(timestamp_str)
                            config.VEHICLE_TIME_SERIES['exit'][vehicle_type]['counts'].append(exit_count)
                            config.VEHICLE_TIME_SERIES['exit'][vehicle_type]['timestamps'].append(timestamp_str)

                        # Check if it's time to update the graphs
                        if (processed_frames % one_minute_frame_limit == 0) or processed_frames == total_frames - 1:
                            # Create and display the graphs
                            entry_fig, exit_fig = create_vehicle_count_line_graphs(
                                config.VEHICLE_TIME_SERIES['entry'],
                                config.VEHICLE_TIME_SERIES['exit']
                            )
                            
                            # Update existing graph placeholders
                            st_entry_graph.plotly_chart(entry_fig, use_container_width=True)
                            st_exit_graph.plotly_chart(exit_fig, use_container_width=True)

                        # Process and display the frame
                        _display_detected_frames(
                            conf,
                            model,
                            st_count,
                            st_video,
                            image
                        )

                        processed_frames += 1

                except Exception as e:
                    st.error(f"Error processing video: {e}")


def process_and_save_video(conf, model):
    source_video = st.sidebar.file_uploader(
        label="Choose a video to process...",
        type=["mp4", "avi", "mov"]
    )

    if source_video:
        # Display the original video
        st.subheader("Original Video")
        st.video(source_video)

        if st.button("Process Video"):
            # Progress tracking variables
            progress_bar = st.progress(0)
            status_text = st.empty()

            with st.spinner("Processing video..."):
                try:
                    # Initialize or reset counters and time series
                    config.OBJECT_COUNTER_ENTRY = defaultdict(int)
                    config.OBJECT_COUNTER_EXIT = defaultdict(int)
                    config.VEHICLE_TIME_SERIES = {
                        'entry': {
                            'car': {'counts': [], 'timestamps': []},
                            'motorcycle': {'counts': [], 'timestamps': []},
                            'bus': {'counts': [], 'timestamps': []},
                            'truck': {'counts': [], 'timestamps': []},
                            'tricycle': {'counts': [], 'timestamps': []},
                            'mini-bus': {'counts': [], 'timestamps': []}
                        },
                        'exit': {
                            'car': {'counts': [], 'timestamps': []},
                            'motorcycle': {'counts': [], 'timestamps': []},
                            'bus': {'counts': [], 'timestamps': []},
                            'truck': {'counts': [], 'timestamps': []},
                            'tricycle': {'counts': [], 'timestamps': []},
                            'mini-bus': {'counts': [], 'timestamps': []}
                        }
                    }

                    # Create a directory for processed videos
                    output_dir = Path("processed_videos")
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # Temporary file to process the uploaded video
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(source_video.read())
                    tfile.close()
                    
                    # Open the video
                    vid_cap = cv2.VideoCapture(tfile.name)
                    
                    # Get video properties
                    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    # Create a unique filename for the processed video
                    output_filename = f"processed_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                    output_path = output_dir / output_filename
                    
                    # VideoWriter object
                    fourcc = cv2.VideoWriter_fourcc(*'avc1')
                    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

                    # Process frames
                    processed_frames = 0
                    start_time = datetime.now()
                    
                    while vid_cap.isOpened():
                        success, frame = vid_cap.read()
                        if not success:
                            break
                        
                        # Draw entry and exit zone lines
                        cv2.polylines(frame, [config.VEHICLE_ENTRY_ZONE], True, (0, 255, 0), 2)
                        cv2.polylines(frame, [config.VEHICLE_EXIT_ZONE], True, (0, 0, 255), 2)
                        
                        # Predict objects in the frame
                        res = model.predict(frame, conf=conf)
                        
                        # Plot detected objects
                        annotated_frame = res[0].plot()
                        
                        # Current timestamp calculation
                        current_time = start_time + timedelta(minutes=processed_frames/fps)
                        timestamp_str = current_time.strftime('%H:%M:%S')
                        
                        # Text preparation
                        in_text = "Vehicles Entering:\n"
                        out_text = "Vehicles Leaving:\n" 
                        all_classes = ['car', 'motorcycle', 'bus', 'truck', 'tricycle', 'mini-bus']

                        # Update time series data
                        for vehicle_type in all_classes:
                            entry_count = config.OBJECT_COUNTER_ENTRY.get(vehicle_type, 0)
                            exit_count = config.OBJECT_COUNTER_EXIT.get(vehicle_type, 0)

                            config.VEHICLE_TIME_SERIES['entry'][vehicle_type]['counts'].append(entry_count)
                            config.VEHICLE_TIME_SERIES['entry'][vehicle_type]['timestamps'].append(timestamp_str)
                            
                            config.VEHICLE_TIME_SERIES['exit'][vehicle_type]['counts'].append(exit_count)
                            config.VEHICLE_TIME_SERIES['exit'][vehicle_type]['timestamps'].append(timestamp_str)

                            if entry_count > 0:
                                in_text += f" {vehicle_type}: {entry_count}\n"
                            if exit_count > 0:
                                out_text += f" {vehicle_type}: {exit_count}\n"
                        
                        # Place timestamp on the frame
                        cv2.putText(annotated_frame, timestamp_str, 
                                    (width - 150, height - 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                        # Define font, size, and color for text
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.7
                        font_color = (255, 255, 255)  # White
                        background_color = (0, 0, 0)  # Black
                        thickness = 2
                        margin = 10

                        # Place entry text inside the video
                        x_in, y_in = width - 220, margin + 20  # Adjusted for entry text
                        x_out, y_out = margin, margin + 20    # Adjusted for exit text

                        for i, line in enumerate(in_text.split('\n')):
                            text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
                            cv2.rectangle(annotated_frame, (x_in - 5, y_in - text_size[1] - 5), 
                                          (x_in + text_size[0] + 5, y_in + 5), background_color, -1)
                            cv2.putText(annotated_frame, line, (x_in, y_in), font, font_scale, font_color, thickness)
                            y_in += text_size[1] + 10

                        for i, line in enumerate(out_text.split('\n')):
                            text_size = cv2.getTextSize(line, font, font_scale, thickness)[0]
                            cv2.rectangle(annotated_frame, (x_out - 5, y_out - text_size[1] - 5), 
                                          (x_out + text_size[0] + 5, y_out + 5), background_color, -1)
                            cv2.putText(annotated_frame, line, (x_out, y_out), font, font_scale, font_color, thickness)
                            y_out += text_size[1] + 10
                        
                        # Write the annotated frame to the output video
                        out.write(annotated_frame)
                        
                        # Update progress
                        processed_frames += 1
                        progress = int((processed_frames / total_frames) * 100)
                        progress_bar.progress(progress)
                        
                        # Update status text
                        status_text.text(f"Processing: {processed_frames}/{total_frames} frames ({progress}%)")

                    # Release video objects
                    vid_cap.release()
                    out.release()

                    # Final progress update
                    progress_bar.progress(100)
                    status_text.text("Video processing complete!")

                    # Display success message
                    st.success(f"Video processed successfully! Saved as {output_filename}")
                    
                    # Display the processed video
                    st.subheader("Processed Video")
                    with open(output_path, "rb") as video_file:
                        st.video(video_file.read())
                    
                    # Generate and display final vehicle count graphs
                    st.subheader("Vehicle Count Over Time")
                    entry_fig, exit_fig = create_vehicle_count_line_graphs(config.VEHICLE_TIME_SERIES['entry'], config.VEHICLE_TIME_SERIES['exit'])
                    # Display the entry and exit graphs in a single column
                    st.plotly_chart(entry_fig, use_container_width=True)
                    st.plotly_chart(exit_fig, use_container_width=True)

                    
                    # Provide download button
                    with open(output_path, "rb") as file:
                        st.download_button(
                            label="Download Processed Video",
                            data=file.read(),
                            file_name=output_filename,
                            mime="video/mp4"
                        )

                except Exception as e:
                    st.error(f"Error processing video: {e}")
                    progress_bar.progress(0)
                    status_text.text("Video processing failed.")
                    
def process_and_save_images(conf, model):
    """
    Process uploaded images, save the processed images with detections.
    
    :param conf: Confidence of YOLOv8 model.
    :param model: An instance of the YOLOv8 class containing the YOLOv8 model.
    :return: None
    """
    source_images = st.sidebar.file_uploader(
        label="Choose images to process...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True  # Allow multiple image uploads
    )

    if source_images:
        st.subheader("Uploaded Images")
        for img in source_images:
            st.image(img, caption=img.name, use_column_width=True)

        if st.button("Process Images"):
            with st.spinner("Processing images..."):
                try:
                    # Reset counters
                    config.OBJECT_COUNTER_ENTRY = {
                        'car': 0, 'motorcycle': 0, 'bus': 0, 
                        'truck': 0, 'tricycle': 0, 'mini-bus': 0
                    }
                    config.OBJECT_COUNTER_EXIT = {
                        'car': 0, 'motorcycle': 0, 'bus': 0, 
                        'truck': 0, 'tricycle': 0, 'mini-bus': 0
                    }

                    # Create a directory for processed images
                    output_dir = Path("processed_images")
                    output_dir.mkdir(parents=True, exist_ok=True)

                    processed_images = []
                    for img in source_images:
                        # Read the image
                        file_bytes = img.read()
                        image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
                        
                        # Draw entry and exit zone lines
                        cv2.polylines(image, [config.VEHICLE_ENTRY_ZONE], True, (0, 255, 0), 2)
                        cv2.polylines(image, [config.VEHICLE_EXIT_ZONE], True, (0, 0, 255), 2)
                        
                        # Predict objects in the image
                        res = model.predict(image, conf=conf)
                        
                        # Plot detected objects
                        annotated_image = res[0].plot()
                        
                        # Get image dimensions
                        height, width = annotated_image.shape[:2]
                        
                        # Prepare vehicle count text
                        entry_text = "Vehicles Entering:\n"
                        exit_text = "Vehicles Leaving:\n"
                        
                        # Add vehicle counts to the text
                        vehicle_types = ['car', 'motorcycle', 'bus', 'truck', 'tricycle', 'mini-bus']
                        for vehicle in vehicle_types:
                            entry_text += f"{vehicle}: {config.OBJECT_COUNTER_ENTRY.get(vehicle, 0)}\n"
                            exit_text += f"{vehicle}: {config.OBJECT_COUNTER_EXIT.get(vehicle, 0)}\n"
                        
                        # Combine texts
                        full_text = entry_text + "\n" + exit_text
                        
                        # Add text to the image
                        cv2.rectangle(annotated_image, (width-250, 10), (width-10, 250), (255,255,255), -1)
                        y0 = 30
                        for i, line in enumerate(full_text.split('\n')):
                            cv2.putText(annotated_image, line, (width-240, y0 + i*25), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
                        
                        # Save the annotated image
                        output_path = output_dir / f"processed_{img.name}"
                        cv2.imwrite(str(output_path), annotated_image)
                        processed_images.append((annotated_image, output_path))

                    st.success("Images processed successfully!")

                    # Display processed images and provide download links
                    st.subheader("Processed Images")
                    for annotated_image, output_path in processed_images:
                        st.image(annotated_image, channels="BGR", use_column_width=True)
                        with open(output_path, "rb") as file:
                            st.download_button(
                                label=f"Download {output_path.name}",
                                data=file.read(),
                                file_name=output_path.name,
                                mime="image/jpeg"
                            )

                except Exception as e:
                    st.error(f"Error processing images: {e}")


def infer_rtsp_stream(conf, model):
    """
    Execute inference for RTSP stream with vehicle count graph.
    :param conf: Confidence of YOLOv8 model
    :param model: An instance of the YOLOv8 class containing the YOLOv8 model.
    :return: None
    """
    # RTSP Stream URL input
    rtsp_url = st.sidebar.text_input("Enter RTSP Stream URL", 
                                     placeholder="rtsp://username:password@ip_address:port/stream")
    
    if rtsp_url:
        try:
            # Reset counters and time series data
            config.OBJECT_COUNTER_ENTRY = defaultdict(int)
            config.OBJECT_COUNTER_EXIT = defaultdict(int)
            
            # Initialize vehicle time series tracking
            vehicle_entry_series = {
                'car': {'counts': [], 'timestamps': []},
                'motorcycle': {'counts': [], 'timestamps': []},
                'bus': {'counts': [], 'timestamps': []},
                'truck': {'counts': [], 'timestamps': []},
                'tricycle': {'counts': [], 'timestamps': []},
                'mini-bus': {'counts': [], 'timestamps': []}
            }
            vehicle_exit_series = {
                'car': {'counts': [], 'timestamps': []},
                'motorcycle': {'counts': [], 'timestamps': []},
                'bus': {'counts': [], 'timestamps': []},
                'truck': {'counts': [], 'timestamps': []},
                'tricycle': {'counts': [], 'timestamps': []},
                'mini-bus': {'counts': [], 'timestamps': []}
            }
            
            # Reset lane tracking data
            config.LANE_TRACKING_DATA = {
                roi['id']: {
                    'vehicle_count': 0,
                    'density': 0,
                    'status': 'Free Flow'
                } for roi in config.RECTANGLE_ROIS
            }

            # Open RTSP stream
            vid_cap = cv2.VideoCapture(rtsp_url)

            # Create columns: wider for video, smaller for counting
            col1, col2, col3 = st.columns([3, 1, 2])  # Adjust column widths

            with col1:
                st_video = st.empty()  # Placeholder for video
            with col2:
                st_count = st.empty()  # Placeholder for counting information
            with col3:
                st_graph = st.empty()  # Placeholder for real-time graph

            # Stop streaming button
            stop_button = st.sidebar.button("Stop Streaming")

            # Time tracking for graph updates
            start_time = datetime.now()
            last_graph_update = start_time
            graph_update_interval = 60  # Update graph every 60 seconds

            # Frame counter
            processed_frames = 0

            # Streaming loop
            while vid_cap.isOpened() and not stop_button:
                success, image = vid_cap.read()
                if success:
                    current_time = start_time + timedelta(minutes=processed_frames/30)  # Assuming 30 fps
                    timestamp_str = current_time.strftime('%H:%M:%S')

                    # Check if it's time to update the graph
                    if (current_time - last_graph_update).total_seconds() >= graph_update_interval:
                        # Compute and store aggregated counts for each vehicle type
                        for vehicle_type in ['car', 'motorcycle', 'bus', 'truck', 'tricycle', 'mini-bus']:
                            entry_count = config.OBJECT_COUNTER_ENTRY.get(vehicle_type, 0)
                            exit_count = config.OBJECT_COUNTER_EXIT.get(vehicle_type, 0)
                            
                            vehicle_entry_series[vehicle_type]['counts'].append(entry_count)
                            vehicle_entry_series[vehicle_type]['timestamps'].append(timestamp_str)
                            vehicle_exit_series[vehicle_type]['counts'].append(exit_count)
                            vehicle_exit_series[vehicle_type]['timestamps'].append(timestamp_str)
                        
                        # Create and display the graph
                        count_fig = create_vehicle_count_line_graphs(vehicle_entry_series, vehicle_exit_series)
                        st_graph.plotly_chart(count_fig, use_container_width=True)
                        
                        # Update last graph update time
                        last_graph_update = current_time

                    # Process and display the frame
                    _display_detected_frames(
                        conf,
                        model,
                        st_count,
                        st_video,
                        image
                    )
                    
                    processed_frames += 1

                    # Add a small delay to control frame rate
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    st.error("Failed to read stream. Check RTSP URL.")
                    break

            # Release the video capture object
            vid_cap.release()

        except Exception as e:
            st.error(f"Error processing RTSP stream: {e}")