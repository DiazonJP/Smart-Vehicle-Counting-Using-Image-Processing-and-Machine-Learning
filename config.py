from pathlib import Path
import sys
from collections import defaultdict
from datetime import datetime
import numpy as np

file_path = Path(__file__).resolve()

root_path = file_path.parent

LANE_TRACKING_DATA = {
    'Entering_ROI': {
        'vehicle_count': 0,
        'density': 0,
        'status': 'free-flow',
        'avg_vehicle_speed': 0,
        'occupancy_time': 0
    },
    'Leaving_ROI': {
        'vehicle_count': 0,
        'density': 0,
        'status': 'free-flow',
        'avg_vehicle_speed': 0,
        'occupancy_time': 0
    }
}

RECTANGLE_ROIS = [
    {
        'id': 'Leaving_ROI',
        'vertices': [
            (650, 400),   # Top-left 
            (810, 400),   # Top-right
            (810, 1000),   # Bottom-right
            (50, 1000)    # Bottom-left
        ],
        'color': (0, 0, 255),  
        'thickness': 3,
        'detection_sensitivity': 0.7,  
        'min_object_size': 20,  
        'max_object_size': 500  
    },
    {
        'id': 'Entering_ROI',
        'vertices': [
            (830, 400),   
            (900, 400), 
            (1800, 1000),  
            (850, 1000)   
        ],
        'color': (0, 255, 0),  
        'thickness': 3,
        'detection_sensitivity': 0.6, 
        'min_object_size': 15,
        'max_object_size': 450
    }
]

OBJECT_COUNTER_ENTRY = {
    'tricycle': 0,
    'motorcycle': 0,
    'car': 0,
    'bus': 0,
    'truck': 0,
    'mini-bus': 0,
    
}

OBJECT_COUNTER_EXIT = {
    'tricycle': 0,
    'motorcycle': 0,
    'car': 0,
    'bus': 0,
    'truck': 0,
    'mini-bus': 0,
}

CROSSED_IDS = set()  

VEHICLE_ENTRY_ZONE = np.array([
    [920, 500],   # Top-left
    [1610, 500],   # Top-right
    [1610, 580],   # Bottom-right
    [920, 580]    # Bottom-left
], np.int32)


VEHICLE_EXIT_ZONE = np.array([
    [80, 500],    # Top-left
    [860, 500],    # Top-right
    [860, 580],    # Bottom-right
    [80, 580]     # Bottom-left
], np.int32)




def point_inside_polygon(point, polygon):
    x, y = point
    inside = False
    j = len(polygon) - 1
    
    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi)
        
        if intersect:
            inside = not inside
        
        j = i
    
    return inside

OBJECT_COUNTER_ENTRY = {}
OBJECT_COUNTER_EXIT = {}

OBJECT_TRACKING_STATE = {}
VEHICLE_TIME_SERIES = {
    'entry': defaultdict(list),
    'exit': defaultdict(list)
}

OBJECT_SIZE_THRESHOLD = 30     # Minimum pixel size for valid object detection

LINE_PROXIMITY_THRESHOLD = 10  # Adjust based on resolution and use case

if root_path not in sys.path:
    sys.path.append(str(root_path))

ROOT = root_path.relative_to(Path.cwd())

SOURCES_LIST = ["Video", "RTSP", "Processed Video", "Processed Image"]

PROCESSED_VIDEO_DIR = ROOT / 'processed_videos'

PROCESSED_VIDEO_DIR.mkdir(exist_ok=True)

DETECTION_MODEL_DIR = ROOT / 'weights' / 'detection'
YOLOv8n = DETECTION_MODEL_DIR / "yolov8n.pt"
YOLOv8s = DETECTION_MODEL_DIR / "yolov8s.pt"
YOLOv8m = DETECTION_MODEL_DIR / "yolov8m.pt"
YOLOv8l = DETECTION_MODEL_DIR / "yolov8l.pt"
YOLOv8x = DETECTION_MODEL_DIR / "yolov8x.pt"
YOLOv8n = DETECTION_MODEL_DIR / "dataset_03.pt"
YOLOv8m = DETECTION_MODEL_DIR / "dataset_06.pt"
YOLOv8l = DETECTION_MODEL_DIR / "dataset_07.pt"
YOLOv8l = DETECTION_MODEL_DIR / "dataset_08.pt"
YOLOv8l = DETECTION_MODEL_DIR / "dataset_09.pt"


DETECTION_MODEL_LIST = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolov8l.pt",
    "yolov8x.pt",
    "dataset_03.pt",
    "dataset_06.pt",
    "dataset_07.pt",
    "dataset_08.pt",
    "dataset_09.pt"]


OBJECT_COUNTER = None
OBJECT_COUNTER1 = None