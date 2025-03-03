#predict.py
import torch
import cv2
from numpy import random
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.utils.plotting import Annotator
import config
from ultralytics.yolo.v8.detect.deep_sort_pytorch.utils.parser import get_config
from ultralytics.yolo.v8.detect.deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
import numpy as np
from os import getcwd

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

palette = (2 * 11 - 1, 2 * 15 - 1, 2 ** 20 - 1)
data_deque = {}

deepsort = None

object_counter = {}

object_counter1 = {}



cwd = getcwd()
cfg_deep = get_config()
cfg_deep.merge_from_file(cwd+"/ultralytics/yolo/v8/detect/deep_sort_pytorch/configs/deep_sort.yaml")

deepsort= DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

def xyxy_to_xywh(xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    x_c = (xyxy[..., 0] + xyxy[..., 2]) / 2  # x center
    y_c = (xyxy[..., 1] + xyxy[..., 3]) / 2  # y center
    w = xyxy[..., 2] - xyxy[..., 0]  # width
    h = xyxy[..., 3] - xyxy[..., 1]  # height
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 0: #person
        color = (85,45,255)
    elif label == 2: # Car
        color = (222,82,175)
    elif label == 3:  # Motobike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    
    cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 -r, y2-r), 2, color, 12)
    
    return img

def UI_box(x, img, color=None, label=None, line_thickness=None):
    """
    Draw a bounding box with very thin lines using transparency and anti-aliasing.
    """
    # Initialize overlay for blending
    overlay = img.copy()
    alpha = 0.3  # Reduced transparency for thinner appearance
    
    # Reduce default line thickness
    tl = line_thickness or round(sum(img.shape) / 4000)  # Significantly reduced default thickness
    tl = max(tl, 1)  # Ensure minimum thickness of 1
    
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    
    # Draw main box with anti-aliasing
    cv2.rectangle(overlay, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    
    # Blend overlay with original image
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
    if label:
        tf = max(tl - 1, 1)  # Font thickness
        font_scale = 0.3  # Reduced font scale
        t_size = cv2.getTextSize(label, 0, fontScale=font_scale, thickness=tf)[0]
        
        # Create label background with transparency
        label_overlay = img.copy()
        cv2.rectangle(
            label_overlay,
            (c1[0], c1[1] - t_size[1] - 3),
            (c1[0] + t_size[0], c1[1] + 3),
            color,
            thickness=-1
        )
        img = cv2.addWeighted(label_overlay, alpha, img, 1 - alpha, 0)
        
        # Draw label text
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            font_scale,
            [255, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA
        )

def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def get_direction(point1, point2):
    direction_str = ""

    # calculate y axis direction
    if point1[1] > point2[1]:
        direction_str += "South"
    elif point1[1] < point2[1]:
        direction_str += "North"
    else:
        direction_str += ""

    # calculate x axis direction
    if point1[0] > point2[0]:
        direction_str += "East"
    elif point1[0] < point2[0]:
        direction_str += "West"
    else:
        direction_str += ""

    return direction_str

def draw_boxes(img, bbox, names, object_id, identities=None, offset=(0, 0)):
    # Draw polygonal zones
    cv2.polylines(img, [config.VEHICLE_ENTRY_ZONE], True, (0, 255, 0), 3)  # Green
    cv2.polylines(img, [config.VEHICLE_EXIT_ZONE], True, (0, 0, 255), 3)   # Red
    
    height, width, _ = img.shape

    # Remove tracked point from buffer if object is lost
    for key in list(data_deque):
        if key not in identities:
            data_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # Calculate object's center and bounding box area
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        box_area = (x2 - x1) * (y2 - y1)
        id = int(identities[i]) if identities is not None else 0
        
        if id not in data_deque:
            data_deque[id] = deque(maxlen=64)
        
        if id not in config.OBJECT_TRACKING_STATE:
            config.OBJECT_TRACKING_STATE[id] = {
                'in_entry_zone': False,
                'in_exit_zone': False,
                'counted_entry': False,
                'counted_exit': False,
                'previous_entry_overlap': 0,
                'previous_exit_overlap': 0
            }

        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        
        data_deque[id].appendleft(center)
        
        # Enhanced zone detection with area overlap
        def calculate_polygon_overlap(box, polygon):
            """
            Calculate the percentage of box overlap with the polygon
            """
            # Create a mask for the polygon
            polygon_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(polygon_mask, [polygon], 255)
            
            # Create a mask for the bounding box
            box_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.rectangle(box_mask, (x1, y1), (x2, y2), 255, -1)
            
            # Calculate overlap
            overlap = cv2.bitwise_and(polygon_mask, box_mask)
            overlap_area = np.sum(overlap > 0)
            
            # Calculate overlap percentage
            return (overlap_area / box_area) * 100 if box_area > 0 else 0

        # Calculate overlap with entry and exit zones
        entry_overlap = calculate_polygon_overlap((x1, y1, x2, y2), config.VEHICLE_ENTRY_ZONE)
        exit_overlap = calculate_polygon_overlap((x1, y1, x2, y2), config.VEHICLE_EXIT_ZONE)
        
        tracking_state = config.OBJECT_TRACKING_STATE[id]
        
        # Improved Entry Zone Counting Logic
        OVERLAP_THRESHOLD = 20  # Percentage of box overlap to trigger counting
        CONSECUTIVE_FRAME_THRESHOLD = 3  # Number of frames with consistent overlap
        
        # Entry Zone Tracking
        if entry_overlap > OVERLAP_THRESHOLD:
            tracking_state['in_entry_zone'] = True
            tracking_state['previous_entry_overlap'] += 1
        else:
            tracking_state['previous_entry_overlap'] = 0
            tracking_state['in_entry_zone'] = False
        
        if (tracking_state['previous_entry_overlap'] >= CONSECUTIVE_FRAME_THRESHOLD and 
            not tracking_state['counted_entry']):
            config.OBJECT_COUNTER_ENTRY[obj_name] = config.OBJECT_COUNTER_ENTRY.get(obj_name, 0) + 1
            tracking_state['counted_entry'] = True
            tracking_state['previous_entry_overlap'] = 0
        
        # Exit Zone Tracking (similar logic)
        if exit_overlap > OVERLAP_THRESHOLD:
            tracking_state['in_exit_zone'] = True
            tracking_state['previous_exit_overlap'] += 1
        else:
            tracking_state['previous_exit_overlap'] = 0
            tracking_state['in_exit_zone'] = False
        
        if (tracking_state['previous_exit_overlap'] >= CONSECUTIVE_FRAME_THRESHOLD and 
            not tracking_state['counted_exit']):
            config.OBJECT_COUNTER_EXIT[obj_name] = config.OBJECT_COUNTER_EXIT.get(obj_name, 0) + 1
            tracking_state['counted_exit'] = True
            tracking_state['previous_exit_overlap'] = 0
        
        # Draw bounding boxes and label
        UI_box(box, img, label=f"{obj_name} {id}", color=color, line_thickness=1)

        # Draw trajectory trail
        for i in range(1, len(data_deque[id])):
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            thickness = max(1, int(np.sqrt(64 / float(i + i))))
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)

    return img

def display_count(img, width, counter, title):
    """Helper function to display vehicle count."""
    for idx, (key, value) in enumerate(counter.items()):
        cnt_str = f"{key}: {value}"
        if title == "Entering":
            pos_x, pos_y = width - 500, 35
            cv2.line(img, (width - 500, 25), (width, 25), [85, 45, 255], 40)
            cv2.putText(img, f'Number of Vehicles {title}', (pos_x, pos_y), 0, 1, [225, 255, 255], thickness=2)
        else:
            pos_x, pos_y = 20, 35
            cv2.line(img, (20, 25), (500, 25), [85, 45, 255], 40)
            cv2.putText(img, f'Number of Vehicles {title}', (pos_x, pos_y), 0, 1, [225, 255, 255], thickness=2)
        cv2.line(img, (pos_x, 65 + (idx * 40)), (width, 65 + (idx * 40)), [85, 45, 255], 30)
        cv2.putText(img, cnt_str, (pos_x, 75 + (idx * 40)), 0, 1, [255, 255, 255], thickness=2)

class DetectionPredictor(BasePredictor):

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_width, example=str(self.model.names))

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_img[i] if isinstance(orig_img, list) else orig_img
            if not isinstance(orig_img, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results
    
    def write_results(self, idx, results, batch):
        """Write inference results to a file or directory."""
        p, im, im0 = batch
        log_string = ''
        all_outputs = []
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.source_type.webcam or self.source_type.from_img:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        result = results[idx]
        log_string += result.verbose()

        if self.args.save or self.args.show:  # Add bbox to image
            plot_args = dict(line_width=self.args.line_width,
                             boxes=self.args.boxes,
                             conf=self.args.show_conf,
                             labels=self.args.show_labels)
            if not self.args.retina_masks:
                plot_args['im_gpu'] = im[idx]
            self.plotted_img = result.plot(**plot_args)
        self.annotator = self.get_annotator(im0)

        all_outputs.append(result)

        if len(result) == 0:
            return log_string
        for c in result.boxes.cls.unique():
            n = (result.boxes.cls == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
                
        # Write
        if self.args.save_txt:
            result.save_txt(f'{self.txt_path}.txt', save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / 'crops', file_name=self.data_path.stem)

        xywh_bboxs = []
        confs = []
        oids = []
        outputs = []
        for r in result:
            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(r.boxes.xyxy)
            xywh_obj = [x_c, y_c, bbox_w, bbox_h]
            xywh_bboxs.append(xywh_obj)
            confs.append([r.boxes.conf.item()])
            oids.append(int(r.boxes.cls))
        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)
          
        outputs = deepsort.update(xywhs, confss, oids, im0)
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]
            
            draw_boxes(im0, bbox_xyxy, self.model.names, object_id,identities)

        return log_string


def predict(cfg=DEFAULT_CFG, use_python=False):
    """Runs YOLO model inference on input image(s)."""
    model = cfg.model or 'yolov8n.pt'
    source = cfg.source if cfg.source is not None else ROOT / 'assets' if (ROOT / 'assets').exists() \
        else 'https://ultralytics.com/images/bus.jpg'

    args = dict(model=model, source=source)
    if use_python:
        from ultralytics import YOLO
        YOLO(model)(**args)
    else:
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()


if __name__ == "main":
    predict()