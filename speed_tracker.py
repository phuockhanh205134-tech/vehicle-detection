import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import defaultdict

# --- 1. SPATIAL CALIBRATION (Derived from forming matrix.py) ---
pts_src = np.array([
    (526, 6), (480, 274), (338, 497), (52, 822), (1532, 790), 
    (1526, 396), (1200, 526), (975, 227), (785, 71), (695, 8)
], dtype=np.float32)

pts_dst = np.array([
    [0, 0], [0, 250], [0, 500], [0, 1000], [500, 1000], 
    [500, 500], [500, 250], [500, 0], [250, 0], [100, 0]
], dtype=np.float32)

M, _ = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)

# --- 2. CONFIGURATION & CONSTANTS ---
# Reference lane width at known distance for calibration
REFERENCE_LANE_WIDTH_PIXELS = 50  # Pixels at mid-distance in video
REFERENCE_LANE_WIDTH_METERS = 10.0  # Real lane is 3 meters
SPEED_LIMIT = 50  # km/h

track_history = defaultdict(lambda: {"prev_pos": None, "speed": 0, "distances": []})

def get_box_width(box):
    """Get bounding box width"""
    return box[2] - box[0]

def estimate_distance_scale(box_width):
    """Estimate perspective scale factor based on box width"""
    # Cars closer to camera have larger bounding boxes
    # Use box width as proxy for distance
    scale = REFERENCE_LANE_WIDTH_PIXELS / max(box_width, 10)
    return max(0.1, min(2.0, scale))  # Clamp between 0.1 and 2.0

def calculate_violation_fine(speed):
    over_limit = speed - SPEED_LIMIT
    if over_limit <= 0: return 0
    if over_limit <= 10: return 200000   # Tier 1
    if over_limit <= 20: return 1000000  # Tier 2
    return 5000000                       # Tier 3 (Severe)

def main():
    model = YOLO("runs/detect/vehicle_model_v114/weights/best.pt")
    cap = cv2.VideoCapture("TRAFFIC_TEST.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    device = 'cpu' if not torch.cuda.is_available() else 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        results = model.track(frame, persist=True, conf=0.6, device=device)

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, obj_id in zip(boxes, ids):
                # Map bottom-center of box to BEV ground plane
                cx, cy = (box[0] + box[2]) / 2, box[3]
                point = np.array([[[cx, cy]]], dtype=np.float32)
                transformed = cv2.perspectiveTransform(point, M)
                tx, ty = transformed[0][0][0], transformed[0][0][1]

                # Estimate distance scale based on box width
                box_width = get_box_width(box)
                distance_scale = estimate_distance_scale(box_width)
                
                # Speed Calculation Logic with perspective correction
                if track_history[obj_id]["prev_pos"] is not None:
                    prev_x, prev_y = track_history[obj_id]["prev_pos"]
                    pixel_dist = np.sqrt((tx - prev_x)**2 + (ty - prev_y)**2)
                    
                    # Apply distance normalization: farther vehicles move less
                    # Correct for perspective by using distance_scale
                    normalized_dist = pixel_dist * distance_scale
                    
                    # Convert to km/h: (normalized_dist * 0.05m/unit) * FPS * 3.6
                    current_speed = (normalized_dist * 0.05) * fps * 3.6
                    
                    # Apply strong smoothing for stability
                    track_history[obj_id]["speed"] = (track_history[obj_id]["speed"] * 0.7) + (current_speed * 0.3)
                    track_history[obj_id]["distances"].append(pixel_dist)

                track_history[obj_id]["prev_pos"] = (tx, ty)
                speed = track_history[obj_id]["speed"]

                # Visualization and Enforcement
                is_speeding = speed > SPEED_LIMIT
                color = (0, 0, 255) if is_speeding else (0, 255, 0)
                
                info_text = f"ID:{obj_id} {speed:.1f}km/h"
                if is_speeding:
                    fine = calculate_violation_fine(speed)
                    info_text += f" FINE: {fine:,} VND"

                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                cv2.putText(frame, info_text, (int(box[0]), int(box[1]) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("Automated Traffic Enforcement System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()