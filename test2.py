import cv2
import os
import sys
import traceback
import torch
from ultralytics import YOLO

MODEL_PATH = "runs/detect/vehicle_model_v114/weights/best.pt"
VIDEO_PATH = "TRAFFIC_TEST.mp4"


def main():
    # 1. PATH TO YOUR NEW BRAIN
    if not os.path.exists(MODEL_PATH):
        print(f"Error: model file not found: {MODEL_PATH}")
        print("Make sure your best.pt is in the correct runs folder.")
        return

    # 2. LOAD YOUR CUSTOM MODEL
    model = YOLO(MODEL_PATH)

    # 3. OPEN VIDEO
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: video file not found: {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: could not open video: {VIDEO_PATH}")
        return

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # 4. RUN TRACKING
        device = 'cpu' if not torch.cuda.is_available() else 0
        results = model.track(frame, persist=True, device=device, conf=0.6)

        for r in results:
            if r.boxes is None:
                continue

            # Extract data
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy()
            ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else []

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                conf = confs[i]
                cls_id = int(clss[i])
                label_name = model.names[cls_id]

                

                # Ground Point (for your future mapping)
                cx, cy = int((x1 + x2) / 2), int(y2)

                # --- VISUALS ---
                obj_id = int(ids[i]) if len(ids) > 0 else "?"
                display_text = f"{label_name} {obj_id} ({conf:.2f})"

                # Select color per class
                if label_name == 'truck':
                    box_color = (0, 0, 255)       # red
                    text_color = (0, 0, 255)
                    dot_color = (0, 0, 255)
                elif label_name == 'bus':
                    box_color = (0, 255, 255)     # yellow
                    text_color = (0, 255, 255)
                    dot_color = (0, 255, 255)
                else:
                    box_color = (255, 0, 0)
                    text_color = (255, 255, 0)
                    dot_color = (0, 0, 255)

                # Draw Box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), box_color, 2)
                
                # Draw Label
                cv2.putText(frame, display_text, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                
                # Draw Ground Dot
                cv2.circle(frame, (cx, cy), 5, dot_color, -1)

        # 5. SHOW RESULTS
        cv2.imshow("Custom Vehicle AI - Testing", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("An error occurred while running test 2.py:")
        traceback.print_exc()