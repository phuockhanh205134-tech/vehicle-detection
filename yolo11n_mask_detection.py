import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Configuration
MODEL_PATH = "yolo11n-seg.pt"
VIDEO_PATH = "New YOLO/IMG_0485.MOV"
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

def get_class_color(class_id):
    """Get consistent color for each class"""
    colors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 0),    # Dark Blue
        (0, 128, 0),    # Dark Green
        (0, 0, 128),    # Dark Red
        (128, 128, 0),  # Dark Cyan
    ]
    return colors[class_id % len(colors)]

def draw_masks_on_frame(frame, results, alpha=0.3):
    """Draw segmentation masks on the frame"""
    output_frame = frame.copy()
    
    if results[0].masks is None:
        return output_frame
    
    masks = results[0].masks.data.cpu().numpy()
    boxes = results[0].boxes.data.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    
    mask_overlay = output_frame.copy()
    
    for i, mask in enumerate(masks):
        # Resize mask to frame dimensions
        if mask.shape != (output_frame.shape[0], output_frame.shape[1]):
            mask = cv2.resize(mask, (output_frame.shape[1], output_frame.shape[0]))
        
        # Get consistent color for this class
        class_id = int(classes[i])
        color = get_class_color(class_id)
        
        # Apply mask to overlay
        mask_binary = (mask > 0.5).astype(np.uint8) * 255
        mask_pixels = mask_binary > 0
        mask_overlay[mask_pixels] = color
        
        # Draw bounding box
        x1, y1, x2, y2 = map(int, boxes[i][:4])
        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        class_name = results[0].names[class_id]
        label = f"{class_name} {confs[i]:.2f}"
        cv2.putText(output_frame, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Blend mask overlay with original frame
    output_frame = cv2.addWeighted(output_frame, 1 - alpha, mask_overlay, alpha, 0)
    return output_frame


def main():
    """Main function"""
    print("Loading YOLOv11n-seg model...")
    model = YOLO(MODEL_PATH)
    
    print(f"Processing video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file: {VIDEO_PATH}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height} @ {fps} FPS, {total_frames} frames")
    print("Press ESC to exit\n")
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    frame_count = 0
    paused = False
    current_frame = None
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Video finished!")
                break
            
            frame_count += 1
            
            # Run inference
            results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, 
                                   iou=IOU_THRESHOLD, device=device, verbose=False)
            
            # Draw masks
            current_frame = draw_masks_on_frame(frame, results, alpha=0.3)
            
            if frame_count % 30 == 0:
                print(f"Frame {frame_count}/{total_frames}")
        
        # Display
        if current_frame is not None:
            cv2.imshow("YOLOv11n Segmentation", current_frame)
        
        # Handle keys
        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == 27:  # ESC
            print("Exiting...")
            break
        elif key == 32:  # SPACE
            paused = not paused
            print("Paused" if paused else "Resumed")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
