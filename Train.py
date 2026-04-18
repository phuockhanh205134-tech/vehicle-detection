from ultralytics import YOLO

def train_custom_yolo():
    # Load model
    model = YOLO("yolo11n.pt")

    # Start training
    results = model.train(
        data="YOLO.vi.yolov11/data.yaml",
        epochs=250,
        imgsz=640,
        device='0',             
        batch=8,           
        workers=0,        
        name="vehicle_model_v1"
    )

if __name__ == "__main__":
    train_custom_yolo()
