# Vehicle Detection

A simple vehicle detection and tracking repository using a custom YOLO model built with Ultralytics.

## Project structure

- `Train.py` - trains a custom YOLO model using Ultralytics.
- `test2.py` - runs detection and tracking on `TRAFFIC_TEST.mp4` using the trained model.
- `YOLO.vii.yolov11/` - dataset configuration and labeled data for training.
- `runs/detect/vehicle_model_v114/weights/best.pt` - example trained model path used by `test2.py`.

## Requirements

- Python 3.10+ (project was tested with Python 3.14)
- `ultralytics`
- `opencv-python`
- `torch` (or `torchvision` as required by Ultralytics)
- `lap` (needed for tracker support)

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

> If you are using a virtual environment, activate it first.

## Training

To train a custom model, run:

```bash
python Train.py
```

`Train.py` uses the Ultralytics YOLO API and the dataset configuration at `YOLO.vii.yolov11/data.yaml`.

## Testing

To test the detection and tracking on the sample video, run:

```bash
python test2.py
```

`test2.py` expects:

- a trained model at `runs/detect/vehicle_model_v114/weights/best.pt`
- a video file named `TRAFFIC_TEST.mp4`

If the model or video file is missing, the script will print an error message.

## Notes

- `test2.py` uses `model.track(...)` with `conf=0.6` and automatically switches to GPU if available.
- It draws bounding boxes, labels, and ground points for each detected object.
- Press `q` in the output window to stop the video playback.

## Optional improvements

- Add a `requirements.txt` file for easier environment setup.
- Add a `README` section with example results and expected classes.
- Save detection output video or logs to disk.
