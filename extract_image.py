import cv2
import glob
import os


def extract_frames(video_path, output_folder, interval_seconds=2, start_index=0):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    frame_interval = max(1, int(fps * interval_seconds))

    frame_count = 0
    saved_count = 0
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    print(f"Processing {video_path} @ {interval_seconds}s interval...")

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_count % frame_interval == 0:
            filename = f"{video_name}_frame_{start_index + saved_count:04d}.jpg"
            filepath = os.path.join(output_folder, filename)
            cv2.imwrite(filepath, frame)
            saved_count += 1
            print(f"  Saved: {filename}")

        frame_count += 1

    cap.release()
    return saved_count


def extract_all_videos(source_folder, output_folder, interval_seconds=2):
    video_patterns = ["*.mp4", "*.mov", "*.avi", "*.mkv"]
    video_files = []

    for pattern in video_patterns:
        video_files.extend(glob.glob(os.path.join(source_folder, pattern)))

    if not video_files:
        print(f"No video files found in {source_folder}")
        return

    total_saved = 0
    current_index = 0

    for video_path in sorted(video_files):
        saved = extract_frames(video_path, output_folder, interval_seconds, start_index=current_index)
        current_index += saved
        total_saved += saved

    print(f"\nDone! Extracted {total_saved} images from {len(video_files)} videos into '{output_folder}'.")


if __name__ == "__main__":
    extract_all_videos(
        source_folder="New YOLO",
        output_folder="my_custom_dataset_images",
        interval_seconds=2,
    )
