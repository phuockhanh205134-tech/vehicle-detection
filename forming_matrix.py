import cv2
import numpy as np

# 1. YOUR 10 SOURCE POINTS (Captured from the curve/intersection)
pts_src = np.array([
    (526, 6), (480, 274), (338, 497), (52, 822), (1532, 790), 
    (1526, 396), (1200, 526), (975, 227), (785, 71), (695, 8)
], dtype=np.float32)

# 2. DEFINE THE 10 TARGET POINTS (The 'Bird's Eye' result)
# We map these to a 500x1000 pixel rectangular grid.
# I have organized these to represent a straight road segment.
width, height = 500, 1000
pts_dst = np.array([
    [0, 0], [0, 250], [0, 500], [0, 1000],  # Left side of road
    [500, 1000], [500, 500], [500, 250],    # Right side of road
    [500, 0], [250, 0], [100, 0]           # Top cross-section
], dtype=np.float32)

# 3. CALCULATE HOMOGRAPHY (Using all 10 points)
# cv2.RANSAC helps ignore any points that were clicked slightly off-center
M, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)

print("Matrix calculated using 10-point fit.")

# 4. FUNCTION TO TRANSFORM YOLO DOTS
def map_to_2d(video_x, video_y):
    # Prepare point for transformation
    point = np.array([[[video_x, video_y]]], dtype=np.float32)
    # Transform using the Matrix
    transformed = cv2.perspectiveTransform(point, M)
    return int(transformed[0][0][0]), int(transformed[0][0][1])

# --- TEST ON VIDEO ---
cap = cv2.VideoCapture("TRAFFIC TEST.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # Warp the whole frame to see if the road looks straight now
    bev_frame = cv2.warpPerspective(frame, M, (width, height))
    
    cv2.imshow("Warped (Straightened) View", bev_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break