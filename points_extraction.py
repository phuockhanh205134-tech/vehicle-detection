import cv2
import numpy as np

pts = []

def draw_polygon(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        pts.pop() # Undo last point

# Load frame
cap = cv2.VideoCapture("TRAFFIC_TEST.mp4")
_, frame = cap.read()
clone = frame.copy()
cv2.namedWindow("Define Intersection Zone")
cv2.setMouseCallback("Define Intersection Zone", draw_polygon)

print("Left Click to add points around the intersection. Right Click to undo. Press 'q' when finished.")

while True:
    img = clone.copy()
    if len(pts) > 0:
        # Draw the lines between points
        cv2.polylines(img, [np.array(pts)], isClosed=True, color=(0, 255, 255), thickness=2)
        for p in pts:
            cv2.circle(img, p, 5, (0, 0, 255), -1)

    cv2.imshow("Define Intersection Zone", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print(f"Your Intersection Polygon: {pts}")