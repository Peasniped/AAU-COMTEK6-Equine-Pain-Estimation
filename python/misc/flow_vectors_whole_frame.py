import cv2
import numpy as np

def draw_optical_flow_grid(prev_gray, gray, frame, grid_points_wide=50):
    # Calculate dense optical flow using Farneback
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
                                        None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = prev_gray.shape

    # Compute magnitude and angle for color-coding
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

    # Normalize magnitude for coloring (0-255)
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # Sample grid points
    step_x = w // grid_points_wide
    step_y = h // (h // step_x)  # Keep the grid roughly square

    for y in range(0, h, step_y):
        for x in range(0, w, step_x):
            dx = flow[y, x, 0]
            dy = flow[y, x, 1]
            m = mag_norm[y, x]

            # Map magnitude to color (blue -> green -> red)
            # You can use a simple colormap, e.g., cv2.COLORMAP_JET
            color = cv2.applyColorMap(np.uint8([[m]]), cv2.COLORMAP_JET)[0][0].tolist()

            # Draw arrowed line
            end_point = (int(x + dx*2), int(y + dy*2))
            cv2.arrowedLine(frame, (x, y), end_point, color, 2, tipLength=0.4)

    return frame

def main(video_path):
    cap = cv2.VideoCapture(video_path)

    ret, prev = cap.read()
    if not ret:
        print("Failed to read video.")
        return
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        vis = draw_optical_flow_grid(prev_gray, gray, frame.copy(), 75)

        cv2.imshow('Dense Optical Flow Grid', vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_gray = gray

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("video/S10_Video.mp4")
