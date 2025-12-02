import cv2
import numpy as np
import time
import math

# ---------- Configurable Parameters ----------

# Virtual object: rectangle in the middle of the screen
RECT_WIDTH = 200
RECT_HEIGHT = 200

# Distance thresholds (in pixels) - tune these
SAFE_THRESHOLD = 150      # > SAFE_THRESHOLD -> SAFE
WARNING_THRESHOLD = 80    # between WARNING_THRESHOLD and SAFE_THRESHOLD -> WARNING
# <= WARNING_THRESHOLD -> DANGER

# Skin color HSV range (you might need to adjust these for your lighting/skin)
LOWER_SKIN = np.array([0, 30, 60], dtype=np.uint8)
UPPER_SKIN = np.array([20, 150, 255], dtype=np.uint8)

# ------------------------------------------------

def point_to_rect_distance(px, py, x1, y1, x2, y2):
    """
    Shortest distance from a point (px, py) to axis-aligned rectangle
    with corners (x1, y1) (top-left) and (x2, y2) (bottom-right)
    """
    if x1 <= px <= x2 and y1 <= py <= y2:
        return 0.0

    dx = max(x1 - px, 0, px - x2)
    dy = max(y1 - py, 0, py - y2)
    return math.sqrt(dx * dx + dy * dy)

def get_hand_point(mask):
    """
    Find largest contour in the mask and return:
    - centroid (cx, cy)
    - topmost point (tx, ty) (approx fingertip when hand is upright)
    Returns None if no hand found.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    hand_contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(hand_contour) < 1000:
        return None

    M = cv2.moments(hand_contour)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    topmost = tuple(hand_contour[hand_contour[:, :, 1].argmin()][0])
    tx, ty = topmost

    return {
        "contour": hand_contour,
        "centroid": (cx, cy),
        "top": (tx, ty)
    }

def classify_state(distance):
    """
    Given distance in pixels, return state: SAFE, WARNING, DANGER
    """
    if distance > SAFE_THRESHOLD:
        return "SAFE"
    elif distance > WARNING_THRESHOLD:
        return "WARNING"
    else:
        return "DANGER"

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Failed to open camera.")
        return

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rect_x1 = w // 2 - RECT_WIDTH // 2
        rect_y1 = h // 2 - RECT_HEIGHT // 2
        rect_x2 = rect_x1 + RECT_WIDTH
        rect_y2 = rect_y1 + RECT_HEIGHT

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(hsv, LOWER_SKIN, UPPER_SKIN)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        hand_info = get_hand_point(skin_mask)

        state = "NO HAND"
        distance = None

        if hand_info is not None:
            contour = hand_info["contour"]
            centroid = hand_info["centroid"]
            top_point = hand_info["top"]

            px, py = top_point
            distance = point_to_rect_distance(px, py, rect_x1, rect_y1, rect_x2, rect_y2)
            state = classify_state(distance)

            cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)
            cv2.circle(frame, centroid, 5, (0, 255, 255), -1)
            cv2.circle(frame, top_point, 7, (0, 0, 255), -1)
            cv2.putText(frame, "Hand", (centroid[0] + 10, centroid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        if state == "SAFE":
            rect_color = (0, 255, 0)
        elif state == "WARNING":
            rect_color = (0, 255, 255)
        elif state == "DANGER":
            rect_color = (0, 0, 255)
        else:
            rect_color = (200, 200, 200)

        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), rect_color, 3)

        if state == "SAFE":
            text_color = (0, 255, 0)
        elif state == "WARNING":
            text_color = (0, 255, 255)
        elif state == "DANGER":
            text_color = (0, 0, 255)
        else:
            text_color = (255, 255, 255)

        cv2.putText(frame, f"STATE: {state}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)

        if distance is not None:
            cv2.putText(frame, f"Distance: {int(distance)} px", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

        if state == "DANGER":
            cv2.putText(frame, "DANGER DANGER", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time + 1e-8)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 160, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Hand Danger Detection - Arvyax POC", frame)
        cv2.imshow("Skin Mask", skin_mask)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
