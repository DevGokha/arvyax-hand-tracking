# Arvyax Hand Danger Detection (POC)

This is a prototype for Arvyax that uses a webcam feed to track the user's hand in real time and trigger a **DANGER DANGER** warning when the hand approaches a virtual object.

## Tech

- Python
- OpenCV
- NumPy
- Classical computer vision (no MediaPipe / no OpenPose / no cloud APIs)

## Approach

1. Capture frames from the webcam.
2. Convert to HSV and apply a **skin-color threshold** to create a binary mask.
3. Use morphological operations and contours to find the **largest skin region** (hand).
4. Compute the **topmost point** of the hand contour (approx fingertip).
5. Draw a **virtual rectangle** in the center of the screen.
6. Compute the **distance from fingertip to rectangle**.
7. Classify state based on distance:
   - SAFE – far from rectangle
   - WARNING – approaching rectangle
   - DANGER – very close or touching the rectangle
8. Overlay:
   - Current state text
   - "DANGER DANGER" in red during danger state
   - FPS counter

## How to Run

```bash
python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

python hand_danger_detection.py
"# arvyax-hand-tracking" 
