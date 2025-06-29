import numpy as np
import cv2 as cv
import argparse

# Set up command line arguments
parser = argparse.ArgumentParser(description='Lucas-Kanade Optical Flow with webcam or video file')
parser.add_argument('--video', type=str, help='Path to video file (default: use webcam)')
args = parser.parse_args()

# Use webcam or video file
if args.video:
    cap = cv.VideoCapture(args.video)
    print(f"Using video file: {args.video}")
else:
    cap = cv.VideoCapture(0)
    print("Using webcam")

if not cap.isOpened():
    if args.video:
        print(f"Error: Could not open video file: {args.video}")
    else:
        print("Error: Could not open webcam")
    exit()

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
if not ret:
    print("Error: Could not read from webcam")
    exit()

old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

print("Press ESC to quit, 'r' to reset tracking points")

while(1):
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
        
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # calculate optical flow
    if p0 is not None and len(p0) > 0:
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
            
            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
            
            # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
    
    img = cv.add(frame, mask)
    cv.imshow('Optical Flow', img)
    
    k = cv.waitKey(30) & 0xff
    if k == 27:  # ESC key
        break
    elif k == ord('r'):  # Reset tracking points
        mask = np.zeros_like(old_frame)
        p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

cap.release()
cv.destroyAllWindows()