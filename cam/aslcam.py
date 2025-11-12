import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector
import os 
from datetime import datetime 


CAM_INDEX = 0               # MacBook cam
USE_HAND_DETECTION = True   
SAVE_EVENTS = False          # write events to events.txt
SHOW_WINDOWS = True         
EVENT_THRESHOLD = 15        # sensitivity for events
CROP_PADDING = 30           # extra pixels around hand box
TARGET_SIZE = (224, 224)    # good size for ML models later


    # (NEUROMORPHIC-LIKE) DETECTION

def find_events(cur_gray, prev_gray, threshold=15):
    
        # difference image
    diff = cur_gray.astype(np.int16) - prev_gray.astype(np.int16)

        # mask where |diff| > threshold
    pos_mask = diff > threshold
    neg_mask = diff < -threshold

    events = []
        # Build an RGB mask to visualize
    event_mask = np.zeros((cur_gray.shape[0], cur_gray.shape[1], 3), dtype=np.uint8)

        # positive events = green, negative = magenta
    event_mask[pos_mask] = (0, 255, 0)     # green
    event_mask[neg_mask] = (255, 0, 255)   # magenta

        # Turn the event locations into a list of points
    ys_pos, xs_pos = np.where(pos_mask)
    for x, y in zip(xs_pos, ys_pos):
        events.append({"x": int(x), "y": int(y), "type": 1})

    ys_neg, xs_neg = np.where(neg_mask)
    for x, y in zip(xs_neg, ys_neg):
        events.append({"x": int(x), "y": int(y), "type": -1})

    return events, event_mask

def save_labeled_frame(img, label, base_dir="../data"):
    dir_path = os.path.join(base_dir, label)
    os.makedirs(dir_path, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(dir_path, f"{label}_{ts}.jpg")
    cv2.imwrite(filename, img)
    print(f"[saved] {filename}")


    # main loop

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("Could not open camera.")
        return

        # Hand detector 
    detector = HandDetector(maxHands=1) if USE_HAND_DETECTION else None

    prev_gray = None
    start_ticks = cv2.getTickCount()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = frame.copy()

            #  detect hand and crop
        if detector is not None:
            hands, img_draw = detector.findHands(img) 
            if hands:
                hand = hands[0]
                x, y, w, h = hand['bbox']

                    # padding
                x1 = max(x - CROP_PADDING, 0)
                y1 = max(y - CROP_PADDING, 0)
                x2 = min(x + w + CROP_PADDING, img.shape[1])
                y2 = min(y + h + CROP_PADDING, img.shape[0])

                crop = img[y1:y2, x1:x2]
            else:
                crop = img
                img_draw = img
        else:
            crop = img
            img_draw = img

            # resize crop to target size (helps with ML later)
        crop_resized = cv2.resize(crop, TARGET_SIZE)

            #  convert to grayscale for event detection
        gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)

        event_mask = np.zeros_like(crop_resized)
        events = []

        if prev_gray is not None:
            events, event_mask = find_events(gray, prev_gray, threshold=EVENT_THRESHOLD)

            # overlay: blend crop + event_mask so you can see where motion happened
        overlay = cv2.addWeighted(crop_resized, 0.7, event_mask, 0.3, 0)
        prev_gray = gray
    
            # show windows (for debugging)
        if SHOW_WINDOWS:
            cv2.imshow("Original", img_draw)
            cv2.imshow("ASL Crop", overlay)
            
            #KEYS
        key = cv2.waitKey(1) & 0xFF

            # ~ for "A"
        if key == ord('~'):
            save_labeled_frame(crop_resized, "A")
            # 1 for "B"
        elif key == ord('1'):
            save_labeled_frame(crop_resized, "B")
            # 2 for "C"
        elif key == ord('2'):
            save_labeled_frame(crop_resized, "C")
           # 3 for "D"
        elif key == ord('3'):
            save_labeled_frame(crop_resized, "D")
            # 4 for "E"
        elif key == ord('4'):
            save_labeled_frame(crop_resized, "E")
            #5  for "F"
        elif key == ord('5'):
            save_labeled_frame(crop_resized, "F")
         # 6 for "G"
        elif key == ord('6'):
            save_labeled_frame(crop_resized, "G")
            # 7 for "H"
        elif key == ord('7'):
            save_labeled_frame(crop_resized, "H")
            # 8 for "I"
        elif key == ord('8'):
            save_labeled_frame(crop_resized, "I")
         # 9 for "J"
        elif key == ord('9'):
            save_labeled_frame(crop_resized, "J")
            # 0 for "K"
        elif key == ord('0'):
            save_labeled_frame(crop_resized, "K")
            # - for "L"
        elif key == ord('-'):
            save_labeled_frame(crop_resized, "L")
         # = for "M"
        elif key == ord('='):
            save_labeled_frame(crop_resized, "M")
            
        elif key == ord('w'):
            save_labeled_frame(crop_resized, "N")
            
        elif key == ord('e'):
            save_labeled_frame(crop_resized, "O")

        elif key == ord('r'):
            save_labeled_frame(crop_resized, "P")
           
        elif key == ord('t'):
            save_labeled_frame(crop_resized, "Q")
           
        elif key == ord('y'):
            save_labeled_frame(crop_resized, "R")
         
        elif key == ord('u'):
            save_labeled_frame(crop_resized, "S")
            
        elif key == ord('i'):
            save_labeled_frame(crop_resized, "T")
            
        elif key == ord('o'):
            save_labeled_frame(crop_resized, "U")
         
        elif key == ord('p'):
            save_labeled_frame(crop_resized, "V")
            
        elif key == ord('a'):
            save_labeled_frame(crop_resized, "W")
            
        elif key == ord('s'):
            save_labeled_frame(crop_resized, "X")

        elif key == ord('d'):
            save_labeled_frame(crop_resized, "Y")
            
        elif key == ord('f'):
            save_labeled_frame(crop_resized, "Z")
            
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
   


if __name__ == "__main__":
    main()