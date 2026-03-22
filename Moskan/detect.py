import cv2
import torch
import re
import easyocr
from ultralytics import YOLO

MODEL_PATH = "C:/Users/YourName/plate_project/best.pt"  # update this

model  = YOLO(MODEL_PATH)
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

def preprocess_plate(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3,
                      interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def clean_text(raw):
    return re.sub(r'[^A-Z0-9]', '', raw.upper())

# Open webcam — 0 = built-in camera, 1 = external USB camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Webcam opened — press Q to quit")

frame_count = 0
last_plates = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read from webcam")
        break

    # Run OCR every 3rd frame — keeps it smooth
    # On skipped frames, just redraw last known boxes
    if frame_count % 3 == 0:
        results = model(frame, verbose=False)[0]
        last_plates = []

        for box in results.boxes:
            conf = box.conf.item()
            if conf < 0.5:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            pad = 5
            crop = frame[max(0,y1-pad):y2+pad,
                         max(0,x1-pad):x2+pad]

            if crop.size == 0:
                continue

            binary = preprocess_plate(crop)
            ocr_out = reader.readtext(binary)
            text = clean_text(' '.join([r[1] for r in ocr_out]))

            last_plates.append({
                'text': text,
                'conf': conf,
                'box': (x1, y1, x2, y2)
            })

    # Always draw the latest known results
    display = frame.copy()
    for p in last_plates:
        x1, y1, x2, y2 = p['box']
        cv2.rectangle(display, (x1,y1), (x2,y2), (0,220,0), 2)
        label = f"{p['text']}  {p['conf']:.2f}"
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_y = max(y1 - 10, 20)
        cv2.rectangle(display,
                      (x1, label_y - th - 6),
                      (x1 + tw + 4, label_y + 2),
                      (0, 0, 0), -1)
        cv2.putText(display, label, (x1+2, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,0), 2)

    # Show FPS in corner
    fps_text = f"Frame {frame_count}"
    cv2.putText(display, fps_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Number Plate Detection — press Q to quit", display)

    # Press Q to exit cleanly
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
print("Closed")