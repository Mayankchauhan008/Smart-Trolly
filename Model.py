import cv2
from ultralytics import YOLO
import pygame  # ✅ for sound

# Initialize sound system
pygame.mixer.init()
beep_sound = pygame.mixer.Sound("beep.mp3")  # Make sure this file exists and is a valid mp3/wav

# Load the YOLO model
model = YOLO(r"D:\5 sem\Smart Trolly\final.pt")

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

detected_labels = set()

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to grab frame")
        break

    results = model(frame, conf=0.9, verbose=False)
    found_object = False

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            found_object = True

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            print(class_name)

            if class_name not in detected_labels:
                detected_labels.add(class_name)
                beep_sound.play()  # ✅ Sound played using pygame

    if not found_object:
        print("No object detected")
        cv2.putText(frame, "No object detected", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Smart Trolley Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()