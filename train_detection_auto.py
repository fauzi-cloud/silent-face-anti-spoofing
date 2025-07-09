import cv2
import torch
import random
import time
import numpy as np
from torchvision import transforms
from PIL import Image
import sys
from models.model import AntiSpoofNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AntiSpoofNet().to(device)
model.load_state_dict(torch.load("results/anti_spoof_model.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

commands = ["Turn LEFT", "Turn RIGHT"]
prev_face_x = None
command = None
command_time = 0
challenge_issued = False
challenge_passed = False

waiting_period = False
wait_start_time = None
WAIT_SECONDS = 3  # Delay before issuing challenge

verification_in_progress = False

print("Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if not verification_in_progress:
        # Only run prediction if not already in verification flow
        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]
            face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
            face_tensor = transform(face_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                pred = model(face_tensor)
                pred_label = pred.argmax().item()

            label_text = "REAL" if pred_label == 1 else "FAKE"
            color = (0, 255, 0) if pred_label == 1 else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label_text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            if pred_label == 1:
                # Lock into verification flow
                verification_in_progress = True
                waiting_period = True
                wait_start_time = time.time()
                face_bbox = (x, y, w, h)
                print("Face is REAL. Locking for verification...")
            break  # Only handle first detected face this frame

    else:
        # We are in verification flow
        (x, y, w, h) = face_bbox

        face_center_x = x + w // 2

        if waiting_period:
            elapsed = time.time() - wait_start_time
            if elapsed >= WAIT_SECONDS:
                # Time to issue challenge
                command = random.choice(commands)
                command_time = time.time()
                challenge_issued = True
                waiting_period = False
                prev_face_x = None
                print(f"Challenge: {command}")
            else:
                remaining = int(WAIT_SECONDS - elapsed)
                cv2.putText(frame, f"Sabarrr cuk... ({remaining}s)",
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 255, 255), 2)

        elif challenge_issued:
            current_time = time.time()
            if current_time - command_time < 5:
                cv2.putText(frame, command, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

                # Check head movement
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                if len(faces) > 0:
                    x_new, y_new, w_new, h_new = faces[0]
                    new_center_x = x_new + w_new // 2

                    if prev_face_x is not None:
                        delta_x = new_center_x - prev_face_x

                        if command == "Turn LEFT" and delta_x < -15:
                            challenge_passed = True
                        elif command == "Turn RIGHT" and delta_x > 15:
                            challenge_passed = True

                    prev_face_x = new_center_x
            else:
                # Challenge timeout → reset
                print("Challenge timeout. Resetting.")
                verification_in_progress = False
                challenge_issued = False
                prev_face_x = None

    if challenge_passed:
        print("✅ Identity verified. Exiting app.")
        cv2.destroyAllWindows()
        cap.release()
        try:
            from plyer import notification
            notification.notify(
                title='Face Anti-Spoofing',
                message='Identity successfully verified!',
                timeout=5
            )
        except ImportError:
            print("Install plyer for system notifications (pip install plyer).")
        sys.exit(0)

    cv2.imshow("Live Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
