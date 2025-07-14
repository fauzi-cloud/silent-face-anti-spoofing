import cv2
import torch
import random
import time
import numpy as np
from torchvision import transforms, models
from PIL import Image
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# LOAD MobileNetV3
model = models.mobilenet_v3_large(weights=None)
num_features = model.classifier[3].in_features
model.classifier[3] = torch.nn.Linear(num_features, 2)
model = model.to(device)

# LOAD HASIL TRAIN
model.load_state_dict(
    torch.load("results/anti_spoof_model.pth", map_location=device)
)
model.eval()

# TRANSFORM
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# FACE DETECTION ( INI ALGORITMA DETEKSI OBJECT DARI OPENCV )
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

WAIT_REAL_SECONDS = 2
wait_real_start = None
accumulating_real = False

verification_in_progress = False

print("Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if not verification_in_progress:
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
                if not accumulating_real:
                    wait_real_start = time.time()
                    accumulating_real = True
                    print("REAL detected. Starting 2s accumulation…")
                else:
                    elapsed = time.time() - wait_real_start
                    cv2.putText(frame, f"Keep steady… {int(WAIT_REAL_SECONDS - elapsed)}s",
                                (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    if elapsed >= WAIT_REAL_SECONDS:
                        verification_in_progress = True
                        accumulating_real = False
                        command = random.choice(commands)
                        command_time = time.time()
                        challenge_issued = True
                        face_bbox = (x, y, w, h)
                        prev_face_x = None
                        print(f"✅ Passed continuous REAL check. Issuing challenge: {command}")
                        break

            else:
                if accumulating_real:
                    print("❌ FAKE detected during REAL accumulation. Resetting.")
                    cv2.putText(frame, "FAKE detected - restarting!", (30, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    accumulating_real = False

    else:
        # VERIVICATION FLOW
        (x, y, w, h) = face_bbox
        face_center_x = x + w // 2

        if challenge_issued:
            current_time = time.time()
            if current_time - command_time < 5:
                cv2.putText(frame, command, (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

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
