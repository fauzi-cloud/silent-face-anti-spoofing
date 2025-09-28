import cv2
import torch
import random
import numpy as np
from torchvision import transforms, models
from PIL import Image
import mediapipe as mp


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


model = models.mobilenet_v3_large(weights=None)
num_features = model.classifier[3].in_features
model.classifier[3] = torch.nn.Linear(num_features, 2)

model.load_state_dict(torch.load("results/anti_spoof_model.pth", map_location=device))
model = model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.6, min_tracking_confidence=0.6)


cap = cv2.VideoCapture(0)


circle_center = (320, 240)  
circle_radius = 140         


progress = 0
progress_target = 100
challenge_done = False
command = None


real_locked = False
real_counter = 0
real_threshold = 10  

print("Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    
    cv2.circle(frame, circle_center, circle_radius, (255, 255, 255), 2)
    cv2.putText(frame, "Posisikan wajah di tengah lingkaran",
                (120, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            
            xs = [lm.x for lm in face_landmarks.landmark]
            ys = [lm.y for lm in face_landmarks.landmark]
            x_min = int(min(xs) * w)
            x_max = int(max(xs) * w)
            y_min = int(min(ys) * h)
            y_max = int(max(ys) * h)

            
            margin = 30
            x_min = max(0, x_min - margin)
            x_max = min(w, x_max + margin)
            y_min = max(0, y_min - margin)
            y_max = min(h, y_max + margin)

            
            face_crop = frame[y_min:y_max, x_min:x_max]
            if face_crop.size == 0:
                continue

            
            face_resized = cv2.resize(face_crop, (224, 224))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_pil = Image.fromarray(face_rgb)
            face_tensor = transform(face_pil).unsqueeze(0).to(device)

            
            with torch.no_grad():
                pred = model(face_tensor)
                label = torch.argmax(pred, 1).item()

            
            if real_locked:
                label = 1
            else:
                if label == 1:
                    real_counter += 1
                    if real_counter >= real_threshold:
                        real_locked = True
                        print("REAL locked ✅")
                else:
                    real_counter = 0

            
            label_text = "REAL" if label == 1 else "FAKE"
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.putText(frame, label_text, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            
            if label == 1:
                nose = face_landmarks.landmark[1]
                nx, ny = int(nose.x * w), int(nose.y * h)
                dist = np.sqrt((nx - circle_center[0])**2 + (ny - circle_center[1])**2)

                if dist < circle_radius:
                    
                    if command is None:
                        command = random.choice(["LEFT", "RIGHT"])
                        print("Challenge:", command)

                    cv2.putText(frame, f"Turn {command}", (200, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

                    
                    if command == "LEFT" and nx > circle_center[0] + 40:  
                        progress += 5
                    elif command == "RIGHT" and nx < circle_center[0] - 40:  
                        progress += 5
                    else:
                        progress = max(0, progress - 2)

                    
                    angle = int(360 * (progress / progress_target))
                    cv2.ellipse(frame, circle_center, (circle_radius, circle_radius),
                                0, -90, -90 + angle, (0, 255, 0), 8)

                    if progress >= progress_target:
                        print("Challenge success ✅")
                        cap.release()
                        cv2.destroyAllWindows()
                        exit(0)   

    cv2.imshow("Anti-Spoofing Challenge", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
