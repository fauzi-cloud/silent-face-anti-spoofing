import cv2
import torch
import random
import time
from torchvision import transforms, models
import numpy as np
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# LOAD MobileNetV3
model = models.mobilenet_v3_large(weights=None)
num_features = model.classifier[3].in_features
model.classifier[3] = torch.nn.Linear(num_features, 2)

# LOAD HASIL TRAIN
model.load_state_dict(
    torch.load("results/anti_spoof_model.pth", map_location=device)
)
model = model.to(device)
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

print("Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]

        # DRAW RECTANGLE
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # CROP FACE
        face_img = frame[y:y + h, x:x + w]

        # CONVERT TO RGB
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # RESIZE 224x224
        face_resized = cv2.resize(face_rgb, (224, 224))

        # CONVERT TO PIL
        face_pil = Image.fromarray(face_resized)

        # TRANSFORM TO TENSOR
        face_tensor = transform(face_pil).unsqueeze(0).to(device)

        # PREDICT
        with torch.no_grad():
            pred = model(face_tensor)
            label = torch.argmax(pred, 1).item()

        # LABEL TEXT
        label_text = "REAL" if label == 1 else "FAKE"
        color = (0, 255, 0) if label == 1 else (0, 0, 255)

        # DRAW LABEL
        cv2.putText(frame, label_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    else:
        challenge_issued = False

    # SHOW FRAME
    cv2.imshow("Anti-Spoofing", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
