import cv2
import torch
import random
import time
from torchvision import transforms
import numpy as np
from PIL import Image
from models.model import AntiSpoofNet

# 1. Choose device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load model
model = AntiSpoofNet()
model.load_state_dict(torch.load("result/anti_spoof_model.pth", map_location=device))
model.to(device)
model.eval()

# 3. Define transform (no need to resize twice)
transform = transforms.Compose([
    transforms.ToTensor()
])

# 4. Load Haar cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# 5. Open webcam
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

        # Draw rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop face
        face_img = frame[y:y + h, x:x + w]

        # Convert to RGB
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # Resize to 224x224
        face_resized = cv2.resize(face_rgb, (224, 224))

        # Convert to PIL
        face_pil = Image.fromarray(face_resized)

        # Transform to tensor
        face_tensor = transform(face_pil).unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            pred = model(face_tensor)
            label = torch.argmax(pred, 1).item()

        # Label text
        label_text = "REAL" if label == 1 else "FAKE"
        color = (0, 255, 0) if label == 1 else (0, 0, 255)

        # Draw label
        cv2.putText(frame, label_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    else:
        challenge_issued = False

    # Show frame
    cv2.imshow("Anti-Spoofing", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
