import cv2
import os

# CHANGE THIS to "fake" when collecting fake samples
label = "real"

save_dir = os.path.join("data", label)
os.makedirs(save_dir, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
count = 0

print("Press S to save face image. Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1)
    if key == ord("s"):
        if len(faces) == 0:
            print("No face detected. Not saving.")
            continue
        x, y, w, h = faces[0]
        face_img = frame[y:y+h, x:x+w]
        filepath = os.path.join(save_dir, f"{label}_{count}.jpg")
        cv2.imwrite(filepath, face_img)
        print(f"✅ Saved {filepath}")
        count += 1

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
