from djitellopy import Tello
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json

# === Загрузка модели и меток классов ===
model = load_model("emotion_classification_model.h5")

with open("class_indices.json") as f:
    class_indices = json.load(f)

emotion_labels = [label for label, _ in sorted(class_indices.items(), key=lambda x: x[1])]

# === Инициализация каскада ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# === Tello подключение ===
tello = Tello()
tello.connect()
tello.streamon()
print(f"Батарея: {tello.get_battery()}%")

frame_reader = tello.get_frame_read()
frame_count = 0
emotion = "..."

try:
    while True:
        frame = frame_reader.frame
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            try:
                face_resized = cv2.resize(face_img, (224, 224))
                face_normalized = face_resized / 255.0
                face_input = np.expand_dims(face_normalized, axis=0)

                if frame_count % 5 == 0:
                    pred = model.predict(face_input, verbose=0)
                    print("Prediction shape:", pred.shape)
                    if pred.shape[1] == len(emotion_labels):
                        emotion = emotion_labels[np.argmax(pred)]
                    else:
                        emotion = "unknown"

                frame_count += 1

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (255, 255, 255), 2)

            except Exception as e:
                print(f"[!] Ошибка обработки лица: {e}")

        cv2.imshow("Tello Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    tello.streamoff()
    cv2.destroyAllWindows()
    print("Отключено.")
