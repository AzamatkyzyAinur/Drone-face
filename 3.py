from djitellopy import Tello 
import cv2
import numpy as np
from deepface import DeepFace
import time

# === Инициализация каскада ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# === Tello подключение ===
tello = Tello()
tello.connect()
tello.streamon()
print(f"Battery: {tello.get_battery()}%")

frame_reader = tello.get_frame_read()
tello.send_rc_control(0, 0, 0, 0)

# === Safe takeoff ===
print("Taking off...")
tello.takeoff()
time.sleep(2)

# === Climb to desired height (e.g. 80 cm) ===
current_height = tello.get_height()
print(f"Current height: {current_height} cm")
target_height = 80
delta_height = target_height - current_height

if delta_height > 20:
    try:
        tello.move_up(delta_height)
        time.sleep(2)
    except Exception as e:
        print(f"[!] Move up error: {e}")
else:
    print("Already close to target height.")

frame_center_x = 640 // 2
rotation_speed = 15

try:
    while True:
        frame = frame_reader.frame
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            face_img = frame[y:y+h, x:x+w]
            face_center_x = x + w // 2
            delta_x = face_center_x - frame_center_x

            if abs(delta_x) > 50:
                yaw_velocity = rotation_speed if delta_x > 0 else -rotation_speed
                tello.send_rc_control(0, 0, 0, yaw_velocity)
            else:
                tello.send_rc_control(0, 0, 0, 0)

            try:
                analysis = DeepFace.analyze(face_img, actions=['gender', 'emotion'], enforce_detection=False)
                gender = analysis[0]['dominant_gender']
                emotion = analysis[0]['dominant_emotion']
                label = f"Gender: {gender}, Emotion: {emotion}"

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            except Exception as e:
                print(f"[!] Face analysis error: {e}")
        else:
            tello.send_rc_control(0, 0, 0, 0)

        cv2.imshow("Tello Face Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(1)
    try:
        tello.streamoff()
    except Exception as e:
        print(f"[!] Streamoff error: {e}")
    tello.land()
    cv2.destroyAllWindows()
    print("Shutdown complete.")
