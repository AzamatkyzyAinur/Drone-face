import cv2
from deepface import DeepFace

# Запуск камеры
cap = cv2.VideoCapture(0)

while True:
    # Считываем кадр с камеры
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Анализируем лицо (пол + эмоции)
        analysis = DeepFace.analyze(frame, actions=['gender', 'emotion'], enforce_detection=False)
        gender = analysis[0]['dominant_gender']  # Получаем пол
        emotion = analysis[0]['dominant_emotion']  # Получаем эмоцию
        
      
        text = f"Пол: {gender}, Эмоция: {emotion}"
        
        
        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    except Exception as e:
        print("Ошибка анализа:", e)
    
    cv2.imshow("Face Recognition", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
