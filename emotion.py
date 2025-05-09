import os
import requests
from duckduckgo_search import DDGS

dataset_dir = "dataset"
emotions = ["sad", "neutral", "happy", "angry"]

# Функция для скачивания изображений
def download_images(emotion, num_images=100):
    emotion_dir = os.path.join(dataset_dir, emotion)
    os.makedirs(emotion_dir, exist_ok=True)

    with DDGS() as ddgs:
        images = list(ddgs.images(f"{emotion} face", max_results=num_images))

        for i, img in enumerate(images):
            try:
                response = requests.get(img["image"], timeout=5)
                file_path = os.path.join(emotion_dir, f"{i+1}.jpg")
                with open(file_path, "wb") as f:
                    f.write(response.content)
            except Exception as e:
                print(f"Ошибка при скачивании {img['image']}: {e}")

# Запуск скачивания
for emotion in emotions:
    print(f"Скачиваем {emotion} изображения...")
    download_images(emotion)

print("Скачивание завершено!")
