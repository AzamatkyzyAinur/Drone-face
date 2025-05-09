import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from PIL import Image
import os

dataset_path = "dataset"  # Путь к папке с изображениями

def check_images(directory):
    """Удаляет поврежденные изображения и пропускает папки."""
    for folder in os.listdir(directory):  
        folder_path = os.path.join(directory, folder)
        if not os.path.isdir(folder_path):  # Пропускаем файлы, если это не папка
            continue
        
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            
            if not os.path.isfile(file_path):  # Пропускаем папки
                continue

            try:
                with Image.open(file_path) as img:
                    img.verify()  # Проверка изображения
            except Exception as e:
                print(f"[!] Удаляю поврежденное изображение: {file_path}, Ошибка: {e}")
                os.remove(file_path)  # Удаляем только файлы

# ✅ Запускаем проверку перед загрузкой датасета
print("🔍 Проверяем изображения...")
check_images(dataset_path)
print("✅ Проверка завершена!\n")



# Аугментация данных
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

test_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Загружаем данные
train_generator = train_datagen.flow_from_directory(
    'dataset/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = test_datagen.flow_from_directory(
    'dataset/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Создание модели
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2,2)),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(pool_size=(2,2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Убираем 50% случайных нейронов
    Dense(train_generator.num_classes, activation='softmax')
])

# Компиляция модели
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Сохранение лучшей модели
checkpoint = ModelCheckpoint(
    "best_model.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

# Обучение модели
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[checkpoint]
)

# Сохранение итоговой модели
model.save("emotion_classification_model.h5")
print("✅ Модель сохранена как 'emotion_classification_model.h5'")
