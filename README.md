# color_classifier

# Нейросеть для классификации изображений

Этот проект представляет собой нейросеть, которая классифицирует изображения как черно-белые или цветные.

## Возможности
- Предобработка и нормализация изображений
- Использование сверточной нейронной сети (CNN) для классификации
- Аугментация данных для улучшения обобщающей способности модели
- Раннее прекращение обучения для предотвращения переобучения
- Возможность дообучения модели с использованием предварительно сохраненных весов
- Классификация и сортировка изображений по соответствующим папкам

## Установка
1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/yourusername/image-classifier.git
   cd image-classifier
   ```
2. Установите зависимости:
   ```bash
   pip install -r requirements.txt
   ```

## Предобработка данных
Изображения хранятся в директории `data/` со следующей структурой:
```
data/
  ├── black_white/
  │   ├── img1.jpg
  │   ├── img2.jpg
  │   └── ...
  ├── color/
  │   ├── img1.jpg
  │   ├── img2.jpg
  │   └── ...
```
Загрузка и предобработка датасета:
```python
X, y = load_data()
X = X / 255.0  # Нормализация значений пикселей
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Архитектура модели
Модель состоит из нескольких сверточных слоев, за которыми следуют пулинг, dropout и полносвязные слои:
```python
model = Sequential([
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.5),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

## Дообучение модели с сохраненными весами
Если необходимо, можно дообучить модель, используя ранее сохраненные веса:
```python
model.save_weights("model_weights.weights.h5")
model.load_weights("model_weights.weights.h5")

history_finetune = model.fit(
    train_generator,  
    steps_per_epoch=len(X_train) // 32,  
    validation_data=validation_generator,  
    epochs=20,  
    callbacks=[early_stopping]
)
```

## Визуализация процесса обучения
Для анализа процесса обучения можно построить график точности на обучающем и валидационном наборах:
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Точность на обучении', color='blue', linestyle='-', marker='o')
plt.plot(history.history['val_accuracy'], label='Точность на валидации', color='orange', linestyle='--', marker='x')
plt.title('Точность модели на обучении и валидации', fontsize=16)
plt.xlabel('Эпоха', fontsize=12)
plt.ylabel('Точность', fontsize=12)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
```

## Классификация и сортировка новых изображений
После обучения модель можно использовать для классификации новых изображений и их сортировки в соответствующие папки:
```python
classify_and_copy_images("unclassified_images", "classified_images", model)
```

