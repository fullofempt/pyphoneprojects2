import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam

# Проверяем доступность GPU и настраиваем TensorFlow
def setup_gpu():
    """Настройка GPU для TensorFlow"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Разрешаем рост памяти GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ Используется GPU: {gpus}")
            return True
        except RuntimeError as e:
            print(f"❌ Ошибка настройки GPU: {e}")
            return False
    else:
        print("⚠️ GPU не обнаружено, используется CPU")
        return False

# Настраиваем GPU при импорте
GPU_AVAILABLE = setup_gpu()

class CatDogClassifier:
    def __init__(self):
        self.model = None
        self.history = None
        self.class_names = ['cat', 'dog']
        self.img_size = (150, 150)
        self.use_generator = True  # Используем генераторы для экономии памяти
        
    def build_improved_model(self):
        """Улучшенная архитектура нейросети"""
        self.model = Sequential([
            # Первый сверточный блок
            Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Второй сверточный блок
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Третий сверточный блок
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Четвертый сверточный блок
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            Dropout(0.25),
            
            # Полносвязные слои
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(1, activation='sigmoid')
        ])
        
        # Оптимизатор с learning rate
        optimizer = Adam(learning_rate=0.001)
        
        self.model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return self.model
    
    def check_data_balance(self, data_dir):
        """Проверка баланса данных"""
        cat_dir = os.path.join(data_dir, 'cat')
        dog_dir = os.path.join(data_dir, 'dog')
        
        cat_count = len([f for f in os.listdir(cat_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]) if os.path.exists(cat_dir) else 0
        dog_count = len([f for f in os.listdir(dog_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]) if os.path.exists(dog_dir) else 0
        
        print(f"🐱 Изображений кошек: {cat_count}")
        print(f"🐶 Изображений собак: {dog_count}")
        
        if cat_count == 0 or dog_count == 0:
            raise ValueError("Не найдено изображений в одной из папок")
        
        # Балансировка весов классов
        total = cat_count + dog_count
        weight_for_cat = total / (2 * cat_count)
        weight_for_dog = total / (2 * dog_count)
        
        class_weights = {0: weight_for_cat, 1: weight_for_dog}
        print(f"⚖️ Веса классов: {class_weights}")
        
        return class_weights, cat_count, dog_count
    
    def create_data_generators(self, data_dir, batch_size=32):
        """Создание генераторов данных для экономии памяти"""
        # Аугментация для тренировочных данных
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            validation_split=0.2
        )
        
        # Только нормализация для валидационных данных
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='binary',
            subset='training',
            shuffle=True
        )
        
        validation_generator = val_datagen.flow_from_directory(
            data_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='binary',
            subset='validation',
            shuffle=False
        )
        
        return train_generator, validation_generator
    
    def train_with_generator(self, data_dir, epochs=50, batch_size=32):
        """Обучение с использованием генераторов (экономно по памяти)"""
        # Проверяем баланс данных
        class_weights, cat_count, dog_count = self.check_data_balance(data_dir)
        
        # Создаем генераторы
        train_generator, validation_generator = self.create_data_generators(data_dir, batch_size)
        
        # Рассчитываем steps
        train_steps = train_generator.samples // batch_size
        val_steps = validation_generator.samples // batch_size
        
        print(f"📊 Train steps: {train_steps}, Validation steps: {val_steps}")
        
        # Улучшенные колбэки
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7, verbose=1),
            ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
        ]
        
        # Построение улучшенной модели
        self.build_improved_model()
        
        # Обучение с генераторами
        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=train_steps,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=val_steps,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1
        )
        
        # Оценка модели
        self.evaluate_with_generator(validation_generator)
        
        return self.history
    
    def evaluate_with_generator(self, validation_generator):
        """Оценка модели с использованием генератора"""
        # Оценка точности
        val_loss, val_accuracy, val_precision, val_recall = self.model.evaluate(
            validation_generator, verbose=0
        )
        
        print("=" * 50)
        print("📊 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ:")
        print(f"Точность (Accuracy): {val_accuracy:.4f}")
        print(f"Precision: {val_precision:.4f}")
        print(f"Recall: {val_recall:.4f}")
        print(f"Потери (Loss): {val_loss:.4f}")
        
        # Для confusion matrix нужно собрать предсказания
        self.generate_confusion_matrix(validation_generator)
    
    def generate_confusion_matrix(self, validation_generator):
        """Генерация confusion matrix по частям"""
        print("\n📋 Генерация Confusion Matrix...")
        
        y_true = []
        y_pred = []
        
        # Сбрасываем генератор
        validation_generator.reset()
        
        # Обрабатываем батчами для экономии памяти
        for i in range(validation_generator.samples // validation_generator.batch_size + 1):
            try:
                X_batch, y_batch = validation_generator.next()
                preds = self.model.predict(X_batch, verbose=0)
                y_true.extend(y_batch)
                y_pred.extend((preds > 0.5).astype(int).flatten())
            except StopIteration:
                break
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        print("📊 Confusion Matrix:")
        print(cm)
        
        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_true, y_pred, target_names=['Cats', 'Dogs']))
        
        # Визуализация confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Cats', 'Dogs'], 
                    yticklabels=['Cats', 'Dogs'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
    def train(self, data_dir, epochs=50, batch_size=32, validation_split=0.2):
        """Основной метод обучения (использует генераторы)"""
        return self.train_with_generator(data_dir, epochs, batch_size)
    
    def predict_image(self, image_path):
        """Предсказание для одного изображения"""
        if self.model is None:
            raise ValueError("Модель не обучена")
        
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.img_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = self.model.predict(img_array, verbose=0)[0][0]
        class_idx = 1 if prediction > 0.5 else 0
        confidence = prediction if class_idx == 1 else 1 - prediction
        
        return self.class_names[class_idx], confidence
    
    def save_model(self, filepath):
        """Сохранение модели"""
        if self.model:
            self.model.save(filepath)
    
    def load_model(self, filepath):
        """Загрузка модели"""
        self.model = load_model(filepath)
    
    def plot_training_history(self):
        """Визуализация процесса обучения"""
        if self.history is None:
            raise ValueError("История обучения отсутствует")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        # Precision
        if 'precision' in self.history.history:
            ax3.plot(self.history.history['precision'], label='Training Precision')
            ax3.plot(self.history.history['val_precision'], label='Validation Precision')
            ax3.set_title('Model Precision')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Precision')
            ax3.legend()
            ax3.grid(True)
        
        # Recall
        if 'recall' in self.history.history:
            ax4.plot(self.history.history['recall'], label='Training Recall')
            ax4.plot(self.history.history['val_recall'], label='Validation Recall')
            ax4.set_title('Model Recall')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Recall')
            ax4.legend()
            ax4.grid(True)
        
        plt.tight_layout()
        plt.show()

# Класс GUI остается без изменений (как в предыдущем сообщении)
# [Здесь должен быть полный код класса GUI]

def main():
    """Основная функция"""
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()