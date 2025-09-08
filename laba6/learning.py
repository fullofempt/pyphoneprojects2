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

class CatDogClassifier:
    def __init__(self):
        self.model = None
        self.history = None
        self.class_names = ['cat', 'dog']
        self.img_size = (150, 150)
        
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
        
        return class_weights
    
    def load_and_preprocess_data(self, data_dir):
        """Загрузка и предобработка данных"""
        images = []
        labels = []
        
        for class_name in self.class_names:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(self.img_size)
                    img_array = np.array(img) / 255.0
                    
                    images.append(img_array)
                    labels.append(1 if class_name == 'dog' else 0)
                except Exception as e:
                    print(f"Ошибка загрузки изображения {img_path}: {e}")
        
        return np.array(images), np.array(labels)
    
    def train(self, data_dir, epochs=50, batch_size=32, validation_split=0.2):
        """Обучение модели"""
        # Проверяем баланс данных
        class_weights = self.check_data_balance(data_dir)
        
        # Загрузка данных
        X, y = self.load_and_preprocess_data(data_dir)
        
        if len(X) == 0:
            raise ValueError("Не найдено изображений для обучения")
        
        # Разделение на тренировочную и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Улучшенная аугментация данных
        datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        # Улучшенные колбэки
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7, verbose=1),
            ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
        ]
        
        # Построение улучшенной модели
        self.build_improved_model()
        
        # Обучение
        self.history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            class_weight=class_weights,  # Добавляем балансировку классов
            verbose=1
        )
        
        # Детальная оценка модели
        self.detailed_evaluation(X_test, y_test)
        
        return self.history
    
    def detailed_evaluation(self, X_test, y_test):
        """Детальная оценка модели"""
        # Предсказания
        y_pred = self.model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Метрики
        test_loss, test_acc, test_precision, test_recall = self.model.evaluate(X_test, y_test, verbose=0)
        
        print("=" * 50)
        print("📊 ДЕТАЛЬНЫЕ РЕЗУЛЬТАТЫ:")
        print(f"Точность (Accuracy): {test_acc:.4f}")
        print(f"Precision: {test_precision:.4f}")
        print(f"Recall: {test_recall:.4f}")
        print(f"Потери (Loss): {test_loss:.4f}")
        
        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred_binary, target_names=['Cats', 'Dogs']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_binary)
        print("📊 Confusion Matrix:")
        print(cm)
        
        # Визуализация confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Cats', 'Dogs'], 
                    yticklabels=['Cats', 'Dogs'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
    
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

class GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Cat vs Dog Classifier")
        self.root.geometry("800x600")
        
        self.classifier = CatDogClassifier()
        self.model_trained = False
        
        self.create_widgets()
    
    def create_widgets(self):
        """Создание элементов интерфейса"""
        # Notebook для вкладок
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Вкладка обучения
        self.train_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.train_frame, text="Обучение")
        
        # Вкладка предсказания
        self.predict_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.predict_frame, text="Предсказание")
        
        # Настройка вкладки обучения
        self.setup_train_tab()
        
        # Настройка вкладки предсказания
        self.setup_predict_tab()
    
    def setup_train_tab(self):
        """Настройка вкладки обучения"""
        # Выбор директории с данными
        ttk.Label(self.train_frame, text="Директория с данными:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.data_dir_var = tk.StringVar()
        ttk.Entry(self.train_frame, textvariable=self.data_dir_var, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.train_frame, text="Обзор", command=self.browse_data_dir).grid(row=0, column=2, padx=5, pady=5)
        
        # Параметры обучения
        ttk.Label(self.train_frame, text="Количество эпох:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.epochs_var = tk.IntVar(value=50)  # Увеличено до 50
        ttk.Entry(self.train_frame, textvariable=self.epochs_var, width=10).grid(row=1, column=1, padx=5, pady=5, sticky='w')
        
        ttk.Label(self.train_frame, text="Размер батча:").grid(row=2, column=0, padx=5, pady=5, sticky='w')
        self.batch_size_var = tk.IntVar(value=32)
        ttk.Entry(self.train_frame, textvariable=self.batch_size_var, width=10).grid(row=2, column=1, padx=5, pady=5, sticky='w')
        
        # Кнопка обучения
        self.train_btn = ttk.Button(self.train_frame, text="Начать обучение", command=self.start_training)
        self.train_btn.grid(row=3, column=0, columnspan=3, pady=10)
        
        # Прогресс бар
        self.progress = ttk.Progressbar(self.train_frame, mode='indeterminate')
        self.progress.grid(row=4, column=0, columnspan=3, sticky='ew', padx=5, pady=5)
        
        # Текстовое поле для логов
        self.log_text = tk.Text(self.train_frame, height=15, width=70)
        self.log_text.grid(row=5, column=0, columnspan=3, padx=5, pady=5)
        
        # Кнопка визуализации
        self.viz_btn = ttk.Button(self.train_frame, text="Показать графики", command=self.show_graphs, state='disabled')
        self.viz_btn.grid(row=6, column=0, columnspan=3, pady=5)
        
        # Кнопка сохранения модели
        self.save_btn = ttk.Button(self.train_frame, text="Сохранить модель", command=self.save_model, state='disabled')
        self.save_btn.grid(row=7, column=0, columnspan=3, pady=5)
    
    def setup_predict_tab(self):
        """Настройка вкладки предсказания"""
        # Загрузка модели
        ttk.Label(self.predict_frame, text="Загрузить модель:").grid(row=0, column=0, padx=5, pady=5, sticky='w')
        self.model_path_var = tk.StringVar()
        ttk.Entry(self.predict_frame, textvariable=self.model_path_var, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.predict_frame, text="Обзор", command=self.browse_model).grid(row=0, column=2, padx=5, pady=5)
        ttk.Button(self.predict_frame, text="Загрузить", command=self.load_model).grid(row=0, column=3, padx=5, pady=5)
        
        # Выбор изображения для предсказания
        ttk.Label(self.predict_frame, text="Изображение для классификации:").grid(row=1, column=0, padx=5, pady=5, sticky='w')
        self.image_path_var = tk.StringVar()
        ttk.Entry(self.predict_frame, textvariable=self.image_path_var, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(self.predict_frame, text="Обзор", command=self.browse_image).grid(row=1, column=2, padx=5, pady=5)
        
        # Кнопка предсказания
        self.predict_btn = ttk.Button(self.predict_frame, text="Классифицировать", command=self.predict, state='disabled')
        self.predict_btn.grid(row=2, column=0, columnspan=4, pady=10)
        
        # Область для отображения изображения
        self.image_label = ttk.Label(self.predict_frame, text="Изображение появится здесь")
        self.image_label.grid(row=3, column=0, columnspan=4, pady=10)
        
        # Результат предсказания
        self.result_var = tk.StringVar(value="Результат: ")
        ttk.Label(self.predict_frame, textvariable=self.result_var, font=('Arial', 14)).grid(row=4, column=0, columnspan=4, pady=10)
    
    def browse_data_dir(self):
        """Выбор директории с данными"""
        directory = filedialog.askdirectory()
        if directory:
            self.data_dir_var.set(directory)
    
    def browse_model(self):
        """Выбор файла модели"""
        filepath = filedialog.askopenfilename(filetypes=[("H5 files", "*.h5"), ("All files", "*.*")])
        if filepath:
            self.model_path_var.set(filepath)
    
    def browse_image(self):
        """Выбор изображения"""
        filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")])
        if filepath:
            self.image_path_var.set(filepath)
            self.display_image(filepath)
    
    def display_image(self, filepath):
        """Отображение изображения"""
        try:
            img = Image.open(filepath)
            img.thumbnail((300, 300))
            from PIL import ImageTk
            photo = ImageTk.PhotoImage(img)
            self.image_label.configure(image=photo)
            self.image_label.image = photo  # Сохраняем ссылку
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {e}")
    
    def log_message(self, message):
        """Добавление сообщения в лог"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def start_training(self):
        """Запуск обучения"""
        data_dir = self.data_dir_var.get()
        if not data_dir or not os.path.exists(data_dir):
            messagebox.showerror("Ошибка", "Выберите корректную директорию с данными")
            return
        
        # Проверка структуры директории
        expected_dirs = ['cat', 'dog']
        for dir_name in expected_dirs:
            if not os.path.exists(os.path.join(data_dir, dir_name)):
                messagebox.showerror("Ошибка", f"В директории должны быть подпапки 'cat' и 'dog'")
                return
        
        # Запуск обучения в отдельном потоке
        self.train_btn.config(state='disabled')
        self.progress.start()
        
        def train_thread():
            try:
                self.log_message("Начало обучения...")
                epochs = self.epochs_var.get()
                batch_size = self.batch_size_var.get()
                
                self.classifier.train(data_dir, epochs=epochs, batch_size=batch_size)
                
                self.log_message("Обучение завершено!")
                if hasattr(self.classifier, 'history') and self.classifier.history:
                    self.log_message(f"Финальная точность: {self.classifier.history.history['val_accuracy'][-1]:.4f}")
                
                # Активируем кнопки после обучения
                self.viz_btn.config(state='normal')
                self.save_btn.config(state='normal')
                self.predict_btn.config(state='normal')
                self.model_trained = True
                
            except Exception as e:
                self.log_message(f"Ошибка при обучении: {e}")
            finally:
                self.progress.stop()
                self.train_btn.config(state='normal')
        
        # Запуск в отдельном потоке для избежания зависания GUI
        import threading
        thread = threading.Thread(target=train_thread)
        thread.daemon = True
        thread.start()
    
    def show_graphs(self):
        """Показать графики обучения"""
        if self.classifier.history:
            self.classifier.plot_training_history()
    
    def save_model(self):
        """Сохранение модели"""
        filepath = filedialog.asksaveasfilename(defaultextension=".h5", filetypes=[("H5 files", "*.h5")])
        if filepath:
            try:
                self.classifier.save_model(filepath)
                messagebox.showinfo("Успех", f"Модель сохранена: {filepath}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить модель: {e}")
    
    def load_model(self):
        """Загрузка модели"""
        filepath = self.model_path_var.get()
        if filepath and os.path.exists(filepath):
            try:
                self.classifier.load_model(filepath)
                messagebox.showinfo("Успех", "Модель загружена")
                self.predict_btn.config(state='normal')
                self.model_trained = True
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить модель: {e}")
    
    def predict(self):
        """Предсказание для изображения"""
        if not self.model_trained:
            messagebox.showerror("Ошибка", "Сначала обучите или загрузите модель")
            return
        
        image_path = self.image_path_var.get()
        if not image_path or not os.path.exists(image_path):
            messagebox.showerror("Ошибка", "Выберите изображение для классификации")
            return
        
        try:
            class_name, confidence = self.classifier.predict_image(image_path)
            self.result_var.set(f"Результат: {class_name} (уверенность: {confidence:.2%})")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при предсказании: {e}")

def main():
    """Основная функция"""
    root = tk.Tk()
    app = GUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()