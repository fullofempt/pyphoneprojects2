import os
import re
from PIL import Image
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

class DataPreprocessor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Data Preprocessor - Умный организатор данных")
        self.root.geometry("800x600")
        
        self.create_widgets()
    
    def create_widgets(self):
        """Создание интерфейса для препроцессинга"""
        # Заголовок
        title_label = ttk.Label(self.root, text="Умный организатор данных для нейросети", 
                               font=('Arial', 14, 'bold'))
        title_label.pack(pady=10)
        
        # Директория с данными
        frame1 = ttk.Frame(self.root)
        frame1.pack(pady=10, fill='x', padx=20)
        
        ttk.Label(frame1, text="Главная директория с данными:").pack(anchor='w')
        
        self.data_dir_var = tk.StringVar()
        entry_frame = ttk.Frame(frame1)
        entry_frame.pack(fill='x', pady=5)
        
        ttk.Entry(entry_frame, textvariable=self.data_dir_var, width=60).pack(side='left', fill='x', expand=True)
        ttk.Button(entry_frame, text="Обзор", command=self.browse_directory).pack(side='right', padx=(5, 0))
        
        # Кнопка анализа
        ttk.Button(self.root, text="🔍 Проанализировать структуру данных", 
                  command=self.analyze_structure).pack(pady=10)
        
        # Информация о структуре
        self.info_frame = ttk.LabelFrame(self.root, text="Обнаруженная структура")
        self.info_frame.pack(pady=10, fill='x', padx=20)
        
        self.info_label = ttk.Label(self.info_frame, text="Выберите директорию для анализа", wraplength=700)
        self.info_label.pack(pady=10, padx=10)
        
        # Кнопка запуска
        self.process_btn = ttk.Button(self.root, text="🚀 Автоматически добавить префиксы", 
                  command=self.add_prefixes, state='disabled', style='Accent.TButton')
        self.process_btn.pack(pady=20)
        
        # Лог
        log_frame = ttk.LabelFrame(self.root, text="Лог выполнения")
        log_frame.pack(pady=10, fill='both', expand=True, padx=20)
        
        self.log_text = tk.Text(log_frame, height=15, width=80)
        scrollbar = ttk.Scrollbar(log_frame, orient='vertical', command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        scrollbar.pack(side='right', fill='y', pady=5)
        
        # Стиль для кнопки
        style = ttk.Style()
        style.configure('Accent.TButton', foreground='white', background='#007acc')
    
    def browse_directory(self):
        """Выбор директории"""
        directory = filedialog.askdirectory()
        if directory:
            self.data_dir_var.set(directory)
            self.log_message(f"Выбрана директория: {directory}")
            self.analyze_structure()
    
    def log_message(self, message):
        """Логирование сообщений"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def analyze_structure(self):
        """Анализ структуры директории"""
        data_dir = self.data_dir_var.get()
        if not data_dir or not os.path.exists(data_dir):
            return
        
        self.log_message("🔍 Анализируем структуру директории...")
        
        # Ищем папки с кошками и собаками
        cat_folders = []
        dog_folders = []
        other_folders = []
        
        for item in os.listdir(data_dir):
            item_path = os.path.join(data_dir, item)
            if os.path.isdir(item_path):
                lower_item = item.lower()
                
                # Проверяем, относится ли папка к кошкам
                if any(keyword in lower_item for keyword in ['cat', 'кот', 'кош', 'кис']):
                    cat_folders.append(item)
                # Проверяем, относится ли папка к собакам
                elif any(keyword in lower_item for keyword in ['dog', 'собак', 'пёс', 'пес', 'щен']):
                    dog_folders.append(item)
                else:
                    other_folders.append(item)
        
        # Формируем информационное сообщение
        info_text = ""
        
        if cat_folders:
            info_text += f"🐱 Обнаружены папки с кошками: {', '.join(cat_folders)}\n"
        if dog_folders:
            info_text += f"🐶 Обнаружены папки с собаками: {', '.join(dog_folders)}\n"
        if other_folders:
            info_text += f"📁 Другие папки: {', '.join(other_folders)}\n"
        
        if not cat_folders and not dog_folders:
            info_text = "❌ Не обнаружено папок с кошками или собаками. Создайте папки с названиями типа 'Cat', 'Cats', 'Dog', 'Dogs' и поместите туда изображения."
            self.process_btn.config(state='disabled')
        else:
            self.process_btn.config(state='normal')
        
        self.info_label.config(text=info_text)
        self.log_message(info_text)
    
    def add_prefixes(self):
        """Добавление префиксов к файлам в папках"""
        data_dir = self.data_dir_var.get()
        if not data_dir or not os.path.exists(data_dir):
            messagebox.showerror("Ошибка", "Выберите существующую директорию с данными")
            return
        
        try:
            self.log_message("=" * 50)
            self.log_message("Начинаем добавление префиксов...")
            
            total_processed = 0
            
            # Обрабатываем все папки в директории
            for folder_name in os.listdir(data_dir):
                folder_path = os.path.join(data_dir, folder_name)
                
                if os.path.isdir(folder_path):
                    lower_folder = folder_name.lower()
                    
                    # Определяем тип папки
                    prefix = None
                    if any(keyword in lower_folder for keyword in ['cat', 'кот', 'кош', 'кис']):
                        prefix = 'cat'
                    elif any(keyword in lower_folder for keyword in ['dog', 'собак', 'пёс', 'пес', 'щен']):
                        prefix = 'dog'
                    
                    if prefix:
                        processed = self.process_folder(folder_path, prefix, folder_name)
                        total_processed += processed
            
            self.log_message("=" * 50)
            self.log_message(f"✅ Обработка завершена! Всего обработано файлов: {total_processed}")
            
            messagebox.showinfo("Успех", f"Префиксы успешно добавлены!\nОбработано файлов: {total_processed}")
            
        except Exception as e:
            self.log_message(f"❌ Ошибка: {e}")
            messagebox.showerror("Ошибка", f"Произошла ошибка: {e}")
    
    def process_folder(self, folder_path, prefix, folder_name):
        """Обработка одной папки - добавление префиксов к файлам"""
        processed_count = 0
        file_number = 1
        
        self.log_message(f"📁 Обрабатываем папку: {folder_name} → префикс: '{prefix}'")
        
        # Получаем все файлы в папке
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            
            # Пропускаем подпапки
            if os.path.isdir(file_path):
                continue
                
            # Проверяем, что это изображение
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.jfif', '.webp')):
                
                # Проверяем, есть ли уже правильный префикс
                lower_filename = filename.lower()
                if lower_filename.startswith(prefix + '_'):
                    self.log_message(f"   ✓ Уже имеет префикс: {filename}")
                    processed_count += 1
                    continue
                
                # Добавляем префикс
                try:
                    extension = os.path.splitext(filename)[1]
                    new_name = f"{prefix}_{file_number:04d}{extension}"
                    new_path = os.path.join(folder_path, new_name)
                    
                    # Переименовываем файл
                    os.rename(file_path, new_path)
                    
                    self.log_message(f"   ✅ Переименовано: {filename} → {new_name}")
                    processed_count += 1
                    file_number += 1
                    
                except Exception as e:
                    self.log_message(f"   ❌ Ошибка переименования {filename}: {e}")
        
        self.log_message(f"   📊 В папке '{folder_name}' обработано: {processed_count} файлов")
        return processed_count

# Запуск приложения
if __name__ == "__main__":
    print("Запуск умного организатора данных...")
    app = DataPreprocessor()
    app.root.mainloop()