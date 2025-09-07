import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

class KnnApp:
    def __init__(self, master):
        self.master = master
        master.title("Лабораторная работа №2 - KNN Классификация")
        master.geometry("1400x900")

        # Переменные для данных и модели
        self.data = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.k_values = [3, 5, 7, 9, 11, 13, 15]
        self.errors = {}
        self.history = {}  # История для каждого K
        self.previous_errors = {}  # Предыдущие ошибки для расчета разницы
        self.iteration_step = 10  # Шаг итераций для вывода истории

        # Создание элементов GUI
        self.create_widgets()

    def create_widgets(self):
        # Фрейм для управляющих элементов
        control_frame = tk.Frame(self.master)
        control_frame.pack(pady=10, fill=tk.X)

        # Кнопка загрузки файла
        self.load_btn = tk.Button(control_frame, text="Загрузить CSV", command=self.load_data)
        self.load_btn.pack(side=tk.LEFT, padx=5)

        # Слайдер для разделения данных
        tk.Label(control_frame, text="Тестовая выборка (%):").pack(side=tk.LEFT, padx=5)
        self.test_size_var = tk.DoubleVar(value=0.2)
        test_size_slider = tk.Scale(control_frame, from_=0.1, to=0.5, resolution=0.05, 
                                  orient=tk.HORIZONTAL, variable=self.test_size_var, length=100)
        test_size_slider.pack(side=tk.LEFT, padx=5)

        # Выбор шага итераций
        tk.Label(control_frame, text="Шаг итераций:").pack(side=tk.LEFT, padx=5)
        self.step_var = tk.IntVar(value=10)
        step_spinbox = tk.Spinbox(control_frame, from_=1, to=50, textvariable=self.step_var, 
                                 width=5, command=self.update_step)
        step_spinbox.pack(side=tk.LEFT, padx=5)

        # Кнопка обучения и анализа модели
        self.train_btn = tk.Button(control_frame, text="Запустить анализ KNN", 
                                 command=self.run_analysis, state=tk.DISABLED)
        self.train_btn.pack(side=tk.LEFT, padx=5)

        # Кнопка показа истории
        self.history_btn = tk.Button(control_frame, text="Показать историю", 
                                   command=self.show_history, state=tk.DISABLED)
        self.history_btn.pack(side=tk.LEFT, padx=5)

        # Прогресс бар
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, 
                                          maximum=100, length=150)
        self.progress_bar.pack(side=tk.LEFT, padx=10)
        self.progress_label = tk.Label(control_frame, text="0%")
        self.progress_label.pack(side=tk.LEFT, padx=5)

        # Фрейм для выбора K для истории
        history_control_frame = tk.Frame(self.master)
        history_control_frame.pack(pady=5, fill=tk.X)

        tk.Label(history_control_frame, text="Выберите K для истории:").pack(side=tk.LEFT, padx=5)
        self.history_k_var = tk.StringVar(value="3")
        history_k_dropdown = ttk.Combobox(history_control_frame, textvariable=self.history_k_var, 
                                         values=[str(k) for k in self.k_values], state="readonly", width=5)
        history_k_dropdown.pack(side=tk.LEFT, padx=5)
        history_k_dropdown.bind('<<ComboboxSelected>>', self.on_history_k_change)

        # Область для вывода результатов
        results_frame = tk.Frame(self.master)
        results_frame.pack(pady=5, fill=tk.BOTH, expand=True)

        # Текстовое поле для основых результатов
        self.results_text = tk.Text(results_frame, height=12, state=tk.DISABLED)
        self.results_text.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 5))

        # Текстовое поле для истории
        self.history_text = tk.Text(results_frame, height=8, state=tk.DISABLED)
        self.history_text.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(5, 0))

        # Полосы прокрутки
        for text_widget in [self.results_text, self.history_text]:
            scrollbar = tk.Scrollbar(text_widget)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            text_widget.config(yscrollcommand=scrollbar.set)
            scrollbar.config(command=text_widget.yview)

        # Фрейм для графиков
        plot_frame = tk.Frame(self.master)
        plot_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        # График данных
        self.figure1, self.ax1 = plt.subplots(figsize=(5, 5))
        self.canvas1 = FigureCanvasTkAgg(self.figure1, master=plot_frame)
        self.canvas1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        # График ошибки
        self.figure2, self.ax2 = plt.subplots(figsize=(5, 4))
        self.canvas2 = FigureCanvasTkAgg(self.figure2, master=plot_frame)
        self.canvas2.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

    def update_step(self):
        """Обновляет шаг итераций."""
        self.iteration_step = self.step_var.get()

    def on_history_k_change(self, event=None):
        """Обработчик изменения выбранного K для истории."""
        if self.history:
            self.show_history()

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        try:
            self.data = pd.read_csv(file_path, encoding='utf-8')
            messagebox.showinfo("Успех", f"Данные загружены! Записей: {len(self.data)}")
            self.train_btn.config(state=tk.NORMAL)
            self.plot_initial_data()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл:\n{e}")

    def plot_initial_data(self):
        """Отображает исходные данные на первом графике."""
        if self.data is None:
            return
        self.ax1.clear()
        colors = {1: 'red', 2: 'green', 3: 'blue'}
        for class_label, color in colors.items():
            class_data = self.data[self.data['Class'] == class_label]
            self.ax1.scatter(class_data['X1'], class_data['X2'], c=color, 
                           label=f'Класс {class_label}', alpha=0.6, edgecolors='w', s=50)
        self.ax1.set_title('Исходные данные (3 класса)')
        self.ax1.set_xlabel('X1')
        self.ax1.set_ylabel('X2')
        self.ax1.legend()
        self.ax1.grid(True)
        self.canvas1.draw()

    def run_analysis(self):
        if self.data is None:
            return

        # Блокируем кнопки во время выполнения
        self.train_btn.config(state=tk.DISABLED)
        self.history_btn.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.progress_label.config(text="0%")
        self.master.update()

        test_size = self.test_size_var.get()
        X = self.data[['X1', 'X2']].values
        y = self.data['Class'].values

        # Разделение на обучающую и тестовую выборки
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Очищаем словари для нового анализа
        self.errors = {}
        self.history = {k: [] for k in self.k_values}
        self.previous_errors = {k: 0 for k in self.k_values}

        # Обновляем текстовое поле
        self.update_results_header()

        # Анализ для каждого значения K
        total_k = len(self.k_values)
        for idx, k in enumerate(self.k_values):
            self.update_progress(f"Обрабатывается K = {k}", idx * 100 / total_k)
            error = self.calculate_error_with_history(k)
            self.errors[k] = error
            self.update_progress(f"Завершено K = {k}", (idx + 1) * 100 / total_k)

        # Находим оптимальное K (с минимальной ошибкой)
        optimal_k = min(self.errors, key=self.errors.get)

        # Обновление интерфейса
        self.update_results_footer(optimal_k)
        self.plot_error_curve()
        
        # Разблокируем кнопки
        self.train_btn.config(state=tk.NORMAL)
        self.history_btn.config(state=tk.NORMAL)

    def update_progress(self, message, progress):
        """Обновляет прогресс бар и выводит сообщение."""
        self.progress_var.set(progress)
        self.progress_label.config(text=f"{progress:.1f}%")
        self.append_to_results(f"⏳ {message}...\n")
        self.master.update()

    def euclidean_distance(self, point1, point2):
        """Вычисляет евклидово расстояние между двумя точками."""
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def knn_predict(self, k, train_data, train_labels, test_point):
        """
        Реализация алгоритма KNN для одного тестового объекта.
        Возвращает предсказанный класс.
        """
        # Вычисляем расстояния от test_point до всех точек в train_data
        distances = [self.euclidean_distance(test_point, train_point) for train_point in train_data]
        # Получаем индексы k ближайших соседей
        k_indices = np.argsort(distances)[:k]
        # Получаем метки этих соседей
        k_nearest_labels = [train_labels[i] for i in k_indices]
        # Возвращаем наиболее частый класс (моду)
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    def calculate_error_with_history(self, k):
        """Вычисляет суммарную погрешность для данного значения K с историей."""
        misclassified_count = 0
        total_test = len(self.X_test)
        step = self.iteration_step
        
        for i, test_point in enumerate(self.X_test):
            prediction = self.knn_predict(k, self.X_train, self.y_train, test_point)
            if prediction != self.y_test[i]:
                misclassified_count += 1
            
            # Сохраняем историю с заданным шагом итераций
            if (i + 1) % step == 0 or i == total_test - 1:
                current_error = misclassified_count / (i + 1)
                previous_error = self.previous_errors.get(k, 0)
                error_diff = current_error - previous_error
                
                self.history[k].append({
                    'iteration': i + 1,
                    'error': current_error,
                    'diff': error_diff,
                    'misclassified': misclassified_count,
                    'total_processed': i + 1
                })
                
                self.previous_errors[k] = current_error
                
                # Выводим прогресс с заданным шагом
                if (i + 1) % step == 0:
                    self.append_to_results(
                        f"   K={k}: Итерация {i+1}/{total_test}, "
                        f"Ошибка: {current_error:.4f}, "
                        f"Δ: {error_diff:+.4f}\n"
                    )
        
        final_error_rate = misclassified_count / total_test
        return final_error_rate

    def update_results_header(self):
        """Очищает и добавляет заголовок в текстовое поле."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "🚀 НАЧАЛО АНАЛИЗА KNN\n")
        self.results_text.insert(tk.END, f"Шаг итераций: {self.iteration_step}\n")
        self.results_text.insert(tk.END, "=" * 60 + "\n\n")
        self.results_text.config(state=tk.DISABLED)

    def append_to_results(self, text):
        """Добавляет текст в результаты."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.insert(tk.END, text)
        self.results_text.see(tk.END)  # Автопрокрутка к концу
        self.results_text.config(state=tk.DISABLED)
        self.master.update()

    def append_to_history(self, text):
        """Добавляет текст в историю."""
        self.history_text.config(state=tk.NORMAL)
        self.history_text.insert(tk.END, text)
        self.history_text.see(tk.END)
        self.history_text.config(state=tk.DISABLED)
        self.master.update()

    def update_results_footer(self, optimal_k):
        """Добавляет итоговые результаты."""
        self.append_to_results("\n" + "=" * 60 + "\n")
        self.append_to_results("✅ АНАЛИЗ ЗАВЕРШЕН\n\n")
        
        self.append_to_results("ИТОГОВЫЕ РЕЗУЛЬТАТЫ:\n")
        self.append_to_results(f"Размер обучающей выборки: {len(self.X_train)}\n")
        self.append_to_results(f"Размер тестовой выборки: {len(self.X_test)}\n")
        self.append_to_results(f"Шаг итераций: {self.iteration_step}\n\n")
        
        self.append_to_results("Суммарная погрешность (ε) для разных K:\n")
        for k, error in self.errors.items():
            self.append_to_results(f"K = {k}: ε = {error:.4f}\n")
        
        self.append_to_results(f"\n🎯 Оптимальное значение K (минимум ошибки): {optimal_k}\n")

    def show_history(self):
        """Показывает историю для выбранного значения K."""
        if not self.history:
            messagebox.showinfo("Информация", "Сначала запустите анализ KNN")
            return
        
        selected_k = int(self.history_k_var.get())
        
        if selected_k not in self.history or not self.history[selected_k]:
            self.append_to_history(f"❌ Нет данных истории для K={selected_k}\n")
            return
        
        self.append_to_history(f"\n{'='*60}\n")
        self.append_to_history(f"📊 ДЕТАЛЬНАЯ ИСТОРИЯ ДЛЯ K={selected_k}\n")
        self.append_to_history(f"{'='*60}\n\n")
        
        self.append_to_history("Итерация | Обработано | Ошибочно | Ошибка    | Δ-изменение\n")
        self.append_to_history("--------|-----------|----------|-----------|------------\n")
        
        for record in self.history[selected_k]:
            self.append_to_history(
                f"{record['iteration']:7d} | "
                f"{record['total_processed']:9d} | "
                f"{record['misclassified']:8d} | "
                f"{record['error']:8.4f} | "
                f"{record['diff']:+.4f}\n"
            )
        
        # Добавляем статистику
        final_error = self.errors.get(selected_k, 0)
        self.append_to_history(f"\n📈 Финальная ошибка для K={selected_k}: {final_error:.4f}\n")

    def plot_error_curve(self):
        """Строит график зависимости ошибки от значения K."""
        self.ax2.clear()
        k_list = list(self.errors.keys())
        error_list = list(self.errors.values())
        
        self.ax2.plot(k_list, error_list, 'bo-', markersize=8, linewidth=2)
        self.ax2.set_title('Зависимость ошибки ε от количества соседей K')
        self.ax2.set_xlabel('Количество соседей (K)')
        self.ax2.set_ylabel('Суммарная погрешность (ε)')
        self.ax2.grid(True, alpha=0.3)
        self.ax2.set_xticks(k_list)
        
        # Подсвечиваем оптимальное K
        optimal_k = min(self.errors, key=self.errors.get)
        optimal_error = self.errors[optimal_k]
        self.ax2.plot(optimal_k, optimal_error, 'ro', markersize=10, 
                     label=f'Оптимальное K={optimal_k}')
        self.ax2.legend()
        
        self.canvas2.draw()

# Запуск приложения
if __name__ == "__main__":
    root = tk.Tk()
    app = KnnApp(root)
    root.mainloop()