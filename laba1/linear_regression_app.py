import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class LinearRegressionApp:
    def __init__(self, master):
        self.master = master
        master.title("Лабораторная работа №1 - Линейная регрессия")
        master.geometry("1200x800")

        # Переменные для данных и модели
        self.data = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.coefficients = None
        self.mse = None
        self.iteration_history = []  # История итераций
        self.current_feature = None

        self.create_widgets()

    def create_widgets(self):
        # Фрейм для управления
        control_frame = tk.Frame(self.master)
        control_frame.pack(pady=5, fill=tk.X)

        # Кнопка загрузки файла
        self.load_btn = tk.Button(control_frame, text="Загрузить CSV", command=self.load_data)
        self.load_btn.pack(side=tk.LEFT, padx=5)

        # Выбор признака
        tk.Label(control_frame, text="Признак (X):").pack(side=tk.LEFT, padx=5)
        self.feature_var = tk.StringVar(value="Площадь жилой недвижимости, м2")
        feature_dropdown = ttk.Combobox(control_frame, textvariable=self.feature_var, 
                                      state="readonly", width=25)
        feature_dropdown['values'] = ("Площадь жилой недвижимости, м2", "Этаж", 
                                    "Новостройка - 1, старое жилье - 0")
        feature_dropdown.pack(side=tk.LEFT, padx=5)

        # Выбор количества итераций
        tk.Label(control_frame, text="Итерации:").pack(side=tk.LEFT, padx=5)
        self.iterations_var = tk.IntVar(value=100)
        iterations_spin = tk.Spinbox(control_frame, from_=10, to=1000, 
                                   increment=10, textvariable=self.iterations_var, width=8)
        iterations_spin.pack(side=tk.LEFT, padx=5)

        # Размер тестовой выборки
        tk.Label(control_frame, text="Тестовая выборка (%):").pack(side=tk.LEFT, padx=5)
        self.test_size_var = tk.DoubleVar(value=0.2)
        test_size_slider = tk.Scale(control_frame, from_=0.1, to=0.5, 
                                  resolution=0.05, orient=tk.HORIZONTAL, 
                                  variable=self.test_size_var, length=100)
        test_size_slider.pack(side=tk.LEFT, padx=5)

        # Кнопка обучения
        self.train_btn = tk.Button(control_frame, text="Обучить модель", 
                                 command=self.train_model, state=tk.DISABLED)
        self.train_btn.pack(side=tk.LEFT, padx=5)

        # Кнопка истории итераций
        self.history_btn = tk.Button(control_frame, text="История итераций", 
                                   command=self.show_history, state=tk.DISABLED)
        self.history_btn.pack(side=tk.LEFT, padx=5)

        # Область для вывода результатов
        results_frame = tk.Frame(self.master)
        results_frame.pack(pady=5, fill=tk.X, padx=10)

        self.results_text = tk.Text(results_frame, height=12, state=tk.DISABLED)
        self.results_text.pack(fill=tk.BOTH, expand=True)

        # Область для графика
        self.figure, self.ax = plt.subplots(figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.get_tk_widget().pack(pady=10, fill=tk.BOTH, expand=True)

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        try:
            self.data = pd.read_csv(file_path, sep=';', encoding='utf-8')
            messagebox.showinfo("Успех", f"Данные загружены! Записей: {len(self.data)}\n"
                                       f"Диапазон стоимости: {self.data.iloc[:, -1].min():,} - "
                                       f"{self.data.iloc[:, -1].max():,} руб.")
            self.train_btn.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл:\n{e}")

    def train_model(self):
        if self.data is None:
            return

        selected_feature = self.feature_var.get()
        test_size = self.test_size_var.get()
        iterations = self.iterations_var.get()
        self.current_feature = selected_feature
        self.iteration_history = []  # Очищаем историю

        try:
            # Извлечение данных
            X = self.data[selected_feature].values
            y = self.data['Y(X1,X2,X3) Целевой параметр - стоимость жилья, руб.'].values

            # Разделение на выборки
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Нормализация данных для градиентного спуска
            X_train_norm = (self.X_train - np.mean(self.X_train)) / np.std(self.X_train)
            y_train_norm = (self.y_train - np.mean(self.y_train)) / np.std(self.y_train)

            # Градиентный спуск
            w0, w1 = 0, 0  # Начальные значения
            learning_rate = 0.01
            n = len(X_train_norm)

            for i in range(iterations):
                # Предсказания
                y_pred = w0 + w1 * X_train_norm
                
                # Градиенты
                dw0 = (-2/n) * np.sum(y_train_norm - y_pred)
                dw1 = (-2/n) * np.sum((y_train_norm - y_pred) * X_train_norm)
                
                # Обновление весов
                w0 -= learning_rate * dw0
                w1 -= learning_rate * dw1
                
                # Вычисление MSE для этой итерации
                mse = np.mean((y_train_norm - y_pred) ** 2)
                
                # Сохранение истории
                self.iteration_history.append({
                    'iteration': i + 1,
                    'w0': w0,
                    'w1': w1,
                    'mse': mse,
                    'dw0': dw0,
                    'dw1': dw1
                })

            # Денормализация коэффициентов
            X_mean = np.mean(self.X_train)
            X_std = np.std(self.X_train)
            y_mean = np.mean(self.y_train)
            y_std = np.std(self.y_train)
            
            w1_denorm = w1 * (y_std / X_std)
            w0_denorm = y_mean - w1_denorm * X_mean
            
            self.coefficients = (w0_denorm, w1_denorm)

            # Прогноз и оценка на тестовой выборке
            y_pred_test = w0_denorm + w1_denorm * self.X_test
            self.mse = np.mean((self.y_test - y_pred_test) ** 2)

            # Обновление интерфейса
            self.update_results(selected_feature)
            self.plot_data(selected_feature)
            self.history_btn.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при обучении модели:\n{e}")

    def update_results(self, feature_name):
        """Выводит результаты в текстовое поле."""
        w0, w1 = self.coefficients
        avg_price = np.mean(self.y_train)
        price_per_sqm = w1 if feature_name == "Площадь жилой недвижимости, м2" else None
        
        result_str = (
            f"РЕЗУЛЬТАТЫ МОДЕЛИ:\n"
            f"Целевая функция: Y = {w0:,.2f} + {w1:,.2f} * {feature_name}\n"
            f"Среднеквадратичная ошибка (MSE): {self.mse:,.2f}\n"
            f"Средняя ошибка: ±{np.sqrt(self.mse):,.0f} руб.\n"
            f"Размер обучающей выборки: {len(self.X_train)}\n"
            f"Размер тестовой выборки: {len(self.X_test)}\n"
            f"Количество итераций: {self.iterations_var.get()}\n"
            f"\nАНАЛИЗ СТОИМОСТИ ЖИЛЬЯ:\n"
            f"Средняя стоимость: {avg_price:,.0f} руб.\n"
        )
        
        if price_per_sqm:
            result_str += f"Стоимость 1 м²: {price_per_sqm:,.0f} руб.\n"
            result_str += f"Базовая стоимость (без площади): {w0:,.0f} руб.\n"
        
        result_str += (
            f"Диапазон стоимости в данных: {min(self.y_train):,.0f} - {max(self.y_train):,.0f} руб.\n"
            f"Модель объясняет {self.calculate_r2():.1%} дисперсии цены"
        )

        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, result_str)
        self.results_text.config(state=tk.DISABLED)

    def calculate_r2(self):
        """Вычисляет коэффициент детерминации R²."""
        y_pred = self.coefficients[0] + self.coefficients[1] * self.X_train
        ss_res = np.sum((self.y_train - y_pred) ** 2)
        ss_tot = np.sum((self.y_train - np.mean(self.y_train)) ** 2)
        return 1 - (ss_res / ss_tot)

    def plot_data(self, feature_name):
        """Строит график с точками и линией регрессии."""
        self.ax.clear()

        # Данные
        self.ax.scatter(self.X_train, self.y_train, color='blue', alpha=0.7, 
                       label='Обучающая выборка', s=30)
        self.ax.scatter(self.X_test, self.y_test, color='red', alpha=0.7, 
                       label='Тестовая выборка', s=30)

        # Линия регрессии
        x_min, x_max = np.min(self.X_train), np.max(self.X_train)
        x_line = np.linspace(x_min, x_max, 100)
        y_line = self.coefficients[0] + self.coefficients[1] * x_line
        self.ax.plot(x_line, y_line, color='green', linewidth=3, 
                    label='Линейная регрессия')

        # Настройки графика
        self.ax.set_xlabel(feature_name, fontsize=12)
        self.ax.set_ylabel('Стоимость жилья, руб.', fontsize=12)
        self.ax.set_title(f'Зависимость стоимости от {feature_name}\n'
                         f'Y = {self.coefficients[0]:,.0f} + {self.coefficients[1]:,.0f}*X', 
                         fontsize=14)
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.ax.ticklabel_format(style='plain', axis='y')

        self.canvas.draw()

    def show_history(self):
        """Показывает окно с историей итераций."""
        if not self.iteration_history:
            messagebox.showinfo("История", "История итераций пуста!")
            return

        history_window = tk.Toplevel(self.master)
        history_window.title(f"История итераций - {self.current_feature}")
        history_window.geometry("800x600")

        # Текстовое поле для истории
        text_widget = tk.Text(history_window, wrap=tk.WORD)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Заголовок
        text_widget.insert(tk.END, "ИСТОРИЯ ИТЕРАЦИЙ ГРАДИЕНТНОГО СПУСКА\n")
        text_widget.insert(tk.END, "=" * 60 + "\n\n")

        # Добавляем каждую итерацию
        for i, hist in enumerate(self.iteration_history):
            if i % 10 == 0 or i == len(self.iteration_history) - 1:  # Каждая 10-я и последняя
                text_widget.insert(tk.END, 
                    f"Итерация {hist['iteration']}:\n"
                    f"  w0 = {hist['w0']:.6f}, w1 = {hist['w1']:.6f}\n"
                    f"  MSE = {hist['mse']:.6f}\n"
                    f"  Градиенты: dw0 = {hist['dw0']:.6f}, dw1 = {hist['dw1']:.6f}\n"
                    f"{'-' * 40}\n"
                )

        text_widget.config(state=tk.DISABLED)

        # Кнопка сохранения истории
        save_btn = tk.Button(history_window, text="Сохранить историю", 
                           command=lambda: self.save_history(text_widget))
        save_btn.pack(pady=5)

    def save_history(self, text_widget):
        """Сохраняет историю в файл."""
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(text_widget.get(1.0, tk.END))
                messagebox.showinfo("Успех", "История сохранена!")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить файл:\n{e}")

# Запуск приложения
if __name__ == "__main__":
    root = tk.Tk()
    app = LinearRegressionApp(root)
    root.mainloop()