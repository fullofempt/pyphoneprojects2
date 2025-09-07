import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class LinearRegressionApp:
    def __init__(self, master): # Переменные для данных и модели
        self.master = master
        master.title("Лабораторная работа №1 - Линейная регрессия")
        master.geometry("1000x700")

        self.data = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.coefficients = None
        self.mse = None

        self.create_widgets()

    def create_widgets(self):
        self.load_btn = tk.Button(self.master, text="Загрузить CSV", command=self.load_data)
        self.load_btn.pack(pady=5)

        tk.Label(self.master, text="Выберите признак (X):").pack(pady=5) # Выбор признака для однофакторной регрессии
        self.feature_var = tk.StringVar(value="Площадь жилой недвижимости, м2")
        feature_dropdown = ttk.Combobox(self.master, textvariable=self.feature_var, state="readonly")
        feature_dropdown['values'] = ("Площадь жилой недвижимости, м2", "Этаж", "Новостройка - 1, старое жилье - 0")
        feature_dropdown.pack(pady=5)

        tk.Label(self.master, text="Размер тестовой выборки (%):").pack(pady=5) # Слайдер для разделения данных
        self.test_size_var = tk.DoubleVar(value=0.2)
        test_size_slider = tk.Scale(self.master, from_=0.1, to=0.5, resolution=0.05, orient=tk.HORIZONTAL, variable=self.test_size_var)
        test_size_slider.pack(pady=5)

        # Кнопка обучения модели
        self.train_btn = tk.Button(self.master, text="Обучить модель", command=self.train_model, state=tk.DISABLED)
        self.train_btn.pack(pady=5)

        # Область для вывода результатов
        self.results_text = tk.Text(self.master, height=8, state=tk.DISABLED)
        self.results_text.pack(pady=5, fill=tk.X, padx=10)

        # Область для графика
        self.figure, self.ax = plt.subplots(figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.get_tk_widget().pack(pady=10, fill=tk.BOTH, expand=True)

    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return
        try:
            # Загрузка данных, разделитель - точка с запятой
            self.data = pd.read_csv(file_path, sep=';', encoding='utf-8')
            messagebox.showinfo("Успех", f"Данные загружены! Записей: {len(self.data)}")
            self.train_btn.config(state=tk.NORMAL) # Активируем кнопку обучения
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить файл:\n{e}")

    def train_model(self):
        if self.data is None:
            return

        selected_feature = self.feature_var.get()
        test_size = self.test_size_var.get()

        try:
            # Извлечение признака X и целевой переменной Y
            X = self.data[selected_feature].values
            y = self.data['Y(X1,X2,X3) Целевой параметр - стоимость жилья, руб.'].values

            # Разделение на обучающую и тестовую выборки
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # --- РЕАЛИЗАЦИЯ МАТЕМАТИКИ ЛИНЕЙНОЙ РЕГРЕССИИ ---
            # Вычисление коэффициентов по МНК: w1 = cov(X,Y) / var(X); w0 = mean(Y) - w1 * mean(X)
            mean_x = np.mean(self.X_train)
            mean_y = np.mean(self.y_train)
            covariance = np.sum((self.X_train - mean_x) * (self.y_train - mean_y))
            variance = np.sum((self.X_train - mean_x) ** 2)
            w1 = covariance / variance
            w0 = mean_y - w1 * mean_x
            self.coefficients = (w0, w1)
            # -------------------------------------------------

            # Прогноз на тестовой выборке
            y_pred = w0 + w1 * self.X_test

            # Вычисление среднеквадратичной ошибки (MSE)
            self.mse = np.mean((self.y_test - y_pred) ** 2)

            # Обновление интерфейса
            self.update_results(selected_feature)
            self.plot_data(selected_feature)

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при обучении модели:\n{e}")

    def update_results(self, feature_name):
        """Выводит результаты в текстовое поле."""
        w0, w1 = self.coefficients
        result_str = (
            f"РЕЗУЛЬТАТЫ МОДЕЛИ:\n"
            f"Целевая функция: Y = {w0:.2f} + {w1:.2f} * {feature_name}\n"
            f"Среднеквадратичная ошибка (MSE): {self.mse:.2f}\n"
            f"Размер обучающей выборки: {len(self.X_train)}\n"
            f"Размер тестовой выборки: {len(self.X_test)}"
        )
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, result_str)
        self.results_text.config(state=tk.DISABLED)

    def plot_data(self, feature_name):
        """Строит график с точками и линией регрессии."""
        self.ax.clear() # Очищаем предыдущий график

        # Тренировочные данные
        self.ax.scatter(self.X_train, self.y_train, color='blue', alpha=0.7, label='Обучающая выборка')
        # Тестовые данные
        self.ax.scatter(self.X_test, self.y_test, color='red', alpha=0.7, label='Тестовая выборка')

        # Линия регрессии
        x_min, x_max = np.min(self.X_train), np.max(self.X_train)
        x_line = np.array([x_min, x_max])
        y_line = self.coefficients[0] + self.coefficients[1] * x_line
        self.ax.plot(x_line, y_line, color='green', linewidth=2, label='Линейная регрессия')

        # Настройки графика
        self.ax.set_xlabel(feature_name)
        self.ax.set_ylabel('Стоимость жилья, руб.')
        self.ax.set_title('Линейная регрессия: Зависимость стоимости от выбранного признака')
        self.ax.legend()
        self.ax.grid(True)

        # Обновляем холст в GUI
        self.canvas.draw()

# Запуск приложения
if __name__ == "__main__":
    root = tk.Tk()
    app = LinearRegressionApp(root)
    root.mainloop()