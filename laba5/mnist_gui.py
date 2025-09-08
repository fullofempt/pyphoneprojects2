import io
import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Optional

import numpy as np


def ensure_tf_cpu() -> None:
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')


class MNISTApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title('MNIST: распознавание рукописных цифр')
        self.geometry('900x600')

        self.model = None  # type: ignore
        self.canvas_size = 280
        self.brush = 14

        self._make_widgets()

    def _make_widgets(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=8, pady=8)

        ttk.Button(top, text='Обучить', command=self.on_train).pack(side=tk.LEFT)
        ttk.Button(top, text='Загрузить модель', command=self.on_load).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(top, text='Эпох:').pack(side=tk.LEFT, padx=(12, 0))
        self.epochs_var = tk.StringVar(value='3')
        ttk.Entry(top, width=5, textvariable=self.epochs_var).pack(side=tk.LEFT)

        body = ttk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        left = ttk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.Y)
        self.canvas = tk.Canvas(left, width=self.canvas_size, height=self.canvas_size, bg='white')
        self.canvas.pack()
        self.canvas.bind('<B1-Motion>', self._paint)
        self.canvas.bind('<Button-1>', self._paint)
        ttk.Button(left, text='Очистить', command=self._clear).pack(pady=6)
        ttk.Button(left, text='Распознать', command=self.on_predict).pack()
        # Large prediction label
        self.pred_var = tk.StringVar(value='?')
        self.pred_label = ttk.Label(left, textvariable=self.pred_var)
        self.pred_label.configure(font=("Arial", 36, "bold"))
        self.pred_label.pack(pady=8)

        right = ttk.Frame(body)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        self.txt = tk.Text(right)
        self.txt.pack(fill=tk.BOTH, expand=True)
        # Автозаполнение числами 0..100
        self._auto_fill_numbers()

        self.status = tk.StringVar(value='Готово')
        ttk.Label(self, textvariable=self.status).pack(fill=tk.X, padx=8, pady=(0, 8))

    # Drawing
    def _paint(self, event) -> None:
        x, y = event.x, event.y
        r = self.brush
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill='black', outline='black')

    def _clear(self) -> None:
        self.canvas.delete('all')

    # Data and model
    def _load_data(self):
        ensure_tf_cpu()
        from tensorflow import keras
        mnist = keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype('float32')/255.0
        x_test = x_test.astype('float32')/255.0
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        return (x_train, y_train), (x_test, y_test)

    def _build_model(self):
        ensure_tf_cpu()
        from tensorflow import keras
        layers = keras.layers
        model = keras.models.Sequential([
            layers.Input(shape=(28,28,1)),
            layers.Conv2D(32, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.Conv2D(64, (3,3), activation='relu'),
            layers.MaxPooling2D((2,2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax'),
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def on_train(self) -> None:
        try:
            epochs = int(self.epochs_var.get())
        except Exception:
            epochs = 3
        self.status.set('Загрузка данных...')
        self.update_idletasks()
        (x_train, y_train), (x_test, y_test) = self._load_data()

        self.status.set('Построение модели...')
        self.update_idletasks()
        self.model = self._build_model()

        self.status.set('Обучение...')
        self.update_idletasks()
        history = self.model.fit(x_train, y_train, epochs=epochs, batch_size=128, validation_split=0.1, verbose=0)
        loss, acc = self.model.evaluate(x_test, y_test, verbose=0)
        self.status.set(f'Готово. Точность: {acc:.4f}')
        self.txt.insert(tk.END, f'Обучение завершено. Test accuracy={acc:.4f}\n')

    def _auto_fill_numbers(self) -> None:
        try:
            nums = ' '.join(str(i) for i in range(10))
            self.txt.insert('1.0', nums + '\n')
        except Exception:
            pass

    def _get_canvas_image(self) -> np.ndarray:
        self.canvas.update()
        try:
            from PIL import ImageGrab
        except Exception:
            messagebox.showerror('Ошибка', 'Установите pillow: pip install pillow')
            raise
        x0 = self.canvas.winfo_rootx()
        y0 = self.canvas.winfo_rooty()
        x1 = x0 + self.canvas.winfo_width()
        y1 = y0 + self.canvas.winfo_height()
        img = ImageGrab.grab(bbox=(x0, y0, x1, y1)).convert('L').resize((28, 28))
        arr = np.array(img).astype('float32')
        arr = 255 - arr  # черное на белом -> белое на черном
        arr = arr/255.0
        arr = np.clip(arr, 0, 1)
        arr = np.expand_dims(arr, axis=(0,-1))
        return arr

    def on_predict(self) -> None:
        if self.model is None:
            messagebox.showerror('Ошибка', 'Сначала обучите или загрузите модель')
            return
        try:
            img = self._get_canvas_image()
        except Exception:
            return
        probs = self.model.predict(img, verbose=0)[0]
        pred = int(np.argmax(probs))
        self.pred_var.set(f'Цифра: {pred}')
        self.txt.insert(tk.END, f'Предсказание: {pred}  probs={[round(float(p),4) for p in probs]}\n')

    def on_load(self) -> None:
        ensure_tf_cpu()
        path = filedialog.askopenfilename(filetypes=[('Keras model', '*.h5;*.keras'), ('All', '*.*')])
        if not path:
            return
        from tensorflow import keras
        try:
            self.model = keras.models.load_model(path)
            self.status.set(f'Загружено: {path}')
        except Exception as e:
            messagebox.showerror('Ошибка загрузки', str(e))


if __name__ == '__main__':
    ensure_tf_cpu()
    MNISTApp().mainloop()


