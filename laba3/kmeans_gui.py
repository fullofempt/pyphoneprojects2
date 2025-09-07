import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Tuple

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def parse_dataset(text: str) -> np.ndarray:
    rows = []
    for raw in text.strip().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        # split by whitespace or semicolon or comma or tab
        parts = [p for p in raw.replace(';', ' ').replace(',', ' ').split() if p]
        if len(parts) < 2:
            continue
        # allow multiple pairs per row: X1 X2 X1 X2 ...
        nums = []
        ok = True
        for p in parts:
            try:
                nums.append(float(p))
            except ValueError:
                ok = False
                break
        if not ok:
            continue
        # collect pairs
        for i in range(0, len(nums) - 1, 2):
            rows.append([nums[i], nums[i + 1]])
    if not rows:
        raise ValueError('Не удалось разобрать данные. Проверьте формат.')
    return np.array(rows, dtype=float)


def kmeans(X: np.ndarray, k: int, max_iter: int = 100, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if k <= 0 or k > n:
        raise ValueError('K должно быть в диапазоне [1, количество точек]')

    # init centroids by random choice of points
    idx = rng.choice(n, size=k, replace=False)
    centroids = X[idx].copy()

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # assign
        dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # update
        for i in range(k):
            pts = X[labels == i]
            if len(pts) > 0:
                centroids[i] = pts.mean(axis=0)
    return labels, centroids


def compute_inertia(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """Sum of squared distances from each point to its assigned centroid."""
    sse = 0.0
    for i, c in enumerate(labels):
        diff = X[i] - centroids[c]
        sse += float(diff @ diff)
    return sse


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title('K-средних: кластеризация (X1, X2)')
        self.geometry('900x600')

        self._make_widgets()

    def _make_widgets(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=8, pady=8)

        ttk.Label(top, text='K:').pack(side=tk.LEFT)
        self.k_var = tk.StringVar(value='3')
        ttk.Entry(top, width=5, textvariable=self.k_var).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(top, text='Итераций:').pack(side=tk.LEFT)
        self.it_var = tk.StringVar(value='100')
        ttk.Entry(top, width=7, textvariable=self.it_var).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(top, text='Seed:').pack(side=tk.LEFT)
        self.seed_var = tk.StringVar(value='0')
        ttk.Entry(top, width=7, textvariable=self.seed_var).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Button(top, text='Кластеризовать', command=self.on_run).pack(side=tk.LEFT)
        ttk.Button(top, text='Очистить', command=self.on_clear).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(top, text='Открыть CSV', command=self.on_open_csv).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(top, text='Сохранить CSV', command=self.on_save_csv).pack(side=tk.LEFT, padx=(8, 0))

        self.err_var = tk.StringVar(value='Ошибка (SSE): -')
        ttk.Label(top, textvariable=self.err_var).pack(side=tk.LEFT, padx=(16, 0))

        body = ttk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.txt = tk.Text(body, width=40)
        self.txt.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)

        fig = Figure(figsize=(5, 4), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.set_xlabel('X1')
        self.ax.set_ylabel('X2')
        self.ax.grid(True, alpha=0.3)
        self.canvas = FigureCanvasTkAgg(fig, master=body)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # hint
        self.txt.insert('1.0', 'Вставьте сюда строки вида: X1 X2\nНапример:\n600 500\n250 300\n300 700\n')

    def on_clear(self) -> None:
        self.txt.delete('1.0', tk.END)
        self.ax.cla()
        self.ax.set_xlabel('X1')
        self.ax.set_ylabel('X2')
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()

    def on_run(self) -> None:
        try:
            X = parse_dataset(self.txt.get('1.0', tk.END))
            k = int(self.k_var.get())
            iters = int(self.it_var.get())
            seed = int(self.seed_var.get())
        except Exception as e:
            messagebox.showerror('Ошибка', str(e))
            return

        labels, centroids = kmeans(X, k, max_iter=iters, seed=seed)
        inertia = compute_inertia(X, labels, centroids)

        # plot
        self.ax.cla()
        colors = np.array(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'])
        if k > len(colors):
            # repeat colors if k is large
            colors = np.resize(colors, k)
        for i in range(k):
            pts = X[labels == i]
            if len(pts) > 0:
                self.ax.scatter(pts[:, 0], pts[:, 1], s=20, c=colors[i], label=f'Кластер {i+1}')
        self.ax.scatter(centroids[:, 0], centroids[:, 1], s=120, marker='X', c='black', label='Центроиды')
        self.ax.set_xlabel('X1')
        self.ax.set_ylabel('X2')
        self.ax.legend(loc='best')
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()
        self.err_var.set(f'Ошибка (SSE): {inertia:.2f}')

    def on_open_csv(self) -> None:
        path = filedialog.askopenfilename(
            title='Выберите CSV с двумя колонками X1,X2',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]
        )
        if not path:
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as e:
            messagebox.showerror('Ошибка чтения файла', str(e))
            return
        # Put file content into text area (parser поддерживает запятые/точки с запятой/пробелы)
        self.txt.delete('1.0', tk.END)
        self.txt.insert('1.0', text)

    def on_save_csv(self) -> None:
        try:
            X = parse_dataset(self.txt.get('1.0', tk.END))
        except Exception as e:
            messagebox.showerror('Ошибка', str(e))
            return
        path = filedialog.asksaveasfilename(
            title='Сохранить как CSV',
            defaultextension='.csv',
            filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]
        )
        if not path:
            return
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write('X1,X2\n')
                for x1, x2 in X:
                    f.write(f'{x1},{x2}\n')
        except Exception as e:
            messagebox.showerror('Ошибка записи файла', str(e))


if __name__ == '__main__':
    App().mainloop()


