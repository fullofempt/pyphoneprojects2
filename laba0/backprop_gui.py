import math
import random as rm
import tkinter as tk
from tkinter import ttk
from typing import List, Tuple

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


alfa: float = 1.0


def sigmoid(s: float) -> float:
    return 1.0 / (1.0 + math.exp(-alfa * s))


def sigmoid1(s: float) -> float:
    y = sigmoid(s)
    return y * (1.0 - y)


def build_dataset() -> np.ndarray:
    data = np.zeros((16, 6), dtype=float)
    idx = 0
    for x1 in range(2):
        for x2 in range(2):
            for x3 in range(2):
                for x4 in range(2):
                    data[idx, 0] = 1.0
                    data[idx, 1] = float(x1)
                    data[idx, 2] = float(x2)
                    data[idx, 3] = float(x3)
                    data[idx, 4] = float(x4)
                    y = (x1 and x2) or (x3 and x4)
                    data[idx, 5] = float(y)
                    idx += 1
    return data


class MLP:
    def __init__(self, input_size: int = 5, hidden_neurons: int = 3, lr: float = 0.5):
        self.w_hidden = np.array([[2 * rm.random() - 1.0 for _ in range(input_size)] for _ in range(hidden_neurons)],
                                 dtype=float)
        self.w_out = np.array([2 * rm.random() - 1.0 for _ in range(hidden_neurons + 1)], dtype=float)
        self.lr = lr

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, float, List[float]]:
        s_hidden = self.w_hidden @ x
        y_hidden = np.array([sigmoid(s) for s in s_hidden], dtype=float)
        y_full = np.concatenate(([1.0], y_hidden))
        s4 = float(self.w_out @ y_full)
        y4 = sigmoid(s4)
        return y_hidden, y4, [*s_hidden, s4]

    def train_epoch(self, data: np.ndarray) -> float:
        indices = list(range(data.shape[0]))
        rm.shuffle(indices)
        sq_errors = 0.0
        for i in indices:
            x = data[i, 0:5]
            target = data[i, 5]

            y_hidden, y4, s_vals = self.forward(x)
            s_hidden = np.array(s_vals[:-1])
            s4 = s_vals[-1]

            delta4 = sigmoid1(s4) * (target - y4)

            deltas_h = np.zeros_like(y_hidden)
            for h in range(len(y_hidden)):
                deltas_h[h] = sigmoid1(float(s_hidden[h])) * delta4 * self.w_out[h + 1]

            y_full = np.concatenate(([1.0], y_hidden))
            self.w_out += self.lr * delta4 * y_full
            for h in range(self.w_hidden.shape[0]):
                self.w_hidden[h, :] += self.lr * deltas_h[h] * x

            err = float(target - y4)
            sq_errors += err * err

        rmse = math.sqrt(sq_errors / data.shape[0])
        return rmse

    def predict(self, x_bits: Tuple[int, int, int, int]) -> float:
        x = np.array([1.0, float(x_bits[0]), float(x_bits[1]), float(x_bits[2]), float(x_bits[3])], dtype=float)
        _, y4, _ = self.forward(x)
        return y4


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Обратное распространение: (X1 И X2) ИЛИ (X3 И X4)")
        self.geometry("740x500")

        self.dataset = build_dataset()
        self.model = MLP(input_size=5, hidden_neurons=3, lr=0.5)
        self.loss_history: List[float] = []

        self._make_widgets()

    def _make_widgets(self) -> None:
        frm = ttk.Frame(self)
        frm.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ctrl = ttk.Frame(frm)
        ctrl.pack(fill=tk.X)

        ttk.Label(ctrl, text="Эпохи:").pack(side=tk.LEFT)
        self.epochs_var = tk.StringVar(value="1000")
        ttk.Entry(ctrl, width=8, textvariable=self.epochs_var).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Label(ctrl, text="Шаг:").pack(side=tk.LEFT)
        self.lr_var = tk.StringVar(value=str(self.model.lr))
        ttk.Entry(ctrl, width=6, textvariable=self.lr_var).pack(side=tk.LEFT, padx=(4, 12))

        ttk.Button(ctrl, text="Обучить", command=self.on_train).pack(side=tk.LEFT)
        ttk.Button(ctrl, text="Сброс", command=self.on_reset).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(ctrl, text="График ошибки", command=self.on_plot).pack(side=tk.LEFT, padx=(8, 0))

        self.err_var = tk.StringVar(value="СКО: -")
        ttk.Label(ctrl, textvariable=self.err_var).pack(side=tk.LEFT, padx=16)

        ttk.Label(ctrl, text="Порог:").pack(side=tk.LEFT)
        self.eps_var = tk.StringVar(value="0.02")
        ttk.Entry(ctrl, width=6, textvariable=self.eps_var).pack(side=tk.LEFT, padx=(4, 0))

        weights = ttk.Frame(frm)
        weights.pack(fill=tk.X, pady=(10, 6))
        self.txt = tk.Text(weights, height=10)
        self.txt.pack(fill=tk.BOTH, expand=True)

        pred = ttk.LabelFrame(frm, text="Предсказание")
        pred.pack(fill=tk.X, pady=(10, 0))

        self.var_x = [tk.IntVar(value=0) for _ in range(4)]
        for i in range(4):
            ttk.Checkbutton(pred, text=f"X{i + 1}", variable=self.var_x[i]).pack(side=tk.LEFT, padx=6)
        ttk.Button(pred, text="Предсказать", command=self.on_predict).pack(side=tk.LEFT, padx=12)
        self.pred_var = tk.StringVar(value="y = ?")
        ttk.Label(pred, textvariable=self.pred_var).pack(side=tk.LEFT, padx=10)

        self._render_weights()

    def _render_weights(self) -> None:
        self.txt.delete("1.0", tk.END)
        self.txt.insert(tk.END, "Веса скрытого слоя (3x5)\n")
        self.txt.insert(tk.END, str(np.round(self.model.w_hidden, 4)) + "\n\n")
        self.txt.insert(tk.END, "Веса выходного нейрона (1x4) [смещение+3]\n")
        self.txt.insert(tk.END, str(np.round(self.model.w_out, 4)) + "\n")

    def on_train(self) -> None:
        try:
            epochs = int(self.epochs_var.get())
            lr = float(self.lr_var.get())
            eps = float(self.eps_var.get())
        except Exception:
            return
        self.model.lr = lr
        last_rmse = 0.0
        self.loss_history = []
        for _ in range(epochs):
            last_rmse = self.model.train_epoch(self.dataset)
            self.loss_history.append(last_rmse)
            if last_rmse <= eps:
                break
        self.err_var.set(f"СКО: {last_rmse:.6f}")
        self._render_weights()
        self.on_plot()

    def on_reset(self) -> None:
        self.model = MLP(input_size=5, hidden_neurons=3, lr=float(self.lr_var.get()))
        self.err_var.set("СКО: -")
        self.loss_history = []
        self._render_weights()

    def on_predict(self) -> None:
        bits = tuple(v.get() for v in self.var_x)
        y = self.model.predict(bits)
        self.pred_var.set(f"y = {y:.4f}  (bin: {1 if y >= 0.5 else 0})")

    def on_plot(self) -> None:
        if not self.loss_history:
            return
        plt.figure(figsize=(6, 3))
        plt.plot(self.loss_history, label="СКО")
        plt.xlabel("Эпоха")
        plt.ylabel("СКО")
        plt.title("Кривая обучения")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    App().mainloop()


