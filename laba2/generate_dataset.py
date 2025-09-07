import pandas as pd
import numpy as np

# Центры кластеров для каждого из 3-х классов
centers = {
    'Class1': {'x1': 570, 'x2': 500},
    'Class2': {'x1': 300, 'x2': 300},
    'Class3': {'x1': 300, 'x2': 600}
}

# Разброс точек вокруг центров
spreads = {
    'Class1': 150,  # Было 300*(RAND()-0.5) -> std ~300/3.46 ~86.6, возьмем 150 для нормального распределения
    'Class2': 200,  # Было 400*(RAND()-0.5) -> std ~115.5, возьмем 200
    'Class3': 150   # Аналогично классу 1
}

np.random.seed(42) # Для воспроизводимости результатов
data = []

for class_label, center in centers.items():
    class_num = int(class_label[-1]) # Извлекаем цифру 1, 2 или 3 из имени 'Class1'
    for _ in range(100):
        x1 = np.random.normal(center['x1'], spreads[class_label])
        x2 = np.random.normal(center['x2'], spreads[class_label])
        data.append([x1, x2, class_num])

# Создаем DataFrame и сохраняем в CSV
df = pd.DataFrame(data, columns=['X1', 'X2', 'Class'])
df.to_csv('knn_dataset.csv', index=False, encoding='utf-8')
print("Датасет 'knn_dataset.csv' успешно сгенерирован!")