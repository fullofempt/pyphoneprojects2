import os
import re
import random
import shutil
import tempfile
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('TkAgg')

TRY_PYMORPHY = True
try:
    import pymorphy2  # type: ignore
except Exception:
    TRY_PYMORPHY = False


RUS_STOPWORDS = set(
    [
        'и','в','во','не','что','он','на','я','с','со','как','а','то','все','она','так','его','но','да','ты','к','у','же','вы','за','бы','по','только','ее','мне','было','вот','от','меня','еще','нет','о','из','ему','теперь','когда','даже','ну','вдруг','ли','если','уже','или','ни','быть','был','него','до','вас','нибудь','опять','уж','вам','ведь','там','потом','себя','ничего','ей','может','они','тут','где','есть','надо','ней','для','мы','тебя','их','чем','была','сам','чтоб','без','будто','чего','раз','тоже','себе','под','будет','ж','тогда','кто','этот','того','потому','этого','какой','совсем','ним','здесь','этом','один','почти','мой','тем','чтобы','нее','сейчас','были','куда','зачем','всех','никогда','можно','при','наконец','два','об','другой','хоть','после','над','больше','тот','через','эти','нас','про','всего','них','какая','много','разве','три','эту','моя','впрочем','хорошо','свою','этой','перед','иногда','лучше','чуть','том','нельзя','такой','им','более','всегда','конечно','всю','между'
    ]
)


TOKEN_RE = re.compile(r"[а-яёa-z0-9-]+", re.IGNORECASE)


def read_title_body(path: str) -> Tuple[str, str]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(path, 'r', encoding='cp1251', errors='ignore') as f:
            text = f.read()
    parts = text.splitlines()
    if not parts:
        return '', ''
    title = parts[0]
    body = '\n'.join(parts[1:])
    return title, body


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def lemmatize(words: List[str]) -> List[str]:
    if not TRY_PYMORPHY:
        return words
    morph = pymorphy2.MorphAnalyzer()
    lemmas = []
    for w in words:
        p = morph.parse(w)
        if p:
            lemmas.append(p[0].normal_form)
        else:
            lemmas.append(w)
    return lemmas


def preprocess(text: str) -> List[str]:
    tokens = tokenize(text)
    tokens = [t for t in tokens if t not in RUS_STOPWORDS and not t.isdigit()]
    tokens = lemmatize(tokens)
    return tokens


def build_freqs_for_dir(topic_dir: str) -> Tuple[Counter, Counter]:
    title_counter: Counter = Counter()
    body_counter: Counter = Counter()
    for root, _, files in os.walk(topic_dir):
        for name in files:
            if not name.lower().endswith(('.txt',)):
                continue
            title, body = read_title_body(os.path.join(root, name))
            title_tokens = preprocess(title)
            body_tokens = preprocess(body)
            title_counter.update(title_tokens)
            body_counter.update(body_tokens)
    return title_counter, body_counter


def save_dict(path: str, counter: Counter) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        for word in sorted(counter):
            f.write(f"{word} {counter[word]}\n")


def load_unique_words_from_text(title: str, body: str) -> Tuple[List[str], List[str]]:
    t = set(preprocess(title))
    b = set(preprocess(body))
    return sorted(t), sorted(b)


def compute_kb(title_words: List[str], body_words: List[str],
               dict_title: Dict[str, int], dict_body: Dict[str, int], k: float = 2.0) -> float:
    kb = 0.0
    for w in title_words:
        kb += k * dict_title.get(w, 0)
    for w in body_words:
        kb += dict_body.get(w, 0)
    return kb


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title('Частотный анализ: словари и классификация')
        self.geometry('900x620')
        # авто-генерация набора данных в папке проекта (laba4)
        self._autogen_base: str = ''
        self._autogen_train: str = ''
        self._autogen_test: str = ''
        self._autogen_created: bool = False
        self._autogen_generate()
        self._make_widgets()
        # корректное удаление временных файлов при закрытии окна
        self.protocol('WM_DELETE_WINDOW', self.on_close)

    def _make_widgets(self) -> None:
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=8, pady=8)

        ttk.Label(top, text='Категории (3 папки):').pack(side=tk.LEFT)
        self.train_dirs: List[str] = ['','','']
        for i in range(3):
            ttk.Button(top, text=f'Выбрать тему {i+1}', command=lambda i=i: self.pick_train_dir(i)).pack(side=tk.LEFT, padx=4)

        ttk.Button(top, text='Построить словари', command=self.build_dicts).pack(side=tk.LEFT, padx=12)
        ttk.Button(top, text='Выбрать тест', command=self.pick_test_dir).pack(side=tk.LEFT, padx=12)
        ttk.Label(top, text='K (заголовок):').pack(side=tk.LEFT, padx=(12,0))
        self.k_var = tk.StringVar(value='2.0')
        ttk.Entry(top, width=6, textvariable=self.k_var).pack(side=tk.LEFT)
        ttk.Button(top, text='Классифицировать тест', command=self.classify_tests).pack(side=tk.LEFT, padx=12)

        body = ttk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.txt = tk.Text(body)
        self.txt.pack(fill=tk.BOTH, expand=True)

        self.status = tk.StringVar(value='Готово')
        ttk.Label(self, textvariable=self.status).pack(fill=tk.X, padx=8, pady=(0,8))

        self.test_dir: str = self.test_dir if hasattr(self, 'test_dir') else ''
        self.dicts: List[Tuple[Dict[str,int], Dict[str,int]]] = []

    def log(self, msg: str) -> None:
        self.txt.insert(tk.END, msg + '\n')
        self.txt.see(tk.END)

    def pick_train_dir(self, idx: int) -> None:
        path = filedialog.askdirectory(title=f'Выберите папку темы {idx+1}')
        if path:
            self.train_dirs[idx] = path
            self.log(f'Тема {idx+1}: {path}')

    def pick_test_dir(self) -> None:
        path = filedialog.askdirectory(title='Выберите папку с тестовыми текстами (3 папки по темам или просто txt-файлы)')
        if path:
            self.test_dir = path
            self.log(f'Тестовая папка: {path}')

    def _autogen_generate(self) -> None:
        # Создаем структуру папок с обучением и тестом внутри laba4
        base = os.path.abspath(os.path.dirname(__file__))
        train_root = os.path.join(base, 'train')
        test_root = os.path.join(base, 'test')
        os.makedirs(train_root, exist_ok=True)
        os.makedirs(test_root, exist_ok=True)

        categories = ['sport', 'technology', 'politics']
        for cat in categories:
            os.makedirs(os.path.join(train_root, cat), exist_ok=True)
            os.makedirs(os.path.join(test_root, cat), exist_ok=True)

        sport_keywords = ['футбол','хоккей','матч','гол','команда','победа','чемпионат','олимпиада','игрок','тренер','счет','турнир','медаль','спортсмен','стадион']
        tech_keywords = ['искусственный интеллект','программирование','гаджет','смартфон','компьютер','технология','инновация','приложение','софт','аппаратное обеспечение','данные','алгоритм','интернет','цифровой','автоматизация']
        politics_keywords = ['президент','правительство','выборы','закон','парламент','министр','международный','договор','политика','страна','лидер','переговоры','экономика','безопасность','дипломатия']

        def generate_text(category: str) -> Tuple[str, str]:
            if category == 'sport':
                titles = [
                    'Победа в решающем матче чемпионата',
                    'Новый рекорд установлен на Олимпиаде',
                    'Трансфер звездного игрока завершен',
                    'Сенсационная победа аутсайдера',
                    'Подготовка к мировому первенству',
                ]
                contents = [
                    'В решающем матче сезона команда одержала уверенную победу со счетом 3:0. Игроки показали великолепную игру и заслужили звание чемпионов.',
                    'На международных соревнованиях был установлен новый мировой рекорд. Спортсмен показал выдающийся результат.',
                    'Клуб объявил о переходе известного футболиста. Сумма трансфера составила рекордные для лиги деньги.',
                    'Неожиданная победа аутсайдера стала главной сенсацией турнира. Команда показала характер и волю к победе.',
                    'Национальная сборная начала интенсивную подготовку к предстоящему чемпионату мира. Тренерский штаб разработал новую тактику.',
                ]
                kw = random.sample(sport_keywords, 3)
            elif category == 'technology':
                titles = [
                    'Прорыв в области искусственного интеллекта',
                    'Новый смартфон установил рекорды продаж',
                    'Инновационная технология изменит будущее',
                    'Кибербезопасность становится приоритетом',
                    'Цифровая трансформация бизнеса',
                ]
                contents = [
                    'Ученые представили новую модель искусственного интеллекта, способную решать сложные задачи. Технология открывает новые возможности.',
                    'Новая модель смартфона побила все рекорды по продажам в первый день. Пользователи высоко оценили инновационные функции.',
                    'Разработана революционная технология, которая может кардинально изменить различные отрасли промышленности.',
                    'В связи с участившимися кибератаками компании инвестируют в системы безопасности. Защита данных становится критически важной.',
                    'Бизнес активно внедряет цифровые технологии для повышения эффективности. Цифровая трансформация становится необходимостью.',
                ]
                kw = random.sample(tech_keywords, 3)
            else:
                titles = [
                    'Важные международные переговоры завершены',
                    'Новый закон принят парламентом',
                    'Выборы прошли в демократической атмосфере',
                    'Экономическое сотрудничество укрепляется',
                    'Встреча на высшем уровне состоялась',
                ]
                contents = [
                    'Завершились важные международные переговоры между лидерами стран. Стороны договорились о сотрудничестве.',
                    'Парламент принял новый закон, направленный на улучшение социальной защиты граждан. Закон вступит в силу с следующего месяца.',
                    'Выборы прошли в спокойной обстановке при высокой явке избирателей. Международные наблюдатели подтвердили честность процесса.',
                    'Страны договорились об укреплении экономического сотрудничества. Подписаны важные торговые соглашения.',
                    'Состоялась встреча на высшем уровне, в ходе которой обсуждались актуальные международные вопросы.',
                ]
                kw = random.sample(politics_keywords, 3)
            title = random.choice(titles)
            content = random.choice(contents) + ' ' + ' '.join(kw) + '.'
            return title, content

        # генерируем файлы: train 25 шт., test 3 шт. для каждой темы
        for cat in categories:
            for i in range(1, 26):
                title, content = generate_text(cat)
                path = os.path.join(train_root, cat, f'{cat}{i}.txt')
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(f'{title}\n\n{content}')
            for i in range(1, 4):
                title, content = generate_text(cat)
                path = os.path.join(test_root, cat, f'{cat}_test{i}.txt')
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(f'{title}\n\n{content}')

        # заполняем пути по умолчанию для GUI
        self.train_dirs = [os.path.join(train_root, c) for c in categories]
        self.test_dir = test_root
        self._autogen_base = base
        self._autogen_train = train_root
        self._autogen_test = test_root
        self._autogen_created = True
        # вывод в лог будет настроен после создания виджетов

    def build_dicts(self) -> None:
        if not all(self.train_dirs):
            messagebox.showerror('Ошибка', 'Укажите все три папки тем.')
            return
        self.dicts.clear()
        out_dir = filedialog.askdirectory(title='Куда сохранить 6 словарей?')
        if not out_dir:
            return
        for i, d in enumerate(self.train_dirs, start=1):
            self.status.set(f'Обработка темы {i}...')
            self.update_idletasks()
            title_c, body_c = build_freqs_for_dir(d)
            self.dicts.append((dict(title_c), dict(body_c)))
            save_dict(os.path.join(out_dir, f'theme{i}_title.txt'), title_c)
            save_dict(os.path.join(out_dir, f'theme{i}_body.txt'), body_c)
            self.log(f'Тема {i}: слов в заголовках={len(title_c)}, слов в тексте={len(body_c)}')
        self.status.set('Словари готовы и сохранены')

    def iter_test_files(self) -> List[str]:
        paths: List[str] = []
        for root, _, files in os.walk(self.test_dir):
            for name in files:
                if name.lower().endswith('.txt'):
                    paths.append(os.path.join(root, name))
        return sorted(paths)

    def classify_tests(self) -> None:
        if not self.dicts:
            messagebox.showerror('Ошибка', 'Сначала постройте словари по обучающим данным.')
            return
        if not self.test_dir:
            messagebox.showerror('Ошибка', 'Укажите папку с тестовыми текстами.')
            return
        try:
            k = float(self.k_var.get())
        except Exception:
            k = 2.0

        self.log('Классификация тестовых файлов...')
        self.log('Формат: имя файла -> KB1, KB2, KB3; предсказание')
        for path in self.iter_test_files():
            title, body = read_title_body(path)
            title_words, body_words = load_unique_words_from_text(title, body)
            kbs: List[float] = []
            for (dt, db) in self.dicts:
                kbs.append(compute_kb(title_words, body_words, dt, db, k=k))
            best = int(max(range(3), key=lambda i: kbs[i])) + 1
            self.log(f'{os.path.basename(path)} -> {kbs[0]:.1f}, {kbs[1]:.1f}, {kbs[2]:.1f}; тема {best}')
        self.status.set('Классификация завершена')

    def on_close(self) -> None:
        # удаляем сгенерированные папки train/test в laba4
        try:
            if self._autogen_created:
                for p in (self._autogen_train, self._autogen_test):
                    if p and os.path.isdir(p):
                        shutil.rmtree(p, ignore_errors=True)
        finally:
            self.destroy()


if __name__ == '__main__':
    App().mainloop()


