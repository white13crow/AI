import numpy as np


class HammingNetwork:
    def __init__(self, n, m, v=0.5):
        """
        Ініціалізація мережі Хеммінга

        Args:
            n (int): розмірність вхідного вектора
            m (int): кількість класів
            v (float): параметр гальмування (0 < v < 1/m)
        """
        self.n = n
        self.m = m
        self.v = v

        # Ініціалізація ваг першого шару
        self.W1 = None
        self.B1 = None

        # Ініціалізація ваг другого шару
        self.W2 = np.zeros((m, m))
        np.fill_diagonal(self.W2, 1.0)  # діагональ = 1
        for i in range(m):
            for j in range(m):
                if i != j:
                    self.W2[i, j] = -self.v

    def initialize_weights(self, patterns):
        """
        Ініціалізація ваг першого шару згідно з алгоритмом

        Args:
            patterns (list): список навчальних зразків
        """
        self.W1 = np.zeros((self.n, self.m))
        self.B1 = np.zeros(self.m)

        for k in range(self.m):
            for i in range(self.n):
                self.W1[i, k] = patterns[k][i] / 2
            self.B1[k] = self.n / 2

    def predict(self, x, max_iterations=100):
        """
        Класифікація вхідного вектора

        Args:
            x (list): вхідний вектор
            max_iterations (int): максимальна кількість ітерацій

        Returns:
            tuple: (переможний клас, виходи нейронів, кількість ітерацій)
        """
        if self.W1 is None:
            raise ValueError("Мережа не навчена! Спочатку викличте initialize_weights()")

        # Перетворення в numpy array
        x = np.array(x)

        # Крок 2: Обчислення станів нейронів першого шару
        y1 = np.zeros(self.m)
        for j in range(self.m):
            net_input = np.sum(self.W1[:, j] * x)
            y1[j] = net_input + self.B1[j]

        # Ініціалізація виходів другого шару
        y2_prev = y1.copy()
        y2_current = y1.copy()

        # Крок 3-4: Ітеративне оновлення
        iteration = 0
        changed = True

        while changed and iteration < max_iterations:
            y2_prev = y2_current.copy()

            # Обчислення нових станів нейронів другого шару
            for j in range(self.m):
                net_input = np.sum(self.W2[j, :] * y2_prev)
                # Передатна функція - поріг (ReLU)
                y2_current[j] = max(0, net_input)

            # Перевірка змін
            changed = not np.allclose(y2_current, y2_prev, atol=1e-6)
            iteration += 1

        # Знаходження переможного нейрона
        winner = np.argmax(y2_current)
        return winner, y2_current, iteration
