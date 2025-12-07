import os

from PIL import Image

import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt


class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))
        self.patterns = []  # Зберігаємо оригінальні патерни
        self.class_labels = []  # Мітки класів

    def train(self, patterns, labels):
        """Навчання мережі Хопфілда згідно з вимогами"""
        self.patterns = [p.copy() for p in patterns]
        self.class_labels = labels.copy()

        # Ініціалізація матриці ваг нулями
        self.weights = np.zeros((self.size, self.size))

        # Обчислення матриці ваг за формулою Хопфілда
        for pattern in patterns:
            pattern = pattern.reshape(-1, 1)  # Перетворюємо у вектор-стовпець
            self.weights += np.dot(pattern, pattern.T)

        # Обнулення діагональних елементів
        np.fill_diagonal(self.weights, 0)

        # Нормалізація (необов'язково, але покращує стабільність)
        if len(patterns) > 0:
            self.weights = self.weights / len(patterns)

    def predict(self, input_pattern, max_iterations=20):
        """Класифікація за методом Хопфілда з аналізом процесу"""
        if len(self.patterns) == 0:
            return -1, 0, input_pattern, []

        pattern = input_pattern.copy()
        iteration_history = []

        st.write("### Процес класифікації мережі Хопфілда")

        # Асинхронне оновлення
        for iteration in range(max_iterations):
            old_pattern = pattern.copy()
            changed_neurons = 0

            # Випадковий порядок оновлення нейронів
            update_order = np.random.permutation(self.size)

            for neuron_idx in update_order:
                # Обчислення суми вагових коефіцієнтів
                net_input = np.dot(self.weights[neuron_idx], pattern)

                # Оновлення стану нейрона
                new_state = 1 if net_input >= 0 else -1

                if new_state != pattern[neuron_idx]:
                    pattern[neuron_idx] = new_state
                    changed_neurons += 1

            # Збереження інформації про ітерацію
            iteration_info = {
                'iteration': iteration + 1,
                'pattern': pattern.copy(),
                'changed_neurons': changed_neurons,
                'stability': np.sum(pattern == old_pattern) / self.size
            }
            iteration_history.append(iteration_info)

            # Перевірка збіжності
            if changed_neurons == 0:
                st.success(f"**Збіжність досягнута на ітерації {iteration + 1}**")
                break

        # Порівняння з еталонними патернами
        best_similarity = -1
        best_index = -1

        for i, stored_pattern in enumerate(self.patterns):
            similarity = self._calculate_similarity(pattern, stored_pattern)
            if similarity > best_similarity:
                best_similarity = similarity
                best_index = i

        return best_index, best_similarity, pattern, iteration_history

    def _calculate_similarity(self, vec1, vec2):
        """Розрахунок схожості між векторами"""
        if len(vec1) != len(vec2):
            return 0

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0

        return dot_product / (norm1 * norm2)


def binarize_features(features, threshold=0):
    """Бінаризація ознакових векторів зі значеннями -1 та 1"""
    features = np.array(features)
    binary_features = np.where(features > threshold, 1, -1)
    return binary_features


def load_reference_images(folder_path="img"):
    """Завантаження зразкових зображень трьох класів"""
    reference_images = {
        'Квадрат': [],
        'Коло': [],
        'Ромб': []
    }

    if not os.path.exists(folder_path):
        st.error(f"Папка {folder_path} не знайдена!")
        return reference_images

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if 'квадрат' in filename.lower():
            class_name = 'Квадрат'
        elif 'коло' in filename.lower():
            class_name = 'Коло'
        elif 'ромб' in filename.lower():
            class_name = 'Ромб'
        else:
            continue

        try:
            image = Image.open(file_path)
            reference_images[class_name].append((filename, image))
        except Exception as e:
            st.warning(f"Не вдалося завантажити {filename}: {e}")

    return reference_images


def extract_absolute_features(pil_image, grid_size):
    """Обчислення абсолютних ознакових векторів"""
    try:
        # Стандартизація розміру
        image = pil_image.resize((150, 150))
        image_array = np.array(image.convert('L'))

        # Бінаризація зображення
        _, binary_image = cv2.threshold(image_array, 128, 255, cv2.THRESH_BINARY)
        binary_image = 255 - binary_image  # Інвертуємо

        # Розрахунок абсолютного вектора ознак
        rows, cols = map(int, grid_size.split('x'))
        img_height, img_width = binary_image.shape
        cell_height = img_height // rows
        cell_width = img_width // cols

        absolute_vector = []

        for i in range(rows):
            for j in range(cols):
                y_start = i * cell_height
                y_end = (i + 1) * cell_height if i < rows - 1 else img_height
                x_start = j * cell_width
                x_end = (j + 1) * cell_width if j < cols - 1 else img_width

                cell = binary_image[y_start:y_end, x_start:x_end]
                black_pixels = np.sum(cell > 128)
                absolute_vector.append(black_pixels)

        return absolute_vector, binary_image

    except Exception as e:
        st.error(f"Помилка при обчисленні ознак: {e}")
        return None, None


def display_matrix(matrix, title):
    """Відображення матриці у зручному форматі"""
    st.write(f"### {title}")

    # Відображення як DataFrame для кращої читабельності
    st.dataframe(matrix, use_container_width=True)

    # Візуалізація теплової карти
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar(cax)
    ax.set_title(title)
    st.pyplot(fig)


# Головний код
st.set_page_config(page_title="Система розпізнавання Хопфілда", layout="wide")
st.write("# Система розпізнавання на базі штучної нейронної мережі Хопфілда")

# Ініціалізація стану
if 'hopfield_net' not in st.session_state:
    st.session_state.hopfield_net = None
if 'patterns_data' not in st.session_state:
    st.session_state.patterns_data = {}

# Вибір параметрів
st.sidebar.write("## Налаштування")
grid_size = st.sidebar.selectbox("Розмір сітки:", ["3x3", "4x4", "5x5"], index=1)
rows, cols = map(int, grid_size.split('x'))
vector_size = rows * cols

# 1. ВВЕДЕННЯ ЗРАЗКОВИХ ЗОБРАЖЕНЬ
st.write("## 1. Введення зразкових зображень трьох класів")

reference_images = load_reference_images("img")

# Відображення зразкових зображень у графічних компонентах
for class_name, images in reference_images.items():
    st.write(f"### Клас: {class_name}")
    if images:
        # Відображення зображень кожного класу
        display_cols = st.columns(min(10, len(images)))
        for idx, (filename, image) in enumerate(images[:10]):
            with display_cols[idx]:
                st.image(image, caption=filename, use_column_width=True)

# 2. ОБЧИСЛЕННЯ ТА ВІДОБРАЖЕННЯ ОЗНАКОВИХ ВЕКТОРІВ
st.write("## 2. Обчислення абсолютних ознакових векторів")

if st.button("Обчислити ознакові вектори"):
    patterns_data = {}
    absolute_vectors = {}

    for class_name, images in reference_images.items():
        if images:
            # Використовуємо перше зображення кожного класу як еталон
            filename, image = images[0]

            # Обчислення абсолютного вектора ознак
            absolute_vector, binary_image = extract_absolute_features(image, grid_size)

            if absolute_vector is not None:
                patterns_data[class_name] = {
                    'filename': filename,
                    'absolute_vector': absolute_vector,
                    'image': image,
                    'binary_image': binary_image
                }
                absolute_vectors[class_name] = absolute_vector

                # Відображення результатів
                st.write(f"### {class_name} - {filename}")
                col1, col2 = st.columns(2)

                with col1:
                    st.image(image, caption="Оригінальне зображення", use_column_width=True)
                    st.image(binary_image, caption="Бінаризоване зображення", use_column_width=True)

                with col2:
                    st.write("**Абсолютний вектор ознак:**")
                    st.text_area("Значення:", " | ".join(map(str, absolute_vector)),
                                 height=150, key=f"abs_{class_name}")
                    st.write(f"**Довжина вектора:** {len(absolute_vector)}")
                    st.write(f"**Сума значень:** {sum(absolute_vector)}")

    st.session_state.patterns_data = patterns_data
    st.session_state.absolute_vectors = absolute_vectors

# 3. БІНАРИЗАЦІЯ ТА СТВОРЕННЯ МЕРЕЖІ ХОПФІЛДА
st.write("## 3. Бінаризація та створення мережі Хопфілда")

if 'patterns_data' in st.session_state and st.session_state.patterns_data:
    if st.button("Провести бінаризацію та навчити мережу"):
        patterns_data = st.session_state.patterns_data
        binary_patterns = []
        class_labels = []

        st.write("### Бінаризація еталонних векторів (значення -1 та 1)")

        for class_name, data in patterns_data.items():
            absolute_vector = data['absolute_vector']

            # Бінаризація з адаптивним порогом
            threshold = np.mean(absolute_vector) if len(absolute_vector) > 0 else 0
            binary_vector = binarize_features(absolute_vector, threshold)

            patterns_data[class_name]['binary_vector'] = binary_vector
            binary_patterns.append(binary_vector)
            class_labels.append(class_name)

            # Відображення бінаризованих векторів
            st.write(f"#### {class_name}")
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Абсолютний вектор:**")
                st.write(absolute_vector)

            with col2:
                st.write("**Бінарний вектор (-1/1):**")
                st.write(binary_vector)

        # 4. ОБЧИСЛЕННЯ МАТРИЦІ КОЕФІЦІЄНТІВ
        st.write("## 4. Обчислення матриці коефіцієнтів мережі Хопфілда")

        # Створення та навчання мережі
        hopfield_net = HopfieldNetwork(vector_size)
        hopfield_net.train(binary_patterns, class_labels)

        # Відображення матриці ваг
        display_matrix(hopfield_net.weights, "Матриця ваг мережі Хопфілда")

        st.session_state.hopfield_net = hopfield_net
        st.session_state.patterns_data = patterns_data

        st.success("Мережа Хопфілда успішно навчена!")

# 5. КЛАСИФІКАЦІЯ НЕВІДОМОГО ЗОБРАЖЕННЯ
st.write("## 5. Класифікація невідомого зображення")

if 'hopfield_net' in st.session_state and st.session_state.hopfield_net is not None:

    # Вибір джерела тестового зображення
    st.write("### Виберіть спосіб введення зображення:")
    input_method = st.radio(
        "Оберіть джерело зображення:",
        ["Вибрати з наявних зображень", "Завантажити своє зображення"],
        key="input_method"
    )

    test_image = None
    true_class = "Невідомий"

    if input_method == "Вибрати з наявних зображень":
        # Вибір зі списку наявних зображень
        test_images = []
        for class_name, images in reference_images.items():
            for idx, (filename, image) in enumerate(images):
                test_images.append((f"{class_name} - {filename}", image, class_name))

        if test_images:
            selected_test = st.selectbox("Оберіть тестове зображення:",
                                         [name for name, img, cls in test_images])

            # Знаходження обраного зображення
            for name, image, cls in test_images:
                if name == selected_test:
                    test_image = image
                    true_class = cls
                    break

    else:  # Завантажити своє зображення
        st.write("### Завантажте своє зображення для класифікації")
        uploaded_file = st.file_uploader(
            "Оберіть файл зображення",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            key="uploaded_test_image"
        )

        if uploaded_file is not None:
            try:
                # Завантаження та обробка зображення
                test_image = Image.open(uploaded_file)
                true_class = "Завантажене зображення"

                # Показ завантаженого зображення
                st.image(test_image, caption="Завантажене зображення", use_column_width=True)
                st.success("Зображення успішно завантажено!")

            except Exception as e:
                st.error(f"Помилка при завантаженні зображення: {e}")

    # Класифікація, якщо зображення вибрано/завантажено
    if test_image is not None:
        hopfield_net = st.session_state.hopfield_net

        st.write("### Вхідне невідоме зображення")
        col1, col2 = st.columns(2)

        with col1:
            st.image(test_image, caption=f"Джерело: {true_class}", use_column_width=True)

        if st.button("Провести класифікацію", key="classify_button"):
            # Обчислення ознакового вектора
            absolute_vector, binary_image = extract_absolute_features(test_image, grid_size)

            if absolute_vector is not None:
                # Бінаризація тестового вектора
                threshold = np.mean(absolute_vector) if len(absolute_vector) > 0 else 0
                test_binary_vector = binarize_features(absolute_vector, threshold)

                with col2:
                    st.image(binary_image, caption="Оброблене зображення", use_column_width=True)

                st.write("#### Ознаковий вектор невідомого зображення")
                col3, col4 = st.columns(2)

                with col3:
                    st.write("**Абсолютний вектор:**")
                    st.text_area("Абсолютні значення:", " | ".join(map(str, absolute_vector)),
                                 height=120, key="test_absolute")

                with col4:
                    st.write("**Бінарний вектор (-1/1):**")
                    st.text_area("Бінарні значення:", " | ".join(map(str, test_binary_vector)),
                                 height=120, key="test_binary")

                # 6. КЛАСИФІКАЦІЯ ЗА МЕТОДОМ ХОПФІЛДА
                st.write("## 6. Класифікація за методом Хопфілда")

                predicted_idx, similarity, final_pattern, iteration_history = hopfield_net.predict(test_binary_vector)

                if predicted_idx != -1:
                    predicted_class = hopfield_net.class_labels[predicted_idx]

                    # 7. АНАЛІЗ ПРОЦЕСУ КЛАСИФІКАЦІЇ
                    st.write("## 7. Аналіз процесу класифікації")

                    # Відображення ітераційного процесу
                    with st.expander("📊 Детальний аналіз ітераційного процесу"):
                        st.write("**Хід класифікації:**")
                        for iter_info in iteration_history:
                            st.write(f"**Ітерація {iter_info['iteration']}:**")
                            st.write(f"- Змінено нейронів: {iter_info['changed_neurons']}")
                            st.write(f"- Стабільність: {iter_info['stability']:.3f}")
                            if iter_info['changed_neurons'] == 0:
                                st.success("Досягнуто стабільний стан!")

                    # Результати класифікації
                    st.write("### Результати класифікації")

                    # Відображення у вигляді карточок
                    result_col1, result_col2, result_col3 = st.columns(3)

                    with result_col1:
                        st.metric("Справжній клас",
                                  true_class if true_class != "Завантажене зображення" else "Невідомий")

                    with result_col2:
                        st.metric("Розпізнаний клас", predicted_class)

                    with result_col3:
                        st.metric("Схожість", f"{similarity:.3f}")

                    # Порівняння з усіма еталонами
                    st.write("**Порівняння з еталонними патернами:**")
                    comparison_data = []

                    for i, class_name in enumerate(hopfield_net.class_labels):
                        pattern_similarity = hopfield_net._calculate_similarity(
                            final_pattern, hopfield_net.patterns[i]
                        )
                        comparison_data.append({
                            'Клас': class_name,
                            'Схожість': pattern_similarity,
                            'Результат': predicted_class == class_name
                        })

                    # Відображення таблиці порівняння
                    for i, data in enumerate(comparison_data):
                        if data['Результат']:
                            st.success(f"✅ **{data['Клас']}:** {data['Схожість']:.3f} **← РОЗПІЗНАНО**")
                        else:
                            st.info(f"📊 {data['Клас']}: {data['Схожість']:.3f}")

                    # Оцінка результатів
                    if true_class != "Завантажене зображення" and true_class != "Невідомий":
                        if predicted_class == true_class:
                            st.success("🎉 **Класифікація правильна!**")

                            if similarity > 0.9:
                                st.success("**Висока впевненість у результаті**")
                            elif similarity > 0.7:
                                st.info("**Середня впевненість у результаті**")
                            else:
                                st.warning("**Низька впевненість у результаті**")
                        else:
                            st.error("❌ **Класифікація неправильна!**")

                            # Аналіз причин помилки
                            st.write("### Аналіз причин помилки")
                            st.write("**Можливі причини:**")
                            st.write("1. Схожість між класами занадто висока")
                            st.write("2. Недостатньо відмінні ознаки")
                            st.write("3. Проблеми з якістю зображення")
                            st.write("4. Неправильний вибір розміру сітки")
                    else:
                        # Для завантажених зображень
                        st.info("🔍 **Результат класифікації завантаженого зображення**")
                        if similarity > 0.8:
                            st.success(f"Висока ймовірність того, що це **{predicted_class}**")
                        elif similarity > 0.6:
                            st.info(f"Середня ймовірність того, що це **{predicted_class}**")
                        else:
                            st.warning(
                                f"Низька ймовірність. Розпізнано як **{predicted_class}**, але результат ненадійний")

else:
    st.info("ℹ️ Спочатку навчіть мережу Хопфілда на еталонних зображеннях")



plt.style.use('default')