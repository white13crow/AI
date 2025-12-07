import io
import os

from PIL import Image
import streamlit as st
import numpy as np
import cv2


class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.uniform(-0.5, 0.5, input_size + 1)  # +1 для bias
        self.learning_rate = learning_rate
        self.errors = []

    def predict(self, inputs):
        inputs_with_bias = np.insert(inputs, 0, 1)
        summation = np.dot(inputs_with_bias, self.weights)
        return 1 if summation >= 0 else 0

    def train(self, inputs, target):
        inputs_with_bias = np.insert(inputs, 0, 1)
        prediction = self.predict(inputs)
        error = target - prediction

        if error != 0:
            self.weights += self.learning_rate * error * inputs_with_bias

        self.errors.append(error)
        return error


class MultiClassPerceptronSystem:
    def __init__(self):
        self.perceptrons = []
        self.training_data = {'Квадрат': [], 'Коло': [], 'Ромб': []}
        self.trained = False
        self.feature_size = 0

    def add_training_data(self, class_name, features):
        self.training_data[class_name].append(features)

    def train_perceptrons(self, learning_rate=0.1):
        # Знаходимо мінімальну кількість зразків серед класів
        min_samples = min(len(self.training_data[cls]) for cls in self.training_data)

        if min_samples == 0:
            raise ValueError("Усі класи повинні мати принаймні один зразок")

        # Визначаємо розмірність ознак
        self.feature_size = len(self.training_data['Квадрат'][0])

        # Створюємо три перцептрони (один для кожного класу)
        self.perceptrons = [Perceptron(self.feature_size, learning_rate) for _ in range(3)]

        # Підготовка даних для навчання
        training_data = []
        for i, (class_name, samples) in enumerate(self.training_data.items()):
            for sample in samples[:min_samples]:  # Використовуємо однакову кількість зразків
                training_data.append((sample, i, class_name))

        # Навчання кожного перцептрона
        convergence = [False, False, False]
        epochs = 100

        for epoch in range(epochs):
            for perceptron_idx in range(3):
                if convergence[perceptron_idx]:
                    continue

                total_error = 0
                for features, target_class, _ in training_data:
                    # Для кожного перцептрона: 1 якщо його клас, 0 якщо інший
                    target = 1 if target_class == perceptron_idx else 0
                    error = self.perceptrons[perceptron_idx].train(features, target)
                    total_error += abs(error)

                if total_error == 0:
                    convergence[perceptron_idx] = True

        self.trained = True
        return all(convergence)

    def predict(self, features):
        if not self.trained:
            raise ValueError("Система не навчена")

        scores = []
        for i, perceptron in enumerate(self.perceptrons):
            inputs_with_bias = np.insert(features, 0, 1)
            score = np.dot(inputs_with_bias, perceptron.weights)
            scores.append(score)

        class_names = ['Квадрат', 'Коло', 'Ромб']
        winning_class = class_names[np.argmax(scores)]

        return winning_class, scores


def extract_features(image_array, grid_size):
    """Видобуває абсолютний та нормований вектори ознак"""
    _, binary_image = cv2.threshold(image_array, 128, 255, cv2.THRESH_BINARY)

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
            black_pixels = np.sum(cell == 0)
            absolute_vector.append(black_pixels)

    total_sum = sum(absolute_vector)
    if total_sum > 0:
        normalized_vector = [val / total_sum for val in absolute_vector]
    else:
        normalized_vector = [0 for _ in absolute_vector]

    return absolute_vector, normalized_vector, binary_image


def create_grid_image(binary_image, grid_size):
    """Створює зображення з накладеною сіткою"""
    grid_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    rows, cols = map(int, grid_size.split('x'))
    img_height, img_width = binary_image.shape
    cell_height = img_height // rows
    cell_width = img_width // cols

    for i in range(rows):
        for j in range(cols):
            y_start = i * cell_height
            y_end = (i + 1) * cell_height if i < rows - 1 else img_height
            x_start = j * cell_width
            x_end = (j + 1) * cell_width if j < cols - 1 else img_width

            if j > 0:
                cv2.line(grid_image, (x_start, 0), (x_start, img_height), (0, 0, 255), 2)
            if i > 0:
                cv2.line(grid_image, (0, y_start), (img_width, y_start), (0, 0, 255), 2)

    cv2.rectangle(grid_image, (0, 0), (img_width - 1, img_height - 1), (0, 255, 0), 2)
    return grid_image


def load_images_from_folder(folder_path, class_name, max_images=10):
    """Завантажує зображення з папки"""
    images = []
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                # Перевіряємо, чи містить назва файлу ім'я класу
                if class_name.lower() in filename.lower():
                    try:
                        img_path = os.path.join(folder_path, filename)
                        pil_image = Image.open(img_path)
                        images.append((pil_image, filename))
                        if len(images) >= max_images:
                            break
                    except Exception as e:
                        st.error(f"Помилка завантаження {filename}: {str(e)}")
    return images


# Основна програма
st.set_page_config(page_title="Перцептронна система розпізнавання", layout="wide")
st.write("# Система розпізнавання перцептронного типу")

# Ініціалізація системи
if 'perceptron_system' not in st.session_state:
    st.session_state.perceptron_system = MultiClassPerceptronSystem()

# Вибір параметрів
col1, col2, col3 = st.columns(3)
with col1:
    grid_size = st.selectbox("Розмір сітки:", ["3x3", "4x4", "5x5", "6x6", "4x5", "5x4"], index=2)
with col2:
    learning_rate = st.slider("Швидкість навчання:", 0.01, 1.0, 0.1, 0.01)
with col3:
    # Автоматично встановлюємо шлях до папки img
    folder_path = st.text_input("Шлях до папки з зображеннями:", value="./img")

# Вкладки для різних функцій
tab1, tab2, tab3, tab4 = st.tabs(["Завантаження даних", "Перегляд ознак", "Навчання", "Тестування"])

with tab1:
    st.header("Завантаження навчальних даних")

    # Перевірка існування папки
    if not os.path.exists(folder_path):
        st.error(f"Папка '{folder_path}' не знайдена! Перевірте шлях.")
    else:
        st.success(f"Папка '{folder_path}' знайдена. Файли в папці:")
        try:
            files = os.listdir(folder_path)
            image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            st.write(f"Знайдено {len(image_files)} зображень:")
            for file in image_files:
                st.write(f"- {file}")
        except Exception as e:
            st.error(f"Помилка читання папки: {str(e)}")

    if st.button("Завантажити зображення з папки"):
        with st.spinner("Завантаження зображень..."):
            # Завантажуємо зображення для кожного класу
            classes = ['Квадрат', 'Коло', 'Ромб']

            total_loaded = 0
            for class_name in classes:
                images = load_images_from_folder(folder_path, class_name)

                if images:
                    st.success(f"Знайдено {len(images)} зображень для класу '{class_name}'")

                    # Обробка кожного зображення
                    for pil_image, filename in images:
                        try:
                            image_array = np.array(pil_image.convert('L'))
                            absolute_vector, normalized_vector, binary_image = extract_features(image_array, grid_size)

                            # Додаємо до навчальних даних
                            st.session_state.perceptron_system.add_training_data(class_name, normalized_vector)
                            total_loaded += 1

                        except Exception as e:
                            st.error(f"Помилка обробки {filename}: {str(e)}")
                else:
                    st.warning(f"Не знайдено зображень для класу '{class_name}' в папці {folder_path}")

            st.success(f"Завантажено {total_loaded} зображень для навчання")

with tab2:
    st.header("Перегляд векторів ознак")

    if st.button("Показати всі вектори ознак"):
        for class_name in ['Квадрат', 'Коло', 'Ромб']:
            if st.session_state.perceptron_system.training_data[class_name]:
                st.subheader(f"🎯 Клас: {class_name}")
                st.write(f"**Кількість зразків:** {len(st.session_state.perceptron_system.training_data[class_name])}")

                # Завантажуємо всі зображення цього класу
                all_images = load_images_from_folder(folder_path, class_name, 100)  # Беремо всі

                if all_images:
                    # Відображаємо всі зображення класу
                    st.write("### Всі зображення класу:")

                    # Створюємо сітку для відображення зображень
                    num_images = len(all_images)
                    cols_per_row = 3  # Кількість зображень в рядку

                    for i in range(0, num_images, cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j in range(cols_per_row):
                            if i + j < num_images:
                                pil_image, filename = all_images[i + j]
                                with cols[j]:
                                    st.image(pil_image, caption=filename, use_column_width=True)

                    st.write("---")

                    # Відображаємо вектори ознак для кожного зображення
                    st.write("### Вектори ознак для кожного зображення:")

                    for idx, (pil_image, filename) in enumerate(all_images, 1):
                        st.write(f"#### Зображення {idx}: {filename}")

                        # Обробка зображення для отримання ознак
                        image_array = np.array(pil_image.convert('L'))
                        absolute_vector, normalized_vector, binary_image = extract_features(image_array, grid_size)
                        grid_image = create_grid_image(binary_image, grid_size)

                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(pil_image, caption="Оригінальне зображення", use_column_width=True)
                        with col2:
                            st.image(grid_image, caption=f"Сегментація ({grid_size})", use_column_width=True)

                        # Відображаємо вектори
                        col3, col4 = st.columns(2)
                        with col3:
                            st.write("**Абсолютний вектор ознак:**")
                            st.text_area(f"Абсолютні значення {idx}",
                                         "; ".join([f"{val}" for val in absolute_vector]),
                                         height=100, key=f"abs_{class_name}_{idx}")

                        with col4:
                            st.write("**Нормований вектор ознак:**")
                            st.text_area(f"Нормовані значення {idx}",
                                         "; ".join([f"{val:.6f}" for val in normalized_vector]),
                                         height=100, key=f"norm_{class_name}_{idx}")

                        st.write("---")

                else:
                    st.warning(f"Не вдалося завантажити зображення для класу '{class_name}'")
            else:
                st.warning(f"Немає даних для класу '{class_name}'")

        # Загальна статистика
        st.subheader("📊 Загальна статистика")
        total_samples = sum(len(st.session_state.perceptron_system.training_data[cls])
                            for cls in ['Квадрат', 'Коло', 'Ромб'])
        st.write(f"**Всього зразків у системі:** {total_samples}")

        for class_name in ['Квадрат', 'Коло', 'Ромб']:
            count = len(st.session_state.perceptron_system.training_data[class_name])
            st.write(f"- {class_name}: {count} зразків")

with tab3:
    st.header("Навчання перцептронів")

    # Статистика даних
    st.subheader("Статистика навчальних даних")
    total_samples = 0
    for class_name in ['Квадрат', 'Коло', 'Ромб']:
        count = len(st.session_state.perceptron_system.training_data[class_name])
        total_samples += count
        st.write(f"{class_name}: {count} зразків")

    st.write(f"**Всього зразків:** {total_samples}")

    if total_samples == 0:
        st.error("Немає даних для навчання. Спочатку завантажте зображення.")
    else:
        if st.button("Почати навчання"):
            try:
                with st.spinner("Навчання..."):
                    success = st.session_state.perceptron_system.train_perceptrons(learning_rate)

                if success:
                    st.success("Навчання успішно завершено!")

                    # Показуємо ваги перцептронів
                    st.subheader("Ваги перцептронів")
                    class_names = ['Квадрат', 'Коло', 'Ромб']

                    for i, (perceptron, class_name) in enumerate(
                            zip(st.session_state.perceptron_system.perceptrons, class_names)):
                        st.write(f"**{class_name} перцептрон:**")
                        st.write(f"Bias (w₀): {perceptron.weights[0]:.4f}")
                        for j, weight in enumerate(perceptron.weights[1:], 1):
                            st.write(f"w{j}: {weight:.4f}")
                        st.write("---")



            except Exception as e:
                st.error(f"Помилка навчання: {str(e)}")

with tab4:
    st.header("Тестування системи")

    if not st.session_state.perceptron_system.trained:
        st.error("Спочатку проведіть навчання перцептронів!")
    else:
        # Спосіб 1: Використання існуючих зображень для тестування
        st.subheader("Тестування на існуючих зображеннях")
        test_class = st.selectbox("Оберіть клас для тестування:", ['Квадрат', 'Коло', 'Ромб'])

        # Знаходимо доступні зображення для обраного класу
        available_images = load_images_from_folder(folder_path, test_class, 100)  # Беремо багато
        if available_images:
            image_options = [f"{i + 1}: {filename}" for i, (_, filename) in enumerate(available_images)]
            selected_image = st.selectbox("Оберіть зображення:", image_options)
            test_image_idx = image_options.index(selected_image) + 1
        else:
            st.error(f"Не знайдено зображень для класу {test_class}")
            test_image_idx = 1

        if st.button("Провести тестування") and available_images:
            if len(available_images) >= test_image_idx:
                pil_image, filename = available_images[test_image_idx - 1]

                try:
                    image_array = np.array(pil_image.convert('L'))
                    absolute_vector, normalized_vector, binary_image = extract_features(image_array, grid_size)
                    grid_image = create_grid_image(binary_image, grid_size)

                    # Класифікація
                    predicted_class, scores = st.session_state.perceptron_system.predict(normalized_vector)

                    # Відображення результатів
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(pil_image, caption=f"Тестове зображення: {filename}", use_column_width=True)
                    with col2:
                        st.image(grid_image, caption=f"Сегментація ({grid_size})", use_column_width=True)

                    st.write("### Результат класифікації:")

                    # Визначаємо правильність класифікації
                    is_correct = predicted_class == test_class

                    if is_correct:
                        st.success(f"**✓ Правильно!** Система визначила як: **{predicted_class}**")
                    else:
                        st.error(f"**✗ Помилка!** Справжній клас: {test_class}, Система визначила: {predicted_class}")

                    st.write("**Бали перцептронів:**")
                    class_names = ['Квадрат', 'Коло', 'Ромб']
                    for cls, score in zip(class_names, scores):
                        emphasis = "**" if cls == predicted_class else ""
                        st.write(f"- {emphasis}{cls}: {score:.4f}{emphasis}")

                    st.write("**Абсолютний вектор ознак:**", absolute_vector)
                    st.write("**Нормований вектор ознак:**", [f"{val:.6f}" for val in normalized_vector])

                except Exception as e:
                    st.error(f"Помилка обробки зображення: {str(e)}")

        # Спосіб 2: Завантаження тестового зображення
        st.subheader("Тестування на новому зображенні")
        test_file = st.file_uploader("Оберіть тестове зображення",
                                     type=["bmp", ".png", ".jpg", ".jpeg"],
                                     key="test_uploader")

        if test_file:
            try:
                image_bytes = test_file.read()
                pil_image = Image.open(io.BytesIO(image_bytes))
                image_array = np.array(pil_image.convert('L'))

                absolute_vector, normalized_vector, binary_image = extract_features(image_array, grid_size)
                grid_image = create_grid_image(binary_image, grid_size)

                # Класифікація
                predicted_class, scores = st.session_state.perceptron_system.predict(normalized_vector)

                # Відображення результатів
                col1, col2 = st.columns(2)
                with col1:
                    st.image(pil_image, caption="Завантажене тестове зображення", use_column_width=True)
                with col2:
                    st.image(grid_image, caption=f"Сегментація ({grid_size})", use_column_width=True)

                st.write("### Результат класифікації:")
                st.success(f"**Система визначила зображення як: {predicted_class}**")

                st.write("**Бали перцептронів:**")
                class_names = ['Квадрат', 'Коло', 'Ромб']
                for cls, score in zip(class_names, scores):
                    emphasis = "**" if cls == predicted_class else ""
                    st.write(f"- {emphasis}{cls}: {score:.4f}{emphasis}")

                st.write("**Абсолютний вектор ознак:**", absolute_vector)
                st.write("**Нормований вектор ознак:**", [f"{val:.6f}" for val in normalized_vector])

            except Exception as e:
                st.error(f"Помилка обробки тестового зображення: {str(e)}")



# Статус системи
st.sidebar.header("Статус системи")
if st.session_state.perceptron_system.trained:
    st.sidebar.success("✅ Система навчена")
    st.sidebar.write(f"Розмірність ознак: {st.session_state.perceptron_system.feature_size}")
else:
    st.sidebar.warning("⏳ Система не навчена")

for class_name in ['Квадрат', 'Коло', 'Ромб']:
    count = len(st.session_state.perceptron_system.training_data[class_name])
    status = "✅" if count > 0 else "❌"
    st.sidebar.write(f"{status} {class_name}: {count} зразків")