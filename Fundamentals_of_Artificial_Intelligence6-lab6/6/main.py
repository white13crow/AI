import io
import os
import re

from PIL import Image
import streamlit as st
import numpy as np
import cv2
from hamming_network import HammingNetwork
from image_processor import process_image_to_vector, create_grid_image, improve_image_processing

st.set_page_config(page_title="Мережа Хеммінга для аналізу зображень", layout="wide")
st.write("# Мережа Хеммінга для класифікації геометричних фігур")

# Інтерфейс Streamlit
st.sidebar.header("Налаштування мережі Хеммінга")

grid_size = st.sidebar.selectbox("Розмір сітки:", ["3x3", "4x4", "5x5", "6x6", "4x5", "5x4"], index=2)
v_param = st.sidebar.slider("Параметр гальмування (v):", 0.001, 0.1, 0.01, 0.001)
max_iter = st.sidebar.slider("Максимальна кількість ітерацій:", 10, 200, 50)

# Додаткові налаштування обробки зображень
st.sidebar.header("Налаштування обробки зображень")
threshold_value = st.sidebar.slider("Поріг бінаризації:", 1, 255, 128, 1)
use_improved_processing = st.sidebar.checkbox("Використовувати покращену обробку", value=True)


# Функція для отримання та групування зображень з папки img
def get_images_grouped_by_class():
    img_folder = "../img"
    if not os.path.exists(img_folder):
        return {}

    image_files = [f for f in os.listdir(img_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    # Групування за класами
    classes = {}

    for img_file in image_files:
        # Визначення класу з імені файлу
        if 'квадрат' in img_file.lower():
            class_name = 'квадрат'
        elif 'коло' in img_file.lower() or 'круг' in img_file.lower():
            class_name = 'коло'
        elif 'трикутник' in img_file.lower() or 'треугольник' in img_file.lower():
            class_name = 'трикутник'
        elif 'ромб' in img_file.lower():
            class_name = 'ромб'
        else:
            class_name = 'інше'

        if class_name not in classes:
            classes[class_name] = []

        classes[class_name].append(img_file)

    return classes, img_folder


# Отримання згрупованих зображень
classes, img_folder = get_images_grouped_by_class()

# Вибір джерела тестового зображення
st.sidebar.header("Тестове зображення")
test_image_source = st.sidebar.radio("Джерело тестового зображення:",
                                     ["З папки img", "Завантажити своє зображення"])

uploaded_file = None
selected_image_path = None
image_name = ""

if test_image_source == "З папки img":
    if not classes:
        st.error("❌ У папці img не знайдено зображень!")
        st.info("📁 Додайте зображення у папку img в корені проекту.")
    else:
        # Створення списку всіх зображень для вибору
        all_images = []
        for class_name, images in classes.items():
            for img in images:
                all_images.append(img)

        selected_image = st.sidebar.selectbox("Оберіть тестове зображення:", all_images)
        selected_image_path = os.path.join(img_folder, selected_image)
        image_name = selected_image
else:
    uploaded_file = st.sidebar.file_uploader("Завантажте своє зображення",
                                             type=["bmp", "png", "jpg", "jpeg"])
    if uploaded_file:
        image_name = uploaded_file.name

# Підготовка даних для навчання
training_patterns = []
training_class_names = []  # Назви класів для кожного зразка
training_files = []  # Імена файлів
class_mapping = {}  # Відображення: індекс нейрона -> клас

st.sidebar.header("Навчання мережі")

if classes:
    # Відображення доступних класів
    st.sidebar.write("**Доступні класи:**")
    total_images = 0
    for class_name, images in classes.items():
        st.sidebar.write(f"**{class_name}** ({len(images)} зображень):")
        for img in images:
            st.sidebar.write(f"  📷 {img}")
        total_images += len(images)

if st.sidebar.button("🎯 Навчити мережу на ВСІХ зображеннях", type="primary"):
    if not classes:
        st.sidebar.error("❌ Немає зображень для навчання! Додайте зображення у папку img.")
    else:
        with st.spinner("Обробка всіх зображень для навчання..."):
            success_count = 0
            total_processed = 0

            # Проходимо по всіх класах та всіх зображеннях
            for class_name, images in classes.items():
                for img_file in images:
                    total_processed += 1
                    try:
                        img_path = os.path.join(img_folder, img_file)
                        with open(img_path, 'rb') as f:
                            if use_improved_processing:
                                norm_vector, abs_vector = improve_image_processing(f, grid_size, threshold_value)
                            else:
                                norm_vector, abs_vector = process_image_to_vector(f, grid_size, threshold_value)

                        # Перевірка, чи вектор не нульовий
                        if sum(abs_vector) > 0:
                            training_patterns.append(norm_vector)
                            training_class_names.append(class_name)
                            training_files.append(img_file)
                            success_count += 1
                            st.sidebar.success(f"✅ {class_name} - {img_file} - успішно додано")
                        else:
                            st.sidebar.warning(f"⚠️ {class_name} - {img_file} - нульовий вектор, пропускаємо")

                    except Exception as e:
                        st.sidebar.error(f"❌ {class_name} - {img_file} - помилка: {str(e)}")

            st.sidebar.write(f"**Підсумок:** Оброблено {success_count} з {total_processed} зображень")

            if success_count >= 2:
                # Ініціалізація мережі
                vector_size = len(training_patterns[0])
                num_neurons = len(training_patterns)  # Кількість нейронів = кількість зображень

                hamming_net = HammingNetwork(vector_size, num_neurons, v_param)
                hamming_net.initialize_weights(training_patterns)

                st.sidebar.success(f"🎉 Мережа навчена! Нейронів: {num_neurons}, Зображень: {success_count}")
                st.session_state.hamming_net = hamming_net
                st.session_state.training_class_names = training_class_names
                st.session_state.training_patterns = training_patterns
                st.session_state.training_files = training_files

                # Створюємо статистику по класах
                class_stats = {}
                for class_name in training_class_names:
                    class_stats[class_name] = class_stats.get(class_name, 0) + 1
                st.session_state.class_stats = class_stats

                # Відображення інформації про навчання
                with st.sidebar.expander("📋 Деталі навчання"):
                    st.write("**Статистика навчання:**")
                    for class_name, count in class_stats.items():
                        st.write(f"- {class_name}: {count} зображень")

                    st.write(f"**Всього навчальних зразків:** {len(training_patterns)}")
                    st.write(f"**Розмірність вектора:** {vector_size}")
                    st.write(f"**Параметр гальмування (v):** {v_param}")

            else:
                st.sidebar.error("❌ Недостатньо успішно оброблених зображень для навчання (потрібно мінімум 2)")

# Перевірка чи мережа вже навчена
if 'hamming_net' in st.session_state:
    hamming_net = st.session_state.hamming_net
    total_samples = len(st.session_state.training_class_names)
    unique_classes = len(set(st.session_state.training_class_names))
    st.sidebar.success(f"✅ Мережа навчена на {total_samples} зображеннях ({unique_classes} класів)")

# Обробка тестового зображення
current_image = uploaded_file if uploaded_file else selected_image_path

if current_image is not None:
    try:
        # Завантаження та обробка зображення
        if isinstance(current_image, str):  # Шлях до файлу
            with open(current_image, 'rb') as f:
                image_bytes = f.read()
            true_class = None
            for class_name, images in classes.items():
                if image_name in images:
                    true_class = class_name
                    break
        else:  # UploadedFile
            image_bytes = current_image.read()
            true_class = "невідомий"

        pil_image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(pil_image.convert('L'))

        # Обробка зображення
        if use_improved_processing:
            norm_vector, abs_vector = improve_image_processing(io.BytesIO(image_bytes), grid_size, threshold_value)
        else:
            norm_vector, abs_vector = process_image_to_vector(io.BytesIO(image_bytes), grid_size, threshold_value)

        # Створення бінарного зображення для візуалізації
        _, binary_image = cv2.threshold(image_array, threshold_value, 255, cv2.THRESH_BINARY)
        grid_image = create_grid_image(binary_image, grid_size)

        # Показ результатів
        col1, col2 = st.columns(2)
        with col1:
            st.image(pil_image, caption=f"Оригінальне зображення: {image_name}", use_column_width=True)
        with col2:
            st.image(grid_image, caption=f"Бінаризація та сегментація ({grid_size})", use_column_width=True)

        # Інформація про зображення
        st.write(f"### Інформація про зображення:")
        st.write(f"**Файл:** {image_name}")
        if true_class:
            st.write(f"**Справжній клас:** {true_class}")
        st.write(f"**Сума пікселів:** {sum(abs_vector)}")
        st.write(f"**Розмірність вектора:** {len(abs_vector)}")

        # Вектори ознак
        st.write("### Вектори ознак:")
        abs_text = "; ".join([f"{val}" for val in abs_vector])
        norm_text = "; ".join([f"{val:.6f}" for val in norm_vector])

        col3, col4 = st.columns(2)
        with col3:
            st.text_area("Абсолютні значення:", abs_text, height=100)
        with col4:
            st.text_area("Нормовані значення:", norm_text, height=100)

        # Класифікація мережею Хеммінга
        if 'hamming_net' in st.session_state and st.session_state.hamming_net is not None:
            hamming_net = st.session_state.hamming_net
            st.write("## Класифікація мережею Хеммінга")

            winner, outputs, iterations = hamming_net.predict(norm_vector, max_iter)

            # Визначення імені класу переможця
            predicted_class = st.session_state.training_class_names[winner]
            predicted_file = st.session_state.training_files[winner]

            st.write(f"**Результат класифікації:**")
            if true_class:
                st.write(f"**Справжній клас:** {true_class}")
            st.write(f"**Розпізнаний клас:** {predicted_class}")
            st.write(f"**Найближчий зразок:** {predicted_file}")

            # Перевірка правильності
            if true_class and true_class != "невідомий":
                if true_class == predicted_class:
                    st.success("🎉 Класифікація правильна!")
                else:
                    st.error("❌ Класифікація неправильна!")

            st.write(f"**Кількість ітерацій:** {iterations}")

            # Групуємо результати по класах
            class_results = {}
            for i, output in enumerate(outputs):
                class_name = st.session_state.training_class_names[i]
                file_name = st.session_state.training_files[i]
                if class_name not in class_results:
                    class_results[class_name] = []
                class_results[class_name].append((output, file_name))

            # Вивід результатів по класах
            st.write("**Результати по класах (середні значення):**")
            for class_name, results in class_results.items():
                avg_output = np.mean([r[0] for r in results])
                best_file = max(results, key=lambda x: x[1])[1]  # Файл з найвищим значенням

                if class_name == predicted_class:
                    st.write(f"🏆 **{class_name}**: {avg_output:.6f} (найкращий: {best_file}) **← ПЕРЕМОЖЕЦЬ**")
                else:
                    st.write(f"**{class_name}**: {avg_output:.6f} (найкращий: {best_file})")

            # Візуалізація середніх результатів по класах
            st.write("**Візуалізація середніх результатів по класах:**")

            avg_class_outputs = []
            class_labels = []
            for class_name, results in class_results.items():
                avg_output = np.mean([r[0] for r in results])
                avg_class_outputs.append(avg_output)
                class_labels.append(class_name)

            chart_data = {
                'Клас': class_labels,
                'Середнє значення': avg_class_outputs
            }

            st.bar_chart(data=chart_data, x='Клас', y='Середнє значення')


        else:
            st.warning("⚠️ Мережа не навчена! Натисніть кнопку 'Навчити мережу на ВСІХ зображеннях' в боковій панелі.")

    except Exception as e:
        st.error(f"❌ Помилка обробки зображення: {str(e)}")
else:
    if test_image_source == "З папки img" and not classes:
        st.info("📁 Додайте зображення у папку img для початку роботи.")
    else:
        st.info("📷 Оберіть або завантажте тестове зображення для аналізу.")

# Інформація про доступні класи
if classes:
    st.sidebar.header("📊 Статистика класів")
    total_all_images = 0
    for class_name, images in classes.items():
        st.sidebar.write(f"**{class_name}:** {len(images)} зображень")
        total_all_images += len(images)
    st.sidebar.write(f"**Всього:** {total_all_images} зображень")
