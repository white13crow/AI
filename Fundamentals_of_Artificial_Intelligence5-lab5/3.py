import io
import os

from PIL import Image
import streamlit as st
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Системa розпізнавання фігур", layout="wide")
st.write("# Системa розпізнавання геометричних фігур")


class PatternRecognitionSystem:
    def __init__(self):
        self.training_data = {}
        self.class_stats = {}
        self.recommended_grid = "6x6"
        self.class_names = ["Квадрат", "Коло", "Ромб", "Трикутник"]

    def auto_detect_grid_size(self, image_shape):
        """Автоматично визначає оптимальний розмір сітки"""
        height, width = image_shape
        return "6x6"  # Фіксовано 6x6 для ваших даних

    def extract_features(self, image_array, grid_size=None):
        """Видобуває абсолютні та нормовані вектори ознак"""
        if grid_size is None:
            grid_size = self.auto_detect_grid_size(image_array.shape)

        rows, cols = map(int, grid_size.split('x'))
        img_height, img_width = image_array.shape

        cell_height = img_height // rows
        cell_width = img_width // cols

        absolute_vector = []

        for i in range(rows):
            for j in range(cols):
                y_start = i * cell_height
                y_end = (i + 1) * cell_height if i < rows - 1 else img_height
                x_start = j * cell_width
                x_end = (j + 1) * cell_width if j < cols - 1 else img_width

                cell = image_array[y_start:y_end, x_start:x_end]
                black_pixels = np.sum(cell == 0)
                absolute_vector.append(black_pixels)

        # Нормування за сумою
        total_sum = sum(absolute_vector)
        if total_sum > 0:
            normalized_vector = [val / total_sum for val in absolute_vector]
        else:
            normalized_vector = [0 for _ in absolute_vector]

        return absolute_vector, normalized_vector, grid_size

    def calculate_statistics(self, class_name):
        """Розраховує статистичні параметри для класу"""
        if class_name not in self.training_data or len(self.training_data[class_name]) == 0:
            return None

        vectors = [item['normalized'] for item in self.training_data[class_name]]
        vectors_array = np.array(vectors)

        stats = {
            'mean': np.mean(vectors_array, axis=0),
            'std': np.std(vectors_array, axis=0),
            'min': np.min(vectors_array, axis=0),
            'max': np.max(vectors_array, axis=0),
            'count': len(vectors),
            'vector_size': len(vectors[0])
        }

        return stats

    def classify_pattern(self, unknown_vector, method='euclidean'):
        """Класифікує невідомий образ"""
        if not self.training_data:
            return "Немає навчальних даних", float('inf')

        best_class = None
        best_distance = float('inf')
        unknown_size = len(unknown_vector)

        for class_name in self.training_data:
            if not self.training_data[class_name]:
                continue

            compatible_vectors = []
            for item in self.training_data[class_name]:
                if len(item['normalized']) == unknown_size:
                    compatible_vectors.append(item['normalized'])

            if not compatible_vectors:
                continue

            class_mean = np.mean(compatible_vectors, axis=0)

            if method == 'euclidean':
                distance = np.linalg.norm(np.array(unknown_vector) - np.array(class_mean))
            elif method == 'manhattan':
                distance = np.sum(np.abs(np.array(unknown_vector) - np.array(class_mean)))
            elif method == 'cosine':
                dot_product = np.dot(unknown_vector, class_mean)
                norm_unknown = np.linalg.norm(unknown_vector)
                norm_class = np.linalg.norm(class_mean)
                distance = 1 - dot_product / (norm_unknown * norm_class) if norm_unknown * norm_class > 0 else 1

            if distance < best_distance:
                best_distance = distance
                best_class = class_name

        return best_class, best_distance


# Ініціалізація системи
if 'recognition_system' not in st.session_state:
    st.session_state.recognition_system = PatternRecognitionSystem()

system = st.session_state.recognition_system


def load_all_images_from_folder(folder_path, class_name):
    """Завантажує ВСІ зображення з папки, що відносяться до класу"""
    if not os.path.exists(folder_path):
        st.warning(f"Папка {folder_path} не існує.")
        return 0

    # Знаходимо всі зображення для цього класу
    image_files = []
    for f in os.listdir(folder_path):
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # Перевіряємо, чи файл належить до поточного класу
            class_keywords = {
                "Квадрат": ["квадрат", "square"],
                "Коло": ["коло", "circle", "круг"],
                "Ромб": ["ромб", "rhombus", "diamond"],
                "Трикутник": ["трикутник", "triangle"]
            }

            filename_lower = f.lower()
            keywords = class_keywords.get(class_name, [class_name.lower()])

            if any(keyword in filename_lower for keyword in keywords):
                image_files.append(f)

    if not image_files:
        st.warning(f"Не знайдено зображень для класу '{class_name}' в папці {folder_path}")
        return 0

    loaded_count = 0
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, filename in enumerate(image_files):
        try:
            status_text.text(f"Обробка {i + 1}/{len(image_files)}: {filename}")
            progress_bar.progress((i + 1) / len(image_files))

            image_path = os.path.join(folder_path, filename)
            pil_image = Image.open(image_path)

            # Конвертуємо в чорно-біле та бінаризуємо
            image_array = np.array(pil_image.convert('L'))
            _, binary_image = cv2.threshold(image_array, 128, 255, cv2.THRESH_BINARY)

            # Виділення ознак
            absolute_vector, normalized_vector, grid_size = system.extract_features(binary_image)

            # Збереження даних
            if class_name not in system.training_data:
                system.training_data[class_name] = []

            system.training_data[class_name].append({
                'image': pil_image,
                'absolute': absolute_vector,
                'normalized': normalized_vector,
                'grid_size': grid_size,
                'filename': filename,
                'vector_size': len(absolute_vector)
            })

            loaded_count += 1

        except Exception as e:
            st.error(f"Помилка обробки {filename}: {str(e)}")

    progress_bar.empty()
    status_text.empty()

    # Оновлення статистики
    if loaded_count > 0:
        system.class_stats[class_name] = system.calculate_statistics(class_name)

    return loaded_count


# Основний інтерфейс
tab1, tab2, tab3 = st.tabs(["Навчання системи", "Класифікація", "Статистика кластерів"])

with tab1:
    st.header("Навчання системи розпізнавання фігур")

    st.info("🎯 **Система завантажує ВСІ зображення з папки для обраного класу**")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Завантаження зображень фігур")

        class_name = st.selectbox("Оберіть клас фігури:", system.class_names)

        st.write("### Масове завантаження")
        st.write(f"**Система знайде ВСІ зображення для:** {class_name}")

        folder_path = st.text_input("Шлях до папки з зображеннями:", value="./img/")

        # Кнопка для завантаження ВСІХ зображень класу
        if st.button("📁 Завантажити ВСІ зображення цього класу"):
            with st.spinner(f"Пошук та обробка ВСІХ зображень для {class_name}..."):
                loaded_count = load_all_images_from_folder(folder_path, class_name)
                if loaded_count > 0:
                    st.success(f"✅ Завантажено ВСІ {loaded_count} зображень для фігури '{class_name}'")

                    # Показуємо список завантажених файлів
                    if class_name in system.training_data:
                        st.write("**Завантажені файли:**")
                        files_col1, files_col2 = st.columns(2)
                        files_list = [sample['filename'] for sample in system.training_data[class_name]]

                        mid_index = len(files_list) // 2
                        with files_col1:
                            for filename in files_list[:mid_index]:
                                st.write(f"• {filename}")
                        with files_col2:
                            for filename in files_list[mid_index:]:
                                st.write(f"• {filename}")
                else:
                    st.error(f"❌ Не знайдено зображень для класу '{class_name}'")

        st.write("---")
        st.subheader("Завантажити всі класи одразу")

        if st.button("🚀 Завантажити ВСІ класи автоматично"):
            total_loaded = 0
            progress_text = st.empty()

            for i, cls in enumerate(system.class_names):
                progress_text.text(f"Обробка класу {i + 1}/{len(system.class_names)}: {cls}")
                loaded_count = load_all_images_from_folder("./img/", cls)
                total_loaded += loaded_count
                if loaded_count > 0:
                    st.success(f"✅ {cls}: {loaded_count} зображень")
                else:
                    st.warning(f"⚠️ {cls}: зображень не знайдено")

            progress_text.empty()

            if total_loaded > 0:
                st.success(f"🎉 Завантажено всього: {total_loaded} зображень")
            else:
                st.error("❌ Не знайдено жодного зображення")

    with col2:
        st.subheader("Поточна навчальна послідовність")

        total_samples = 0
        for class_name in system.class_names:
            if class_name in system.training_data and system.training_data[class_name]:
                samples = system.training_data[class_name]
                vector_size = samples[0]['vector_size']
                grid_size = samples[0]['grid_size']

                st.write(f"### {class_name}")
                st.write(f"**Кількість зразків:** {len(samples)}")
                st.write(f"**Розмірність векторів:** {vector_size} ознак")
                st.write(f"**Використана сітка:** {grid_size}")

                # Показуємо всі зображення класу в сітці
                st.write("**Усі завантажені зображення:**")
                num_cols = 4  # Кількість стовпців для відображення
                cols = st.columns(num_cols)

                for i, sample in enumerate(samples):
                    with cols[i % num_cols]:
                        st.image(sample['image'], width=80, caption=sample['filename'])

                # Детальна інформація про вектори
                with st.expander(f"📊 Статистика векторів для {class_name}"):
                    st.write("**Останній оброблений зразок:**")
                    last_sample = samples[-1]

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.write("📈 Абсолютні значення:")
                        abs_text = "; ".join([f"{val}" for val in last_sample['absolute'][:8]])
                        if len(last_sample['absolute']) > 8:
                            abs_text += f"..."
                        st.text_area("", abs_text, height=60, key=f"abs_{class_name}")

                    with col_b:
                        st.write("📊 Нормовані значення:")
                        norm_text = "; ".join([f"{val:.4f}" for val in last_sample['normalized'][:8]])
                        if len(last_sample['normalized']) > 8:
                            norm_text += f"..."
                        st.text_area("", norm_text, height=60, key=f"norm_{class_name}")

                    # Загальна статистика по класу
                    if class_name in system.class_stats and system.class_stats[class_name]:
                        stats = system.class_stats[class_name]
                        st.write(f"**Статистика класу:**")
                        st.write(f"- Середнє значення ознак: {np.mean(stats['mean']):.4f}")
                        st.write(f"- Середнє відхилення: {np.mean(stats['std']):.4f}")

                total_samples += len(samples)
                st.write("---")
            else:
                st.write(f"### {class_name}")
                st.write("**Кількість зразків:** 0")
                st.write("---")

        # Загальна статистика
        st.subheader("📈 Загальна статистика")
        st.write(f"**Всього зразків у системі:** {total_samples}")
        st.write(
            f"**Навчені класи:** {sum(1 for cls in system.class_names if cls in system.training_data and system.training_data[cls])}/{len(system.class_names)}")

with tab2:
    st.header("Класифікація невідомої фігури")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Введення фігури для розпізнавання")

        classification_method = st.selectbox("Метод класифікації:",
                                             ["euclidean", "manhattan", "cosine"])

        unknown_file = st.file_uploader("Оберіть зображення фігури",
                                        type=["bmp", "png", ".jpg", ".jpeg"])

        if unknown_file and st.button("🔍 Розпізнати фігуру"):
            try:
                # Обробка зображення
                image_bytes = unknown_file.read()
                pil_image = Image.open(io.BytesIO(image_bytes))
                image_array = np.array(pil_image.convert('L'))
                _, binary_image = cv2.threshold(image_array, 128, 255, cv2.THRESH_BINARY)

                # Виділення ознак
                absolute_vector, normalized_vector, detected_grid = system.extract_features(binary_image)

                # Відображення результатів
                st.image(pil_image, caption=f"Тестова фігура: {unknown_file.name}", use_column_width=True)

                st.write("### Результати аналізу:")
                st.write(f"**Використана сітка:** {detected_grid}")
                st.write(f"**Розмірність вектора:** {len(absolute_vector)} ознак")

                # Класифікація
                if system.training_data:
                    result_class, distance = system.classify_pattern(normalized_vector, classification_method)

                    if result_class:
                        confidence = max(0, 1 - distance) * 100

                        # Результат класифікації
                        st.success(f"**Результат розпізнавання:** {result_class}")

                        # Метрики
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Відстань", f"{distance:.6f}")
                        with col2:
                            st.metric("Впевненість", f"{confidence:.1f}%")
                        with col3:
                            st.metric("Метод", classification_method)


                        st.write("**Детальний аналіз:**")
                        analysis_data = []
                        for class_name in system.class_names:
                            if class_name in system.training_data:
                                compatible_samples = [item for item in system.training_data[class_name]
                                                      if len(item['normalized']) == len(normalized_vector)]
                                analysis_data.append({
                                    'Клас': class_name,
                                    'Зразків': len(system.training_data[class_name]),
                                    'Сумісних': len(compatible_samples),
                                    'Відстань': distance if class_name == result_class else "N/A"
                                })

                        if analysis_data:
                            df = pd.DataFrame(analysis_data)
                            st.dataframe(df, hide_index=True)

                    else:
                        st.error("Не знайдено сумісних навчальних даних")

                else:
                    st.error("Система ще не навчена")

            except Exception as e:
                st.error(f"Помилка обробки: {str(e)}")

    with col2:
        st.subheader("Стан системи")

        if system.training_data:
            st.success("✅ Система навчена")

            # Статистика навчених класів
            st.write("**Навчені класи:**")
            for class_name in system.class_names:
                if class_name in system.training_data:
                    count = len(system.training_data[class_name])
                    st.write(f"✅ **{class_name}:** {count} зразків")
                else:
                    st.write(f"❌ **{class_name}:** не навчений")

            # Загальна інформація
            total_samples = sum(len(system.training_data[cls]) for cls in system.training_data)
            st.write(f"**Всього навчальних зразків:** {total_samples}")

        else:
            st.warning("⚠️ Система не навчена")

with tab3:
    st.header("Статистична обробка кластерів фігур")

    if system.class_stats:
        for class_name in system.class_names:
            if class_name in system.class_stats and system.class_stats[class_name]:
                stats = system.class_stats[class_name]

                st.subheader(f"📊 {class_name}")

                # Статистичні картки
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Зразків", stats['count'])
                with col2:
                    st.metric("Ознак", stats['vector_size'])
                with col3:
                    avg_mean = np.mean(stats['mean'])
                    st.metric("Середнє", f"{avg_mean:.4f}")
                with col4:
                    avg_std = np.mean(stats['std'])
                    st.metric("Відхилення", f"{avg_std:.4f}")

                # Візуалізація
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

                ax1.bar(range(stats['vector_size']), stats['mean'], alpha=0.7, color='blue')
                ax1.set_title(f'Середні значення ознак - {class_name}')
                ax1.set_xlabel('Номер ознаки')
                ax1.set_ylabel('Значення')
                ax1.grid(True, alpha=0.3)

                ax2.bar(range(stats['vector_size']), stats['std'], alpha=0.7, color='red')
                ax2.set_title(f'Стандартні відхилення - {class_name}')
                ax2.set_xlabel('Номер ознаки')
                ax2.set_ylabel('Відхилення')
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                st.pyplot(fig)

                st.write("---")
    else:
        st.info("📊 Статистика з'явиться після навчання системи")

# Бічна панель
st.sidebar.header("⚙️ Керування системою")

if st.sidebar.button("🔄 Очистити всі дані"):
    st.session_state.recognition_system = PatternRecognitionSystem()
    st.rerun()

