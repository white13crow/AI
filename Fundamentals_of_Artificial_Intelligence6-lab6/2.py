import io
import os
import glob

from PIL import Image
import streamlit as st
import numpy as np
import cv2
import pandas as pd


st.set_page_config(page_title="Аналіз векторів ознак", layout="wide")
st.write("# Аналіз векторів ознак зображень (Манхетенська норма)")

# Шлях до папки з еталонними образами
IMAGE_FOLDER = "img"
CLASS_NAMES = ["Коло", "Трикутник", "Квадрат", "Ромb"]
SUPPORTED_EXTENSIONS = ['*.png', '*.jpg', '*.jpeg', '*.bmp']

GRID_ROWS, GRID_COLS = 5, 5

# Ініціалізація станів сесії
if 'reference_classes' not in st.session_state:
    st.session_state.reference_classes = {}

    # Автоматичне завантаження еталонних образів для кожного класу
    for class_name in CLASS_NAMES:
        image_files = []

        # Шукаємо файли з різними розширеннями
        for extension in SUPPORTED_EXTENSIONS:
            pattern = os.path.join(IMAGE_FOLDER, f"{class_name}_*{extension}")
            found_files = glob.glob(pattern)
            image_files.extend(found_files)

        if image_files:
            # Завантажуємо всі зображення класу
            all_images = []
            for img_path in image_files:
                try:
                    pil_image = Image.open(img_path)
                    all_images.append({
                        'image': pil_image,
                        'path': img_path
                    })
                except Exception as e:
                    st.error(f"Помилка при завантаженні {img_path}: {str(e)}")

            if all_images:
                st.session_state.reference_classes[class_name] = {
                    'all_images': all_images,  # всі зображення класу
                    'image_paths': image_files
                }
                st.success(f"Еталонний клас '{class_name}' завантажено успішно! Знайдено {len(image_files)} зображень.")
        else:
            st.warning(f"Не знайдено зображень для класу {class_name}!")


# Функція для обчислення вектора ознак
def calculate_feature_vector(image_array, rows, cols):
    _, binary_image = cv2.threshold(image_array, 128, 255, cv2.THRESH_BINARY)

    img_height, img_width = binary_image.shape
    cell_height = img_height // rows
    cell_width = img_width // cols

    absolute_vector = []

    for i in range(rows):
        for j in range(cols):
            # Визначення меж комірки
            y_start = i * cell_height
            y_end = (i + 1) * cell_height if i < rows - 1 else img_height
            x_start = j * cell_width
            x_end = (j + 1) * cell_width if j < cols - 1 else img_width

            # Підрахунок пікселів у комірці
            cell = binary_image[y_start:y_end, x_start:x_end]
            black_pixels = np.sum(cell == 0)
            absolute_vector.append(black_pixels)

    # Нормування за сумою
    total_sum = sum(absolute_vector)
    if total_sum > 0:
        normalized_vector = [val / total_sum for val in absolute_vector]
    else:
        normalized_vector = [0 for _ in absolute_vector]

    return absolute_vector, normalized_vector


# Функція для обчислення манхетенської відстані
def manhattan_distance(vec1, vec2):
    """
    Обчислює манхетенську відстань між двома векторами
    Формула: d(X₁, X₂) = Σ|xₖ¹ - xₖ²|
    """
    return np.sum(np.abs(np.array(vec1) - np.array(vec2)))


# Обробка еталонних образів
if st.session_state.reference_classes:
    for class_name, class_data in list(st.session_state.reference_classes.items()):
        try:
            # Обчислюємо вектори для ВСІХ зображень класу
            all_vectors = []
            for img_data in class_data['all_images']:
                image_array = np.array(img_data['image'].convert('L'))
                abs_vector, norm_vector = calculate_feature_vector(image_array, GRID_ROWS, GRID_COLS)

                all_vectors.append({
                    'absolute_vector': abs_vector,
                    'normalized_vector': norm_vector,
                    'image': img_data['image'],
                    'path': img_data['path']
                })

            # Оновлення даних еталонного класу
            st.session_state.reference_classes[class_name].update({
                'all_vectors': all_vectors  # зберігаємо вектори для всіх зображень
            })

        except Exception as e:
            st.error(f"Помилка при обробці еталонного образу {class_name}: {str(e)}")


# Відображення векторів еталонних образів
if st.session_state.reference_classes:
    st.write("## Вектори ознак еталонних образів")

    for class_name, class_data in st.session_state.reference_classes.items():
        if 'all_vectors' in class_data:
            with st.expander(f"Вектори для {class_name} ({len(class_data['all_vectors'])} зображень)"):

                # Відображаємо ВСІ зображення класу з їх векторами
                for i, vector_data in enumerate(class_data['all_vectors']):
                    st.write(f"### Зображення {i + 1}: {os.path.basename(vector_data['path'])}")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(
                            vector_data['image'],
                            caption=f"{os.path.basename(vector_data['path'])}",
                            use_column_width=True
                        )

                    with col2:
                        # Абсолютний вектор
                        st.write("**Абсолютний вектор:**")
                        abs_text = "; ".join([f"{val}" for val in vector_data['absolute_vector']])
                        st.text_area(f"Абсолютні значення:", abs_text, height=80, key=f"abs_{class_name}_{i}")

                        # Нормований вектор
                        st.write("**Нормований вектор:**")
                        norm_text = "; ".join([f"{val:.6f}" for val in vector_data['normalized_vector']])
                        st.text_area(f"Нормовані значення:", norm_text, height=80, key=f"norm_{class_name}_{i}")

                    st.markdown("---")  # роздільник між зображеннями


# Секція для аналізу невідомого образу
st.write("## Аналіз невідомого образу")

unknown_file = st.file_uploader(
    "Оберіть зображення невідомого образу",
    type=["bmp", "png", "jpg", "jpeg"],
    key="unknown_uploader"
)

if unknown_file is not None and len(st.session_state.reference_classes) >= 2:
    try:
        # Обробка невідомого зображення
        image_bytes = unknown_file.read()
        pil_image = Image.open(io.BytesIO(image_bytes))

        # Відображення невідомого образу
        st.write("### Невідомий образ:")
        st.image(pil_image, caption="Невідоме зображення", use_column_width=True)

        # Обчислення векторів ознак
        image_array = np.array(pil_image.convert('L'))
        abs_vector, norm_vector = calculate_feature_vector(image_array, GRID_ROWS, GRID_COLS)

        # Абсолютний вектор
        st.write("### Абсолютний вектор ознак невідомого образу:")
        abs_text = "; ".join([f"{val}" for val in abs_vector])
        st.text_area("Абсолютні значення:", abs_text, height=100, key="unknown_abs")

        # Нормований вектор
        st.write("### Нормований вектор ознак невідомого образу:")
        norm_text = "; ".join([f"{val:.6f}" for val in norm_vector])
        st.text_area("Нормовані значення:", norm_text, height=100, key="unknown_norm")

        # Обчислення мір відповідності
        st.write("## Міри відповідності невідомого образу до еталонів")
        st.latex(r"d(X_1, X_2) = \sum_k |x_k^1 - x_k^2|")

        # Перевірка, чи всі еталони мають вектори ознак
        valid_references = {}
        for class_name, class_data in st.session_state.reference_classes.items():
            if 'all_vectors' in class_data:
                # Використовуємо перше зображення класу для порівняння
                valid_references[class_name] = class_data['all_vectors'][0]

        if not valid_references:
            st.error("Еталонні образи не мають обчислених векторів ознак!")
        else:
            # Обчислення манхетенської відстані для кожного еталонного класу
            results = []
            for class_name, class_data in valid_references.items():
                # Манхетенська відстань для нормованих векторів
                manhattan_norm = manhattan_distance(norm_vector, class_data['normalized_vector'])

                results.append({
                    'Клас': class_name,
                    'Манхетенська відстань': manhattan_norm
                })

            # Відображення результатів у вигляді таблиці
            df_results = pd.DataFrame(results)


            # Стилізація таблиці - підсвічуємо мінімальні значення
            def highlight_min(row):
                styles = [''] * len(row)

                # Мінімальні значення для манхетенських відстаней
                if row.name == 'Манхетенська відстань':
                    min_val = df_results[row.name].min()
                    if row[row.name] == min_val:
                        styles[df_results.columns.get_loc(row.name)] = 'background-color: lightgreen'

                return styles

            st.dataframe(df_results.style.apply(highlight_min, axis=1))

            # Класифікація на основі манхетенської відстані
            st.write("## Результат класифікації")

            if len(results) > 0:
                # Знаходимо найближчий клас за манхетенською відстанню
                min_manhattan_norm = min(results, key=lambda x: x['Манхетенська відстань'])

                st.success(
                    f"**Результат класифікації:**\n"
                   f"Невідомий образ належить до класу: **{min_manhattan_norm['Клас']}**\n"
                   f"Манхетенська відстань: {min_manhattan_norm['Манхетенська відстань']:.6f}"
                )

    except Exception as e:
        st.error(f"Помилка при обробці невідомого зображення: {str(e)}")
elif unknown_file is not None and len(st.session_state.reference_classes) < 2:
    st.error("❌ Для класифікації потрібно щонайменше 2 еталонних класи!")
else:
    st.info("Завантажте зображення невідомого образу для аналізу")