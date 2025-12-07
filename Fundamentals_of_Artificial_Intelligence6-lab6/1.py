import io

from PIL import Image
import streamlit as st
import numpy as np
import cv2

st.set_page_config(page_title="Аналіз векторів ознак", layout="wide")
st.write("# Аналіз векторів ознак зображень")

grid_size = st.selectbox("Розмір сітки:", ["3x3", "4x4", "5x5", "6x6", "4x5", "5x4"], index=2)
uploaded_file = st.file_uploader("Оберіть зображення", type=["bmp", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        # Завантаження зображення
        image_bytes = uploaded_file.read()
        pil_image = Image.open(io.BytesIO(image_bytes))
        image_array = np.array(pil_image.convert('L'))
        _, binary_image = cv2.threshold(image_array, 128, 255, cv2.THRESH_BINARY)
        grid_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

        # Розрахунок вектора ознак
        rows, cols = map(int, grid_size.split('x'))
        img_height, img_width = binary_image.shape
        cell_height = img_height // rows
        cell_width = img_width // cols

        absolute_vector = []

        # Малювання сітки та підрахунок пікселів
        for i in range(rows):
            for j in range(cols):
                # Визначення меж комірки
                y_start = i * cell_height
                y_end = (i + 1) * cell_height if i < rows - 1 else img_height
                x_start = j * cell_width
                x_end = (j + 1) * cell_width if j < cols - 1 else img_width

                # Малювання ліній сітки
                if j > 0:  # Вертикальні лінії
                    cv2.line(grid_image, (x_start, 0), (x_start, img_height), (0, 0, 255), 2)
                if i > 0:  # Горизонтальні лінії
                    cv2.line(grid_image, (0, y_start), (img_width, y_start), (0, 0, 255), 2)

                # Підрахунок  пікселів у комірці
                cell = binary_image[y_start:y_end, x_start:x_end]
                black_pixels = np.sum(cell == 0)
                absolute_vector.append(black_pixels)

        # Малювання зовнішніх ліній
        cv2.rectangle(grid_image, (0, 0), (img_width - 1, img_height - 1), (0, 255, 0), 2)

        # Показ зображень
        col1, col2 = st.columns(2)
        with col1:
            st.image(pil_image, caption="Оригінальне зображення", use_column_width=True)
        with col2:
            st.image(grid_image, caption=f"Сегментація ({grid_size})", use_column_width=True)

        # НОРМУВАННЯ ЗА СУМОЮ
        total_sum = sum(absolute_vector)
        if total_sum > 0:
            normalized_vector = [val / total_sum for val in absolute_vector]
        else:
            normalized_vector = [0 for _ in absolute_vector]

        # Абсолютний вектор
        st.write("### Абсолютний вектор ознак:")
        abs_text = "; ".join([f"{val}" for val in absolute_vector])
        st.text_area("Абсолютні значення:", abs_text, height=100)

        # Нормований вектор
        st.write("### Нормований вектор ознак (за сумою):")
        norm_text = "; ".join([f"{val:.6f}" for val in normalized_vector])
        st.text_area("Нормовані значення:", norm_text, height=100)

    except Exception as e:
        st.error(f"Помилка: {str(e)}")
else:
    st.info("Завантажте зображення для початку аналізу")
