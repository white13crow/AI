import io
import numpy as np
import cv2
from PIL import Image


def process_image_to_vector(image_file, grid_size, threshold=128):
    """
    Обробка зображення та отримання вектора ознак

    Args:
        image_file: файл зображення
        grid_size (str): розмір сітки (наприклад, "5x5")
        threshold (int): поріг бінаризації

    Returns:
        tuple: (нормований вектор, абсолютний вектор)
    """
    image_bytes = image_file.read()
    pil_image = Image.open(io.BytesIO(image_bytes))
    image_array = np.array(pil_image.convert('L'))

    # Бінаризація з заданим порогом
    _, binary_image = cv2.threshold(image_array, threshold, 255, cv2.THRESH_BINARY)

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

    # Нормування за сумою
    total_sum = sum(absolute_vector)
    if total_sum > 0:
        normalized_vector = [val / total_sum for val in absolute_vector]
    else:
        normalized_vector = [0 for _ in absolute_vector]

    return normalized_vector, absolute_vector


def improve_image_processing(image_file, grid_size, threshold=128):
    """
    Покращена обробка зображення з кращою бінаризацією

    Args:
        image_file: файл зображення
        grid_size (str): розмір сітки
        threshold (int): поріг бінаризації

    Returns:
        tuple: (нормований вектор, абсолютний вектор)
    """
    image_bytes = image_file.read()
    pil_image = Image.open(io.BytesIO(image_bytes))
    image_array = np.array(pil_image.convert('L'))

    # Покращена бінаризація з адаптацією
    if threshold == 0:  # Автоматичний поріг
        threshold = np.mean(image_array) - 20  # Трохи нижче середнього

    # Застосовуємо морфологічні операції для покращення
    _, binary_image = cv2.threshold(image_array, threshold, 255, cv2.THRESH_BINARY)

    # Морфологічне закриття для заповнення дірок
    kernel = np.ones((3, 3), np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

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

    # Нормування за сумою
    total_sum = sum(absolute_vector)
    if total_sum > 0:
        normalized_vector = [val / total_sum for val in absolute_vector]
    else:
        normalized_vector = [0 for _ in absolute_vector]

    return normalized_vector, absolute_vector


def create_grid_image(binary_image, grid_size):
    """
    Створення зображення з накладеною сіткою

    Args:
        binary_image: бінарне зображення
        grid_size (str): розмір сітки

    Returns:
        numpy array: зображення з сіткою
    """
    grid_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    rows, cols = map(int, grid_size.split('x'))
    img_height, img_width = binary_image.shape
    cell_height = img_height // rows
    cell_width = img_width // cols

    # Малювання сітки
    for i in range(rows):
        for j in range(cols):
            y_start = i * cell_height
            x_start = j * cell_width

            if j > 0:  # Вертикальні лінії
                cv2.line(grid_image, (x_start, 0), (x_start, img_height), (0, 0, 255), 2)
            if i > 0:  # Горизонтальні лінії
                cv2.line(grid_image, (0, y_start), (img_width, y_start), (0, 0, 255), 2)

    # Зовнішній контур
    cv2.rectangle(grid_image, (0, 0), (img_width - 1, img_height - 1), (0, 255, 0), 2)

    return grid_image
