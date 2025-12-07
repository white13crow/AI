import subprocess
import sys
import os


def install_requirements():
    """Встановлення залежностей"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Залежності успішно встановлено!")
    except subprocess.CalledProcessError:
        print("Помилка встановлення залежностей!")


def check_img_folder():
    """Перевірка наявності папки img"""
    if not os.path.exists("../img"):
        print("Увага: Папка img не знайдена в корені проекту!")
        print("Створіть папку img та додайте туди зображення для роботи програми")
        create_img = input("Створити папку img? (y/n): ")
        if create_img.lower() == 'y':
            os.makedirs("../img", exist_ok=True)
            print("Папка img створена!")


def run_streamlit():
    """Запуск Streamlit додатку"""
    try:
        subprocess.check_call([sys.executable, "-m", "streamlit", "run", "main.py"])
    except subprocess.CalledProcessError:
        print("Помилка запуску Streamlit!")


if __name__ == "__main__":
    print("Перевірка структури проекту...")
    check_img_folder()

    if not os.path.exists("requirements.txt"):
        print("Файл requirements.txt не знайдено!")
    else:
        print("Встановлення залежностей...")
        install_requirements()
        print("Запуск додатку...")
        run_streamlit()