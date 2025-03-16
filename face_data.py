from setmode import *  # Импорт настроек режима работы (например, ffmode, frmode)
import sqlite3  # Библиотека для работы с базой данных SQLite
import numpy as np  # Библиотека для работы с массивами и математическими операциями
from PIL import Image  # Библиотека для работы с изображениями
import cv2  # Библиотека для работы с изображениями и компьютерным зрением

# Подключение к базе данных SQLite
conn = sqlite3.connect('face_data.db3', check_same_thread=False)
cursor = conn.cursor()

# Создание таблицы для хранения лиц, если она не существует
cursor.execute('''
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    namefile TEXT NOT NULL,         -- Имя файла
    filesize INTEGER NOT NULL,      -- Размер файла
    namepercone TEXT NOT NULL,      -- Имя человека
    vector512 BLOB,                    -- Вектор признаков (в зависимости от режима)
    vector128 BLOB,                 -- Вектор признаков (128-мерный)
    photo BLOB                      -- Фото в формате BLOB
)
''')
conn.commit()  # Применение изменений в базе данных


def load_faces_from_db():
    """
    Загрузка всех лиц из базы данных.

    :return: Кортеж из двух элементов:
             - known_file: Словарь с информацией о файлах.
             - known_vector: Список словарей с информацией о лицах (векторы и фото).
    """
    # Выборка данных из базы данных в зависимости от режима (frmode)
    cursor.execute('''
        SELECT namefile, filesize, namepercone,
               {}, photo
        FROM faces {}
    '''.format(('vector512' if frmode == 'adaface' else 'vector128'),
               ('')))

    faces = cursor.fetchall()  # Получение всех записей из базы данных
    known_file = {}  # Словарь для хранения информации о файлах
    known_vector = []  # Список для хранения информации о лицах (векторы и фото)

    for namefile, filesize, namepercone, vector_b, photo_b in faces:
        # Преобразуем raw-данные обратно в изображение
        image_array = np.frombuffer(photo_b, dtype=np.uint8).reshape((112, 112, -1))  # Восстанавливаем массив NumPy
        photo = Image.fromarray(image_array, 'RGB')  # Создаем изображение из массива
        photo = cv2.cvtColor(np.array(photo), cv2.COLOR_RGB2BGR)  # Преобразуем PIL.Image в формат OpenCV (BGR)

        # Если вектор признаков существует, добавляем его в список known_vector
        if vector_b is not None:
            vector = np.frombuffer(vector_b, dtype=np.float32)  # Восстанавливаем вектор из BLOB
            known_vector.append({
                "namepercone": namepercone,
                "vector": vector,
                "photo": photo
            })

        # Добавляем информацию о файле в словарь known_file
        known_file[f"{namefile}_{filesize}"] = {
            "namefile": namefile,
            "filesize": filesize,
            "namepercone": namepercone
        }

    print(f'[i] В базе найдено {len(faces)} записей.')
    return known_file, known_vector


def add_face_to_db(namefile, filesize, namepercone, vector128:np.array, vector512:np.array, photo):
    """
    Добавление нового лица в базу данных.

    :param namefile: Имя файла.
    :param filesize: Размер файла.
    :param namepercone: Имя человека.
    :param vector: Вектор признаков.
    :param photo: Фото в формате NumPy массива.
    """
    # Преобразуем фото и вектор в бинарный формат (BLOB)
    photo_blob = np.array(photo).tobytes()

    # Списки для хранения полей и значений
    fields = ['namefile', 'filesize', 'namepercone', 'photo']
    values = [namefile, filesize, namepercone, photo_blob]

    # Добавляем vector512, если он не None
    if vector512 is not None:
        vector_blob512 = vector512.tobytes()
        fields.append('vector512'); values.append(vector_blob512)
    if vector128 is not None:
        vector_blob128 = vector128.tobytes()
        fields.append('vector128'); values.append(vector_blob128)

    # Формируем SQL-запрос
    insert_query = f'''
        INSERT INTO faces ({', '.join(fields)})
        VALUES ({', '.join(['?'] * len(values))})'''

    # Выполняем запрос и фиксируем изменения в базе данных
    cursor.execute(insert_query, values)
    conn.commit()

# Пример использования функций
if __name__ == "__main__":
    # Загрузка данных из базы
    known_file, known_vector = load_faces_from_db()

    # Пример добавления нового лица
    # add_face_to_db("new_face.jpg", 1024, "Иванов Иван", np.random.rand(128).astype(np.float32), np.random.rand(112, 112, 3).astype(np.uint8))
