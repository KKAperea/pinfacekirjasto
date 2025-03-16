# Вывод сообщения о начале инициализации
print(f'[i] -= Версия 1.01 =-', flush=True)

# Импортируем необходимые модули
from setmode import *  # Модуль для настройки режимов (ffmode, frmode и т.д.)
from pinfacekirjasto.PinFace import PinFace, cv2  # Модуль для работы с лицами
from cv2 import numpy as np  # Импортируем NumPy через OpenCV
# import numpy as np  # Альтернативный импорт NumPy

# Инициализация объекта для работы с лицами
pFace = PinFace(ffmode=ffmode, frmode=frmode,fmode =[])
'''
Параметры:
- ffmode: Режим обнаружения лиц. Может быть 'opencv', 'adaface' или 'mediapipe'.
- frmode: Режим распознавания лиц. Может быть 'sface' или 'adaface'.
'''

# Импортируем дополнительные модули
from runmelib import dt, percentage_fcbline, percentage_fcb, DrawInfo, SqrReize,facessavejpeg, appendTXT # Вспомогательные функции
from time import time, sleep  # Модуль для работы со временем
from camstream import VideoCapture, VideoFileReader  # Модуль для работы с видеопотоком
import face_data  # Модуль для работы с базой данных лиц

# Загружаем известные лица из базы данных
_, known_vector = face_data.load_faces_from_db()

# Импортируем конфигурационные параметры
from config import (
    OUTPUT_DIR, TIME_DELAY, STREAM_TO_PARSE, NAME_STREAM, RESIZE, CROP_PARAMS,
    DESC_CAM, EXTRA_QUEUE, COMP_NAME, COMP_TYPE, FRAME_NAME, WRITE_FRAME
)

# Глобальные переменные
timedalaytitle = 0  # Время последнего обновления заголовка окна
prior_additional_data = '-=!@#$%^&*()=-'  # Предыдущие дополнительные данные
prior_unknown_vector = np.ones((1, vector_size), dtype=np.float32)  # Вектор для неизвестных лиц


# Загрузка текстуры для фона
try:
    img_tekstura = cv2.imread('tekstura800600.jpg_')  # Пытаемся загрузить текстуру
except:
    img_tekstura = np.zeros((800, 600, 3), dtype=np.uint8)  # Если текстура не найдена, создаем черный фон
if img_tekstura is None:
    img_tekstura = np.zeros((800, 520, 3), dtype=np.uint8)  # Если загрузка не удалась, создаем черный фон



# Функция для обработки кадра
def ReadFrame(frame, modelayer=0, timeframe=None):
    global timedalaytitle, prior_additional_data, prior_unknown_vector

    # Обнаружение лиц на кадре
    bboxes, faces, facescv = pFace.face_detection(frame)
    tf = pFace.timeff  # Время, затраченное на обнаружение лиц


    if len(faces) != 0:
        tr = time()  # Засекаем время начала распознавания
        embeddings = pFace.face_recognition(faces=faces, facescv=facescv)  # Распознавание лиц
        # Обработка каждого обнаруженного лица
        for nfaces_, faces_ in enumerate(faces):
            bounding_boxes = bboxes[nfaces_]
            left, top, right, bottom = int(bounding_boxes[0]), int(bounding_boxes[1]), int(bounding_boxes[2]), int(bounding_boxes[3])
            embedding = embeddings[nfaces_]  # Векторное представление лица

            # Проверка на спуфинг (если включен режим антиспуфинга)
            ts = time()
            if antispoofingmode:
                is_real, antispoof_score = antispoof_model.analyze(img=frame, facial_area=(left, top, right - left, bottom - top))
            else:
                is_real, antispoof_score = True, 1
            ts = str(round((time() - ts) * 1000)).rjust(5)  # Время, затраченное на проверку спуфинга


            # Поиск в базе данных известных лиц
            tb = time()
            for item in known_vector:
                vector = item["vector"]
                item["distance"] = np.linalg.norm(vector - embedding)  # Вычисляем расстояние между векторами
            sorted_data = sorted(known_vector, key=lambda x: x["distance"])  # Сортируем по расстоянию

            if len(sorted_data) > 0:
                distance_first = sorted_data[0]['distance']
            else:
                distance_first = 2  # Если база данных пуста, устанавливаем расстояние 2

            # Выбираем три ближайших лица
            top_three = [{"namepercone": item["namepercone"], "distance": item["distance"], "photo": item["photo"]}
                         for item in sorted_data[:3] if item["distance"] < (1.1 if ffmode == 'AdaFace' else 1.05)]

            if len(top_three) == 0:  # Если лицо не распознано
                # Сохраняем неизвестное лицо
                unknown_distance = np.linalg.norm(prior_unknown_vector - embedding)
                if unknown_distance > 0.5:
                    namefile = 'output/' + dt.now().strftime('%Y%m%d') + '_unknown/' + timeframe + '_' + str(nfaces_ + 1) + '.jpeg'
                    #faces_.save(namefile, "JPEG", quality=100)
                    facessavejpeg(faces_, namefile)
                    prior_unknown_vector = embedding
                    New_face = True
                else:
                    New_face = False

                all_Ok = False
                additional_data: str = 'X/Z'
                percent, percent2 = 0, 0
                ev_data, ev_data2 = 2, 2
                distance_first = unknown_distance
                ev_data_unknown = unknown_distance*unknown_distance
                avatara = None
            else:  # Если лицо распознано
                all_Ok = True
                additional_data: str = top_three[0]["namepercone"]
                ev_data = top_three[0]["distance"]
                ev_data2 = ev_data * ev_data
                percent = percentage_fcb(ev_data)
                percent2 = percentage_fcb(ev_data2)
                avatara = top_three[0]["photo"]

                # Обработка имени
                if additional_data.find(' 19') > -1:
                    additional_data = additional_data[:additional_data.find(' 19')]
                if additional_data.find(' 20') > -1:
                    additional_data = additional_data[:additional_data.find(' 20')]

                # Сохраняем распознанное лицо
                if prior_additional_data != additional_data:
                    namefile = namefile = 'output/' + dt.now().strftime('%Y%m%d') + '_known/' + timeframe + '-' + additional_data.replace(' ', '_') + '.jpeg'
                    #faces_.save(namefile, "JPEG", quality=100)
                    facessavejpeg(faces_, namefile)
                    '''
                    from os.path import isfile as ospathisfile
                    namefile = 'known/' + additional_data.replace(' ', '_') + 'm.jpeg'
                    if not ospathisfile(namefile):
                        faces_.save(namefile, "JPEG", quality=100)
                    '''
                    prior_additional_data = additional_data
                    New_face = True
                else:
                    New_face = False
                unknown_distance = 0

            tb = str(round((time() - tb) * 1000)).rjust(5)  # Время, затраченное на поиск в базе данных

            # Отрисовка рамки вокруг лица
            square_size_fix = 360
            frame2 = SqrReize(frame.copy(), left, top, right, bottom, square_size_fix=square_size_fix)
            cv2.rectangle(frame2, (1, 1), (square_size_fix - 1, square_size_fix - 1), (0, 128, 0) if all_Ok else (0, 0, 128), 3)

            # Добавление текстуры и аватара
            img_tekstura_ = img_tekstura.copy()
            y, x, _ = img_tekstura_.shape
            x1, y1 = (x - square_size_fix) // 2, (y - square_size_fix) // 2 - 120
            img_tekstura_[y1:y1 + square_size_fix, x1:x1 + square_size_fix] = frame2
            if not avatara is None:
                img_tekstura_[y - 120:y - 8, x - 120:x - 8] = avatara

            # Добавление информации на изображение
            img_tekstura_ = DrawInfo(img_tekstura_, additional_data=additional_data, sys_info=f'{percent2:3.1f} {ev_data:.2f}', is_Real=is_real, all_Ok=all_Ok)
            cv2.setWindowTitle(FRAME_NAME, FRAME_NAME + f' {ev_data2:.2f}')
            cv2.imshow(FRAME_NAME, img_tekstura_)

        tr = str(round((time() - tr) * 1000)).rjust(5)  # Время, затраченное на обработку кадра
        cv2.waitKey(1)

        # Вывод информации в консоль
        percentage_fcbline1 = f'{percentage_fcbline(ev_data2) * 100:.2f}'.rjust(6)
        if not all_Ok:
            percentage_fcbline2 = f'{percentage_fcbline(ev_data_unknown) * 100:.2f}'.rjust(6)

        print(f' {i:>4} | {tf:>4} {tr:>4} {ts:>4} {tb:>4} |', right - left, bottom - top, '|',
              f'{f"{percent:3.1f}":>5} {f"{percent2:3.1f}":>5} {f"{min(ev_data, distance_first):.2f}":>5} {f"{ev_data2:.2f}":>5} | {percentage_fcbline1} {percentage_fcbline2 if not all_Ok else "      "} | {additional_data} {"New face" if New_face else ""}')

        appendTXT('output/' + dt.now().strftime('%Y%m%d')+'.txt',
                  f'{i:>5}|{timeframe:>22}|{percentage_fcbline1}|{"New face" if New_face else "        "}|{percentage_fcbline2 if not all_Ok else "      "}|{additional_data}\n')
        timedalaytitle = time()

    else:  # Если лица не обнаружены
        time_diff = time() - timedalaytitle
        print('.', end='', flush=True)
        if 3 < time_diff < 10:
            img_tekstura_ = DrawInfo(img_tekstura.copy(), additional_data='', is_Real=None, all_Ok=True)
            cv2.imshow(FRAME_NAME, img_tekstura_)
        if 10 < time_diff < 15:
            prior_additional_data = '-=!@#$%^&*()=-'
            prior_unknown_vector  = np.ones((1, vector_size), dtype=np.float32)  # Вектор для неизвестных лиц
        pass
    return

# Инициализация видеопотока
#STREAM_TO_PARSE = "S:\\Python24\\Video\\17_2.mp4"
#STREAM_TO_PARSE = "S:\Python24\Video\Anton.mp4"
#STREAM_TO_PARSE = "S:\\Python24\\Video\\2miVIDEO_20241003_135240918.mp4"
#STREAM_TO_PARSE = "S:\\Python24\\Video\\video_2024-10-02_16-19-53.mp4"
#STREAM_TO_PARSE = "S:\\Python24\\Video\\video_2024-10-03_13-03-41.mp4"
STREAM_TO_PARSE = 'S:\\#Video\\2.mp4'  # Путь к видеофайлу
cap = VideoFileReader(STREAM_TO_PARSE, interval_seconds=1)  # Чтение видео с интервалом 1 секунда
TIME_DELAY = 0  # Задержка между кадрами

#cap = VideoCapture(STREAM_TO_PARSE, namestream=NAME_STREAM, resize=RESIZE, crop_params=CROP_PARAMS)


# Инициализация переменных
i, t = 0, 0  # Счетчик кадров и время

# Настройка окна OpenCV
cv2.imshow(FRAME_NAME, img_tekstura)
cv2.namedWindow(FRAME_NAME, cv2.WINDOW_NORMAL)  # Открываем окно в полноэкранном режиме


'''
cv2.namedWindow(FRAME_NAME, cv2.WINDOW_NORMAL)
cv2.WINDOW_NORMAL: Позволяет изменять размер окна вручную.
cv2.WINDOW_AUTOSIZE: Окно автоматически подстраивается под размер изображения (по умолчанию).
cv2.WINDOW_FULLSCREEN: Открывает окно в полноэкранном режиме.
cv2.WINDOW_KEEPRATIO: Сохраняет пропорции изображения при изменении размера окна.
cv2.WINDOW_GUI_NORMAL: Окно с базовым интерфейсом (без дополнительных элементов).
cv2.WINDOW_GUI_EXPANDED: Окно с расширенным интерфейсом (включая панель инструментов).

cv2.setWindowTitle

'''



# Основной цикл обработки видео
while True:
    # Регулировка задержки между кадрами
    elapsed_time = time() - t  # Прошедшее время с последнего кадра
    sleep_duration = max(0, TIME_DELAY / 1000 - elapsed_time)  # Вычисляем время для sleep
    sleep(sleep_duration)  # Задержка
    t = time()  # Обновляем время

    # Чтение кадра
    frame = cap.read()
    if not cap.ok:  # Если видео закончилось, выходим из цикла
        break

    if frame is None:  # Если кадр пустой, пропускаем
        print('-', end='', flush=True)
        sleep(0.1)
        continue

    i += 1  # Увеличиваем счетчик кадров

    # Обработка кадра
    ReadFrame(frame.copy(), 0, dt.now().strftime('%Y%m%d-%H%M%S-%f'))

    # Выход по нажатию клавиши 'q'
    if chr(cv2.waitKey(1) & 255) == 'q':
        break
