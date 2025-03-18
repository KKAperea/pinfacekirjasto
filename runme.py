# Вывод сообщения о начале инициализации
print(f'[i] -= Версия 1.02 =-', flush=True)

# Импортируем необходимые модули
from setmode import *  # Модуль для настройки режимов (ffmode, frmode и т.д.)
from pinfacekirjasto.PinFace import PinFace, cv2  # Модуль для работы с лицами
from cv2 import numpy as np  # Импортируем NumPy через OpenCV
from os import name as osname
from time import time, sleep  # Модуль для работы со временем

# Импортируем дополнительные модули
from runmelib import dt, percentage_fcbline, percentage_fcb, DrawInfo, SqrReize,facessavejpeg, appendTXT, fcc_data # Вспомогательные функции
from camstream import VideoCapture, VideoFileReader  # Модуль для работы с видеопотоком
import face_data  # Модуль для работы с базой данных лиц


#import subprocess

frame = None
relay_line = None
try:
    if osname != 'nt':
        import gpiod
        # Настройка GPIO
        CHIP = '/dev/gpiochip0'  # Имя чипа GPIO (обычно 'gpiochip0' для Orange Pi)
        RELAY_PIN = 2      # В gpio readall это графа 'name' имя пина PL02

        # Задержки включения и отключения реле (в секундах)
        DELAY_ON = 0.5  # Задержка включения реле
        DELAY_OFF = 10  # Задержка отключения реле

        #try:
            # Инициализация GPIO
        chip = gpiod.Chip(CHIP)
        print(f"[i] Открыт чип GPIO: {CHIP}")

        # Проверка доступности пина
        try:
            relay_line = chip.get_line(RELAY_PIN)
            print(f"[+] Пин {RELAY_PIN} доступен")
        except Exception as e:
            print(f"[x] Пин {RELAY_PIN} недоступен: {e}")
            relay_line = None

        if relay_line != None:
            # Настройка пина как выходного
            relay_line.request(consumer="RELAY_CONTROL", type=gpiod.LINE_REQ_DIR_OUT)
            print(f"[i] Пин {RELAY_PIN} настроен как выход")
        #except Exception as e:
        #    relay_line = None
        #    print(f"[x] Ошибка чипа GPIO: {e}")

        def relay_line_open():
            relay_line.set_value(1)

            namefile = 'output/' + dt.now().strftime('%Y%m%d') + '_raid/' + dt.now().strftime('%Y%m%d-%H%M%S') + '.jpeg'
            facessavejpeg(frame.copy(), namefile)

            print("[i] Реле включено")
            sleep(DELAY_ON)
            relay_line.set_value(0)
            print("[i] Реле выключено")
            sleep(3)  # Реле выключено на заданное время

except:
    relay_line = None
    pass


if relay_line is None:
    print("[-] Замок недоступен")

RFIDtag = None

try:
    if osname != 'nt':
        import serial, threading

        # Открытие последовательного порта с тайм-аутом 1 секунда
        ser = serial.Serial('/dev/ttyS3', 9600, timeout=1)
        print("[+] Последовательный порт - открыт")

        def read_serial():
            """Чтение данных с последовательного порта"""
            print("[+] Чтение - запущено")
            last_execution_time = 0  # Время последнего выполнения
            rfid_tag_prior = ''  # Предыдущий RFID-ключ

            while True:
                data = ser.read(14)  # Чтение 14 байт
                if data:  # Проверка, что данные получены
                    if data[0] == 0x02 and data[13] == 0x03:  # Проверка стартового и стопового байта
                        rfid_tag = data[1:11].hex()  # Извлечение данных метки
                        ser.reset_input_buffer()  # Очистка входного буфера порта

                        current_time = time()  # Текущее время

                        # Если прошло больше 7 секунд с последнего выполнения или ключ изменился
                        if (current_time - last_execution_time >= 7) or (rfid_tag != rfid_tag_prior):
                            print("\n[i] RFID Tag:", rfid_tag)  # Вывод метки
                            if relay_line is not None:
                                if rfid_tag in ['30323030333332304335', '30323030333332343332', '30323030333443354642', '30323030333445353034']:
                                    relay_line_open()  # Выполняем функцию
                                else:
                                    print(f"[!] Пропуск RFID Tag: {rfid_tag} запрещен")
                            last_execution_time = current_time  # Обновляем время последнего выполнения
                        else:
                            print(f"[-] Пропуск RFID Tag: {rfid_tag} (выполнено менее 7 секунд назад)")

                        rfid_tag_prior = rfid_tag  # Обновляем предыдущий RFID-ключ

                sleep(0.5)  # Небольшая задержка для снижения нагрузки на CPU


        # Запуск потока для чтения данных с порта
        serial_thread = threading.Thread(target=read_serial, daemon=True)
        serial_thread.start()
        RFIDtag = True
except:
    RFIDtag = None
    pass


if RFIDtag is None:
    print("[-] RFID метки недоступны")

# import numpy as np  # Альтернативный импорт NumPy

# Инициализация объекта для работы с лицами
pFace = PinFace(ffmode=ffmode, frmode=frmode,fmode =[])
'''
Параметры:
        - ffmode: Режим обнаружения лиц. Может быть 'opencv', 'mtcnn','mediapipe'
        - frmode: Режим распознавания лиц. Может быть 'sface', 'adaface'
'''

# Загружаем известные лица из базы данных
_, known_vector = face_data.load_faces_from_db()

# Импортируем конфигурационные параметры
from config import (
    OUTPUT_DIR, TIME_DELAY, STREAM_TO_PARSE, NAME_STREAM, RESIZE, CROP_PARAMS,
    DESC_CAM, EXTRA_QUEUE, COMP_NAME, COMP_TYPE, FRAME_NAME, WRITE_FRAME
)

# Глобальные переменные
timedalaytitle = time()  # Время последнего обновления заголовка окна
prior_additional_data = '-=!@#$%^&*()=-'  # Предыдущие дополнительные данные
prior_unknown_vector = np.ones((1, vector_size), dtype=np.float32)  # Вектор для неизвестных лиц


# Загрузка текстуры для фона
try:
#    img_tekstura = cv2.imread('tekstura780580.jpg')  # Пытаемся загрузить текстуру
#    img_tekstura = cv2.imread('tekstura800600.jpg')  # Пытаемся загрузить текстуру
    img_tekstura = cv2.imread('tekstura848480.jpg')  # Пытаемся загрузить текстуру
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
        '''
        if osname != 'nt':
            try:
                # Используем команду xrandr для включения монитора
                subprocess.run(["xrandr", "--output", "HDMI-1", "--auto"], check=True)
                #print("Монитор включён с помощью xrandr.")
            except subprocess.CalledProcessError as e:
                pass
        '''

        tr = time()  # Засекаем время начала распознавания
        embeddings = pFace.face_recognition(faces=faces, facescv=facescv)  # Распознавание лиц
        # Обработка каждого обнаруженного лица

        for nfaces_, faces_ in enumerate(faces):

            # Добавление текстуры и аватара
            img_tekstura_ = img_tekstura.copy()

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
            known_vector_filtered = []
            for item in known_vector:
                vector = item["vector"]
                dist = np.linalg.norm(vector - embedding)  # Вычисляем расстояние между векторами
                #item["distance"] = dist
                if dist < 1.05:
                    known_vector_filtered.append({"distance" : dist, "namepercone": item["namepercone"], "photo": item["photo"]})

            #sorted_data = sorted(known_vector, key=lambda x: x["distance"])  # Сортируем по расстоянию
            sorted_data = sorted(known_vector_filtered, key=lambda x: x["distance"])  # Сортируем по расстоянию


            if len(sorted_data) > 0:
                distance_first = sorted_data[0]['distance']
            else:
                distance_first = 2  # Если база данных пуста, устанавливаем расстояние 2

            # Выбираем три ближайших лица
            top_three = [{"namepercone": item["namepercone"], "distance": item["distance"], "photo": item["photo"]}
#                         for item in sorted_data[:3] if item["distance"] < (1.1 if ffmode == 'AdaFace' else 1.05)]
                         for item in sorted_data[:3] if item["distance"] < 1.05]

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
            frame2 = SqrReize(frame.copy(), left, top, right, bottom, square_size_fix=fcc_data['mainpicture']['size'])
            cv2.rectangle(frame2, (3, 3), (fcc_data['mainpicture']['size'] - 3, fcc_data['mainpicture']['size'] - 3),
                          (0, 128, 0) if all_Ok else (0, 0, 128), 8)

            y, x, _ = img_tekstura_.shape


            img_tekstura_[fcc_data['mainpicture']['verticalindent']:fcc_data['mainpicture']['verticalindent'] + fcc_data['mainpicture']['size'],
                          fcc_data['mainpicture']['x1']:fcc_data['mainpicture']['x1'] + fcc_data['mainpicture']['size']] = frame2
            if not avatara is None:
                img_tekstura_[y - 120:y - 8, x - 120:x - 8] = avatara

            # Добавление информации на изображение
            img_tekstura_ = DrawInfo(img_tekstura_, additional_data=additional_data, sys_info=f'{percent2:3.1f} {ev_data:.2f}', is_real=is_real, all_Ok=all_Ok)
            cv2.setWindowTitle(FRAME_NAME, FRAME_NAME + f' {ev_data2:.2f}')
            cv2.imshow(FRAME_NAME, img_tekstura_)
            cv2.waitKey(1)



            if not relay_line is None:
                if all_Ok:
                    if is_real:
                        relay_line_open()

        tr = str(round((time() - tr) * 1000)).rjust(5)  # Время, затраченное на обработку кадра


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
        '''
        additional_data = ''
        percent2 = 0
        ev_data = 2
        is_real = True
        all_Ok = False
        ev_data2 = 0
        '''
        print('.', end='', flush=True)
        if 3 < time_diff < 20:
            img_tekstura_ = DrawInfo(img_tekstura.copy(), onlytime=True)
            cv2.imshow(FRAME_NAME, img_tekstura_)
        if 10 < time_diff < 25:
            prior_additional_data = '-=!@#$%^&*()=-'
            prior_unknown_vector  = np.ones((1, vector_size), dtype=np.float32)  # Вектор для неизвестных лиц

    return

# Инициализация видеопотока
#STREAM_TO_PARSE = "S:\\Python24\\Video\\17_2.mp4"
#STREAM_TO_PARSE = "S:\Python24\Video\Anton.mp4"
#STREAM_TO_PARSE = "S:\\Python24\\Video\\2miVIDEO_20241003_135240918.mp4"
#STREAM_TO_PARSE = "S:\\Python24\\Video\\video_2024-10-02_16-19-53.mp4"
#STREAM_TO_PARSE = "S:\\Python24\\Video\\video_2024-10-03_13-03-41.mp4"

STREAM_TO_PARSE = '2.mp4'  # Путь к видеофайлу
cap = VideoFileReader(STREAM_TO_PARSE, interval_seconds=1)  # Чтение видео с интервалом 1 секунда
TIME_DELAY = 0  # Задержка между кадрами

#cap = VideoCapture(1, namestream=NAME_STREAM, resize=RESIZE, crop_params=CROP_PARAMS)
#cap = VideoCapture(STREAM_TO_PARSE, namestream=NAME_STREAM)


# Инициализация переменных
i, t = 0, 0  # Счетчик кадров и время

# Настройка окна OpenCV
cv2.namedWindow(FRAME_NAME, cv2.WINDOW_GUI_NORMAL)  # Открываем окно в полноэкранном режиме
cv2.moveWindow(FRAME_NAME, 1, 1)  # Окно 1 на (100, 100)
cv2.imshow(FRAME_NAME, img_tekstura)
#cv2.resizeWindow(FRAME_NAME,  img_tekstura.shape[1]-50, img_tekstura.shape[0]-50)
if osname != 'nt':
    cv2.setWindowProperty(FRAME_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

'''
img_tekstura_ = DrawInfo(img_tekstura.copy(),
                         additional_data='Пробная заставка для проверки',
                         sys_info= 'Системная панель',
                         is_real=False,
                         all_Ok=True)
cv2.waitKey(1)
'''


'''
Если вам нужно окно без рамок, но не в полноэкранном режиме, можно использовать флаг cv2.WINDOW_GUI_NORMAL.
Если вам нужно окно, которое автоматически подстраивается под размер изображения, используйте флаг cv2.WINDOW_AUTOSIZE.
Если вам нужно сохранить пропорции изображения при изменении размера окна, используйте флаг cv2.WINDOW_KEEPRATIO.
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
    #print(f's = {int(sleep_duration)}',end = '')
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

if not relay_line is None:
    relay_line.release()
    print("[i] Ресурсы GPIO освобождены")
