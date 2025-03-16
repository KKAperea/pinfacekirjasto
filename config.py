CompName = {"name" : "Raspberry", "type" : "Raspberry Pi OS"}

WRITE_FRAME = True
CamConfig1 = {
#    "stream_to_parse" : "rtsp://user:KKK123kkk@192.168.111.210:554/ISAPI/Streaming/Channels/101",
    "stream_to_parse" : "rtsp://user:KKK123kkk@188.143.241.178:554/ISAPI/Streaming/Channels/101",
    "name": "Door",
    "resize" : 0.75,
    "crop_params" : [100, 800, 400, 1200],
    "timedelay" : 400,
    "extraqueue" : 0,
    "desc" : "Door to the office"
}

CamConfig3 = {
    "stream_to_parse" : "rtsp://user:KKK123kkk@192.168.111.212:554/ISAPI/Streaming/Channels/101",
    "name": "cam03",
    "resize" : 1,
    "timedelay" : 200,
    "extraqueue" : 0,
    "desc" : "Настольная камера"
}

# Выбор активной конфигурации камеры
#CamConfig = CamConfig3
CamConfig = CamConfig3


# Импорт необходимых функций из модуля os
from os import remove as osremove, mkdir as osmkdir
from os.path import isdir as ospathisdir, exists as ospathexists

# Извлечение информации о компьютере
COMP_NAME = CompName.get('name', 'compname - NoName')  # Имя компьютера
COMP_TYPE = CompName.get('type', 'comptype - NoOs')   # Тип операционной системы

# Извлечение параметров камеры
STREAM_TO_PARSE = CamConfig['stream_to_parse']  # URL потока камеры
TIME_DELAY = CamConfig.get('timedelay', 300)    # Задержка по времени (мс)
NAME_STREAM = CamConfig.get('name', 'NoName')   # Имя камеры
RESIZE = CamConfig.get('resize', None)          # Параметры изменения размера
CROP_PARAMS = CamConfig.get('crop_params', None)  # Параметры обрезки
DESC_CAM = CamConfig.get('desc', 'desc - NoName')  # Описание камеры
EXTRA_QUEUE = CamConfig.get('extraqueue', 1)    # Дополнительный кеш видеопотока

# Удаление переменной CamConfig для освобождения памяти
del CamConfig

# Формирование имени кадра
FRAME_NAME = COMP_NAME + ' ' + NAME_STREAM

# Создание директории для выходных данных
OUTPUT_DIR = f"{NAME_STREAM}_OutputStream"  # Имя директории на основе имени камеры

# Вывод информации о конфигурации
print(f'''[i] Имя компьютера = {COMP_NAME}; Тип компьютера = {COMP_TYPE}
[i] Имя потока = {DESC_CAM}; Задержка по времени = {TIME_DELAY} ms; {f'Кеш видеопотока = {EXTRA_QUEUE}' if EXTRA_QUEUE != 0 else ''}''')

# Создание основной директории, если она не существует
if not ospathisdir(OUTPUT_DIR):
    osmkdir(OUTPUT_DIR)

# Создание поддиректории для миниатюр, если она не существует
if not ospathisdir(f"{OUTPUT_DIR}/thumbnails"):
    osmkdir(f"{OUTPUT_DIR}/thumbnails")

