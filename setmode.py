'''
Режимы работы для распознавания лиц и обработки изображений
ffmode: Режим для извлечения признаков лица (Face Feature Extraction Mode)
frmode: Режим для распознавания лиц (Facial Recognition Mode)
antispoofingmode: Режим антиспуфинга (проверка на подделку лица)

       Параметры:
        - ffmode: Режим обнаружения лиц. Может быть или/и 'opencv', 'adaface'
        - frmode: Режим распознавания лиц. Может быть 'sface' или/и 'mtcnn'.
'''
ffmode = 'opencv'  # Режим извлечения признаков лица (по умолчанию 'opencv')
#ffmode = 'adaface'  # Режим извлечения признаков лица (по умолчанию 'opencv')
frmode = 'sface'  # Режим распознавания лиц (по умолчанию 'sface')
#frmode = 'mtcnn'

antispoofingmode = False  # Отключение антиспуфинга по умолчанию


# 'adaface'
# 'mediapipe'


# Определение размера вектора признаков в зависимости от режима распознавания
if frmode == 'sface':
    vector_size = 128  # Размер вектора для режима 'opencv'
elif frmode == 'adaface':
    vector_size = 512  # Размер вектора для режима 'adaface'

# Вывод информации о выбранных режимах
#print(f'[i] Face_detection       = {ffmode}', flush=True)  # Режим детекции лиц
#print(f'[i] Facial_recognition   = {frmode}', flush=True)  # Режим распознавания лиц

# Основной каталог для обработки изображений
pathinput = 'input'  # Основная директория для входных данных

# Список подкаталогов для создания
paths = [
    pathinput,  # Основная директория
    pathinput + '/ok',        # Для успешно обработанных изображений
    pathinput + '/error',     # Для изображений с ошибками
    pathinput + '/empty',     # Для изображений без лиц
    pathinput + '/many',      # Для изображений с несколькими лицами
    pathinput + '/double',    # Для изображений с дубликатами
    pathinput + '/blacklist', # Для изображений в черном списке
    'unknown',  # Для неизвестных лиц
    'known',    # Для известных лиц
]

# Импорт функции для создания директорий
from os import makedirs

# Создание директорий, если они не существуют
for path_ in paths:
    makedirs(path_, exist_ok=True)  # Создание директории, если она не существует


'''
if ffmode == 'AdaFace':
    from face_alignment import align
    from AdaFace import inference
    model = inference.load_pretrained_model()
elif ffmode == 'opencv':
    from SFace import FaceRecognizerSface
    import OpenCv
    FaceRecognizer = FaceRecognizerSface()
'''

if antispoofingmode:
    print(f'[o] Load antispoofing...\r', flush=True, end='')
    from antispoofing import AntiSpoofing as FasNet
    antispoof_model = FasNet.Fasnet()
    print(f'[+] Antispoofing ON         ', flush=True)
else:
    print(f'[-] Antispoofing OFF', flush=True)
