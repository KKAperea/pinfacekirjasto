import cv2
from cv2 import numpy as np
from datetime import datetime as dt



def percentage_fcbline(euclidean_distance_):
    if euclidean_distance_< 1.270588:
        ret = 0.22 * (1 - euclidean_distance_ /  1.270588) + 0.78 # math.sqrt(1.2)) + 0.79
    elif euclidean_distance_<1.341640786 :   #math.sqrt(1.8):
        ret = 0.78 * (1.341640786 - euclidean_distance_) / (1.341640786 - 1.270588)
    else:
        ret = 0
    return ret




#import numpy as np

from scipy.interpolate import interp1d
# Преобразование евклидова расстояния в проценты
EUCLIDEAN_DISTANCE_TO_PERCENT_STOP_INSIDE_FUNCTION_IF = 1.8  # Порог, при котором прекращается расчет процента схожести (hardcoded)
EUCLIDEAN_DISTANCE_TO_PERCENT_STOP = 1.8  # Порог, при котором прекращается расчет процента схожести (1.7 == 0% схожести)
EUCLIDEAN_DISTANCE_TO_PERCENT_NUM = 18  # Количество точек для интерполяции
EUCLIDEAN_DISTANCE_TO_PERCENT_X = np.linspace(0, EUCLIDEAN_DISTANCE_TO_PERCENT_STOP, num=EUCLIDEAN_DISTANCE_TO_PERCENT_NUM, endpoint=True)  # Ось X для интерполяции
EUCLIDEAN_DISTANCE_TO_PERCENT_Y = np.array([
    1.0,  # 0
    0.99,  # 0.1
    0.95,  # 0.2
    0.92,  # 0.3
    0.90,  # 0.4
    0.86,  # 0.5
    0.84,  # 0.6
    0.83,  # 0.7
    0.82,  # 0.8
    0.81,  # 0.9
    0.80,  # 1.0
    0.79,  # 1.1
    0.78,  # 1.2
    0.70,  # 1.3
    0.6,   # 1.4
    0.3,   # 1.5
    0.1,   # 1.6
    0,     # 1.7
])  # Ось Y для интерполяции (проценты схожести)

# Справочная информация по настройкам MTCNN
# Если не детектируются лица на групповых фото или есть артефакты, можно изменить настройки MTCNN:
# Внимание: изменение self.factor может значительно увеличить время детекции.
# Для улучшения детекции на групповых фото рекомендуется вырезать лица отдельными файлами.

# Интересные настройки MTCNN:
# Для настройки чувствительности MTCNN (детекция лиц) отредактируйте файл ./face_alignment/mtcnn.py:
# Дефолтные настройки:
# self.min_face_size = 20
# self.thresholds = [0.6, 0.7, 0.9]
# self.nms_thresholds = [0.7, 0.7, 0.7]
# self.factor = 0.85

# Альтернативные настройки (могут быть ложные срабатывания):
# self.min_face_size = 20
# self.thresholds = [0.5, 0.5, 0.5]
# self.nms_thresholds = [0.7, 0.7, 0.7]
# self.factor = 0.85


def percentage_fcb(euclidean_distance_):
    if 0 <= euclidean_distance_ <= EUCLIDEAN_DISTANCE_TO_PERCENT_STOP_INSIDE_FUNCTION_IF:
        ret = interp1d(EUCLIDEAN_DISTANCE_TO_PERCENT_X, EUCLIDEAN_DISTANCE_TO_PERCENT_Y)(euclidean_distance_) * 100
    else:
        ret = 0
    return ret


#--------------------------- Вычисление центров
def cv2putText(frame,
               current_text: str = '',
               alignment: str = 'center',
               indent: int = 15,
               bottom: int = 50,
               font: int = cv2.FONT_HERSHEY_SIMPLEX,
               font_scale: float = 1.1,
               thickness: int = 2,
               color: tuple = (255, 255, 255)
               ):
    """
    Добавляет текст на изображение (frame) с указанным выравниванием.

    Параметры:
    - frame: Изображение (numpy.ndarray), на которое будет добавлен текст.
    - current_text: Текст для добавления. Если пустая строка, функция возвращает исходное изображение.
    - alignment: Выравнивание текста. Допустимые значения: 'center', 'left', 'right'.
    - indent: Отступ от края изображения (в пикселях) для выравнивания текста.
    - bottom: Отступ от нижнего края изображения (в пикселях) для размещения текста.
    - font: Шрифт текста (по умолчанию cv2.FONT_HERSHEY_SIMPLEX).
    - font_scale: Масштаб шрифта (по умолчанию 1.1).
    - thickness: Толщина линии текста (по умолчанию 2).
    - color: Цвет текста в формате BGR (по умолчанию белый (255, 255, 255)).

    Возвращает:
    - Изображение с добавленным текстом (numpy.ndarray).
    """

    # Если текст пустой, возвращаем исходное изображение
    if current_text == '':
        return frame

    # Проверяем, что выбранное выравнивание допустимо
    assert alignment in ['center', 'left', 'right'], f"Недопустимое значение alignment: {alignment}."

    # Получаем размеры изображения
    height, width, _ = frame.shape

    # Вычисляем размеры текста
    (text_width, text_height), _ = cv2.getTextSize(current_text, font, font_scale, thickness)

    # Определяем отступ (margin) в зависимости от выравнивания
    if alignment == 'center':
        # Выравнивание по центру: отступ рассчитывается как половина ширины изображения минус половина ширины текста
        margin = (width - text_width) // 2 - indent
    elif alignment == 'left':
        # Выравнивание по левому краю: отступ равен значению indent
        margin = indent
    elif alignment == 'right':
        # Выравнивание по правому краю: отступ равен ширине изображения минус ширина текста и indent
        margin = width - text_width - indent

    # Добавляем текст на изображение
    cv2.putText(frame, current_text, (margin, bottom), font, font_scale, color, thickness)

    # Возвращаем изображение с добавленным текстом
    return frame

from PIL import Image, ImageDraw, ImageFont
def image2putText(draw,
                  current_text: str = '',
                  alignment: str = 'center',
                  indent: int = 15,
                  bottom: int = 50,
                  font_path: str = "arial.ttf",
                  font_size: int = 30,
                  color: tuple = (255, 255, 255),
                  width: int = 480):
    """
    Добавляет текст на изображение с использованием объекта `draw` (PIL.ImageDraw.Draw).

    Параметры:
    - draw: Объект `PIL.ImageDraw.Draw`, используемый для рисования на изображении.
    - current_text: Текст для добавления. Если пустая строка, функция ничего не делает.
    - alignment: Выравнивание текста. Допустимые значения: 'center', 'left', 'right'.
    - indent: Отступ от края изображения (в пикселях) для выравнивания текста.
    - bottom: Отступ от верхнего края изображения (в пикселях) для размещения текста.
    - font_path: Путь к файлу шрифта (по умолчанию "arial.ttf").
    - font_size: Размер шрифта (по умолчанию 30).
    - color: Цвет текста в формате RGB (по умолчанию белый (255, 255, 255)).
    - width: Ширина изображения (по умолчанию 480 пикселей).

    Возвращает:
    - None (функция изменяет переданный объект `draw`).
    """
    # Если текст пустой, завершаем выполнение функции
    if current_text == '':
        return

    # Проверяем, что выбранное выравнивание допустимо
    assert alignment in ['center', 'left', 'right'], f"Недопустимое значение alignment: {alignment}."

    width, height = draw.im.size

    # Загружаем шрифт
    font = ImageFont.truetype(font_path, font_size)

    # Вычисляем ширину текста
    text_width = draw.textlength(current_text, font=font)

    # Определяем отступ (margin) в зависимости от выравнивания
    if alignment == 'center':
        # Выравнивание по центру: отступ рассчитывается как половина ширины изображения минус половина ширины текста
        margin = (width - text_width) // 2 - indent
    elif alignment == 'left':
        # Выравнивание по левому краю: отступ равен значению indent
        margin = indent
    elif alignment == 'right':
        # Выравнивание по правому краю: отступ равен ширине изображения минус ширина текста и indent
        margin = width - text_width - indent

    # Добавляем текст на изображение
    draw.text((margin, bottom), current_text, font=font, fill=color)

def find_nearest_space(text):
    # Находим середину строки
    middle = len(text) // 2

    # Ищем пробел слева от середины
    left_space = text.rfind(" ", 0, middle)

    # Ищем пробел справа от середины
    right_space = text.find(" ", middle)

    # Выбираем ближайший пробел
    if left_space == -1 and right_space == -1:
        return -1  # Пробел не найден
    elif left_space == -1:
        return right_space  # Только правый пробел
    elif right_space == -1:
        return left_space  # Только левый пробел
    else:
        # Сравниваем расстояния до середины
        if abs(middle - left_space) <= abs(middle - right_space):
            return left_space
        else:
            return right_space

def DrawInfo(frame, additional_data: str = "Дополнительные данные", sys_info: str = '', is_Real: bool = True, all_Ok: bool = True):
    """
    Добавляет информацию на изображение (frame), включая дату, время, системную информацию и статусы.

    Параметры:
    - frame: Изображение (numpy.ndarray), на которое будет добавлена информация.
    - additional_data: Дополнительные данные (по умолчанию "Дополнительные данные").
    - sys_info: Системная информация (по умолчанию пустая строка).
    - is_Real: Флаг, указывающий на реальность данных (по умолчанию True).
    - all_Ok: Флаг, указывающий на успешность операции (по умолчанию True).

    Возвращает:
    - Изображение с добавленной информацией (numpy.ndarray).
    """

    # Получаем текущую дату и время
    current_date = dt.now().strftime("%d.%m.%Y")  # Форматируем дату в виде "день.месяц.год"
    current_time = dt.now().strftime("%H:%M:%S")  # Форматируем время в виде "часы:минуты:секунды"

    # Настройки шрифта и цвета для текста
    font = cv2.FONT_HERSHEY_SIMPLEX  # Шрифт
    font_scale = 1.1  # Масштаб шрифта
    color = (255, 255, 255)  # Цвет текста (белый)
    thickness = 2  # Толщина линии текста

    # Добавляем текущую дату в левый верхний угол изображения
    cv2putText(frame, current_text=current_date, alignment='left', indent=15, bottom=50,
               font=font, font_scale=font_scale, color=(128, 128, 128), thickness=thickness)

    # Добавляем текущее время в правый верхний угол изображения
    cv2putText(frame, current_text=current_time, alignment='right', indent=15, bottom=50,
               font=font, font_scale=font_scale, color=(128, 128, 128), thickness=thickness)

    # Добавляем информацию о статусе is_Real (реальность данных)
    if is_Real is not None:  # Проверяем, что is_Real не равно None
        if not is_Real:
            # Если is_Real равно False, добавляем текст "False" красного цвета
            cv2putText(frame, current_text='False', alignment='center', indent =0, bottom=500, color=(0, 0, 255))
            # Создаем копию изображения для наложения полупрозрачной рамки
            overlay = frame.copy()
            # Рисуем полупрозрачный прямоугольник
            #cv2.rectangle(overlay, top_left, bottom_right, box_color, -1)  # -1 означает заливку
            cv2.rectangle(overlay, (frame.shape[1] // 2 - 120, 470), (frame.shape[1] // 2 + 120, 510), (0,0,255), -1)  # -1 означает заливку
            #cv2.rectangle(overlay, (10,40), (30,50), (0,0,255), -1)  # -1 означает заливку
            # Наложение рамки на изображение с учетом прозрачности
            box_alpha = 0.3
            cv2.addWeighted(overlay, box_alpha, frame, 1 - box_alpha, 0, frame)
        else:
            # Если is_Real равно True, добавляем текст "True" зеленого цвета
            cv2putText(frame, current_text='True', alignment='center', indent = 0, bottom=500, color=(0, 255, 0))

    # Добавляем информацию о статусе all_Ok (успешность операции)
    if not all_Ok:
        # Если all_Ok равно False, добавляем текст "Not Found" красного цвета
        cv2putText(frame, current_text='Not Found', alignment='center', indent = 0, bottom=550, color=(0, 0, 255))

    # Добавляем системную информацию, если она не пустая
    if sys_info != '':
        cv2putText(frame, current_text=sys_info, alignment='center', indent = 0, bottom=780,
                   font=font, font_scale=font_scale - 0.3, color=(128, 128, 128), thickness=thickness)

    # rus ---------------------------------------------------------------------------
    # Преобразуем изображение из OpenCV в Pillow
    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Создаём объект для рисования
    draw = ImageDraw.Draw(image_pil)

    # Указываем шрифт (замените путь на ваш шрифт .ttf)
    font_path = "arial.ttf"  # Пример: шрифт Arial
    font_size = 30
    color = (255, 255, 255)  # Цвет текста (белый)

    # Добавляем дополнительные данные, если они не пустые
    if additional_data != '':
        if additional_data.find(' ') == -1:
            # Если текст не содержит пробелов, добавляем его одной строкой
            image2putText(draw=draw, current_text=additional_data, alignment='center', indent=0,
                          bottom=610, font_path=font_path, font_size=font_size, color=color)
        else:
            # Если текст содержит пробелы, разбиваем его на две строки
            nearest_space = find_nearest_space(additional_data)  # Находим ближайший пробел
            additional_data1 = additional_data[:nearest_space]  # Первая часть текста
            additional_data2 = additional_data[nearest_space + 1:]  # Вторая часть текста

            # Добавляем первую часть текста
            image2putText(draw=draw, current_text=additional_data1, alignment='center', indent=0,
                          bottom=610, font_path=font_path, font_size=font_size, color=color)
            # Добавляем вторую часть текста
            image2putText(draw=draw, current_text=additional_data2, alignment='center', indent=0,
                          bottom=650, font_path=font_path, font_size=font_size, color=color)

    # Преобразуем изображение обратно в формат OpenCV
    frame = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

    # Возвращаем изображение с добавленной информацией
    return frame


def SqrReize(frame, left, top, right, bottom, square_size_fix = 300):

    if (right-left) % 2 == 1: left =left - 1
    if (bottom-top) % 2 == 1: top  =top  - 1

    # Вычисляем центр области
    center_y = (top + bottom) // 2
    center_x = (left + right) // 2

    # Размер квадрата (до масштабирования)
    square_size = max(bottom - top, right - left)  # Размер квадрата равен максимальной стороне

    # Вычисляем координаты для квадрата
    start_y = center_y - square_size // 2 - 30
    end_y = center_y + square_size // 2 + 30
    start_x = center_x - square_size // 2 - 30
    end_x = center_x + square_size // 2 + 30

    # Проверяем, выходит ли квадрат за пределы кадра
    pad_top = max(0, -start_y)
    pad_bottom = max(0, end_y - frame.shape[0])
    pad_left = max(0, -start_x)
    pad_right = max(0, end_x - frame.shape[1])

    # Обрезаем кадр до квадрата (с учётом выхода за пределы)
    start_y = max(0, start_y)
    end_y = min(frame.shape[0], end_y)
    start_x = max(0, start_x)
    end_x = min(frame.shape[1], end_x)

    frame2 = frame[start_y:end_y, start_x:end_x]

    # Добавляем padding, если квадрат выходит за пределы кадра
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        frame2 = np.pad(frame2, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')

    # Масштабируем квадрат до 300x300
    frame2_resized = cv2.resize(frame2, (square_size_fix, square_size_fix))
    del frame2
    return frame2_resized

def facessavejpeg(faces_, namefile):
    try:
        faces_.save(namefile, "JPEG", quality=100)
    except:
        # Путь к файлу
        from pathlib import Path
        file_path = Path(namefile)
        # Создание всех недостающих каталогов
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # Создание файла
        try:
            faces_.save(namefile, "JPEG", quality=100)
        except:
            print('[!] Не могу создать файл ' +namefile)


def appendTXT(namefile, text):
    with open(namefile, "a+", encoding = "windows-1251", errors='ignore') as fileA:
        try:
            fileA.write(text)
        except:
            pass
