# ver 0.01
import queue as Queue
import cv2
from  threading import Lock, Thread
from time import sleep
from datetime import datetime as dt
from typing import Generator

class VideoFileReader:
    """
    Класс для чтения кадров из видеофайла через заданные интервалы времени.
    """

    def __init__(self, name: str, namestream='NoName', resize = None, crop_params = None, interval_seconds: float = 0.1):
        """
        Инициализация объекта для чтения видео.

        :param video_path: Путь к видеофайлу.
        """
        self.name = name
        self.interval_seconds = interval_seconds
        self.cap = cv2.VideoCapture(name)
        self.resize = resize
        self.crop_params = crop_params
        self.ok = True

        if not self.cap.isOpened():
            raise ValueError(f"Ошибка: не удалось открыть видеофайл {name}.")

        # Читаем кадр
        ret, frame = self.cap.read()

        print("[+] Первый кадр успешно прочитан.")
        self.total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.resizeY, self.resizeX, _  = frame.shape
        self.resizewinY, self.resizewinX = self.resizeY, self.resizeX

        # Обработка параметров изменения масштаба,  resize - может быть кортежем, или числом

        if resize is not None:
            if isinstance(resize, (list, tuple)):
                self.resizeX, self.resizeY = resize
                resize = self.resizeY / frame.shape[0]
            else:
                self.resizeY = int(self.resizeY * resize) if resize > 0 else -int(self.resizeY / resize)
                self.resizeX = int(self.resizeX * resize) if resize > 0 else -int(self.resizeX / resize)
            self.resizewinY, self.resizewinX = self.resizeY, self.resizeX

            # Обработка параметров обрезки кадра
            if crop_params is not None:
                self.crop_params = [
                    int(ii * resize) if resize > 0 else int(-ii / resize)
                    for ii in crop_params
                ]
                self.resizewinY = self.crop_params[1] - self.crop_params[0]
                self.resizewinX = self.crop_params[3] - self.crop_params[2]

        # Получаем FPS видео
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            raise ValueError("FPS видео равно 0. Проверьте корректность видеофайла.")

        print(f"FPS видео: {self.fps}")

        # Устанавливаем начальное время
        self.current_time = 0

    def reset(self, frame):
        """
        Метод для изменения масштаба и обрезки кадра.

        :param frame: Исходный кадр.
        :return: Обработанный кадр.
        """
        if frame is None:
            return None

        # Изменение масштаба кадра
        if not self.resize is None:
            frame = cv2.resize(frame, (self.resizeX,self.resizeY),interpolation = cv2.INTER_AREA)

        # Обрезка кадра, если заданы параметры обрезки frame[100:800, 400:1200]
        if not self.crop_params is None:
            y1, y2, x1, x2 = self.crop_params
            frame = frame[y1:y2, x1:x2]
        return frame


    def read(self):
        """

        :param interval_seconds: Интервал времени в секундах между кадрами.
        """
        # Преобразуем интервал времени в миллисекунды
        interval_ms = int(self.interval_seconds * 1000)
        # Устанавливаем текущую временную метку
        self.cap.set(cv2.CAP_PROP_POS_MSEC, self.current_time)
        # Читаем кадр
        ret, frame = self.cap.read()


        # Если кадры закончились, выходим из цикла
        if not ret:
            # Получаем текущий номер кадра
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.ok = current_frame < self.total_frames
            return None

        # Увеличиваем текущее время на интервал
        self.current_time += interval_ms

        # Возвращаем кадр
        return self.reset(frame)


    def stop(self):
        """
        Освобождает ресурсы, связанные с видео.
        """
        if self.cap.isOpened():
            self.cap.release()
            print("Ресурсы видео освобождены.")

    def __del__(self):
        """
        Деструктор для автоматического освобождения ресурсов.
        """
        self.stop()


# Класс для буферизованного видеозахвата
class VideoCapture:

    def __init__(self, name, namestream='NoName', resize = None, crop_params = None):
        """
        Инициализация видеозахвата.

        :param name: Источник видеозахвата (например, путь к файлу или индекс камеры).
        :param namestream: Имя потока (по умолчанию 'NoName').
        :param resize: изменение маштаба кадра
        :param crop_params: Параметры обрезки кадра в формате (y1, y2, x1, x2).
                           Если None, обрезка не выполняется.
        """
        # Параметры для повторных попыток инициализации видеопотока
        self.max_retries = 10  # Максимальное количество попыток
        self.retry_delay = 5  # Задержка между попытками (в секундах)
        self.retry_count = 0  # Счетчик попыток

        self.namestream = namestream
        self.resize = resize
        self.crop_params = crop_params
        self.name = name
        self.ok = True



        # Инициализация видеозахвата
        self.cap = cv2.VideoCapture(self.name)
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,15000)  # Тайм-аут 15 секунд

        print(f"[i] Инициализация видеопотока: {namestream}")

        # Проверка успешности открытия видеозахвата
        if not self.cap.isOpened():
            print("[e] Ошибка: Не удалось открыть видеозахват.")
            self.stop()
            return

        print("[+] Видеозахват успешно открыт.")
        #self.cap.set(cv2.CAP_PROP_POS_FRAMES, 5000)


        # Чтение первого кадра
        self.ret, frame = self.cap.read()
        if not self.ret:
            print("[e] Ошибка: Не удалось прочитать кадр.")
            self.frame = None
            self.stop()
            # exit(1)  # Выход из программы
            return



        self.resizeY, self.resizeX, _  = frame.shape
        print("[+] Первый кадр успешно прочитан. ",self.resizeY, 'x',self.resizeX)
        self.resizewinY, self.resizewinX = self.resizeY, self.resizeX

        # Обработка параметров изменения масштаба,  resize - может быть кортежем, или числом

        if resize is not None:
            if isinstance(resize, (list, tuple)):
                self.resizeX, self.resizeY = resize
                resize = self.resizeY / frame.shape[0]
            else:
                self.resizeY = int(self.resizeY * resize) if resize > 0 else -int(self.resizeY / resize)
                self.resizeX = int(self.resizeX * resize) if resize > 0 else -int(self.resizeX / resize)
            self.resizewinY, self.resizewinX = self.resizeY, self.resizeX

            # Обработка параметров обрезки кадра
            if crop_params is not None:
                self.crop_params = [int(ii * resize) if resize > 0 else int(-ii / resize)
                                    for ii in crop_params]
        if crop_params is not None:
            self.resizewinY = self.crop_params[1] - self.crop_params[0]
            self.resizewinX = self.crop_params[3] - self.crop_params[2]

        # Применение изменений к кадру
        frame = self.reset(frame)
        self.frame = frame

        # Инициализация очереди и флагов
        self.q2 = Queue.Queue(maxsize=25)  # Очередь для хранения кадров
        self.stopped = False  # Флаг для остановки потока
        self.newframe = True  # Флаг наличия нового кадра
        self.lock = Lock()  # Блокировка для обеспечения потокобезопасности

        # Запуск потока для чтения кадров
        self.thread = Thread(target=self._reader, name=self.namestream, daemon=True, args=())
        self.thread.start()

    def reset(self, frame):
        """
        Метод для изменения масштаба и обрезки кадра.

        :param frame: Исходный кадр.
        :return: Обработанный кадр.
        """
        if frame is None:
            return None

        # Изменение масштаба кадра
        if not self.resize is None:
            frame = cv2.resize(frame, (self.resizeX,self.resizeY),interpolation = cv2.INTER_AREA)

        # Обрезка кадра, если заданы параметры обрезки frame[100:800, 400:1200]
        if not self.crop_params is None:
            y1, y2, x1, x2 = self.crop_params
            frame = frame[y1:y2, x1:x2]
        return frame

    def _reader(self):
        """
        Внутренний метод для чтения кадров из видеопотока.
        """
        try:
            while not self.stopped:
                self.ret, frame = self.cap.read()

                if not self.ret:
                    print(f"[e] Ошибка: Не удалось прочитать кадр. Попытка {self.retry_count + 1}/{self.max_retries}")
                    self.cap = cv2.VideoCapture(self.name)

                    print(f"[i] Инициализация видеопотока: {self.namestream}")

                    # Проверка успешности открытия видеозахвата
                    if not self.cap.isOpened():
                        print("[e] Ошибка: Не удалось открыть видеозахват.")
                    else:
                        print("[+] Видеозахват успешно открыт.")
                        self.retry_count = 0

                    self.retry_count += 1

                    if self.retry_count == self.max_retries:
                        self.stopped = True
                        self.stop()
                        break

                    sleep(self.retry_delay)
                    continue

                self.retry_count = 0
                frame = self.reset(frame)

                # Установка флага нового кадра с использованием блокировки
                with self.lock:
                    #print('>', end ='')
                    self.newframe = True
                    self.frame = frame
                #print('<', end ='')

        except Exception as e:
            print(f"[e] Ошибка в потоке {self.namestream}: Имя ошибки: {e.__class__.__name__} - {str(e)}")
            self.stop()

    def read(self):
        """
        Метод для получения последнего доступного кадра.

        :return: Последний кадр или None, если нового кадра нет.
        """
        #print('*')
        with self.lock:
            if self.newframe:
                self.newframe = False
                return self.frame
        return None

    def stop(self):
        """
        Метод для остановки видеозахвата и освобождения ресурсов.
        """
        self.stopped = True
        try:
            if self.thread.is_alive():
                try:
                    self.thread.join()
                except:
                    pass
        except:
            print('[!] Не был запущен видеопоток ')
            exit(0)


        self.cap.release()

        # Проверка и очистка очереди, если она существует
        if hasattr(self, 'q2') and isinstance(self.q2, Queue.Queue):
            self.q2.queue.clear()  # Очистка очереди


        print(f"[-] Видеозахват для потока {self.namestream} остановлен.")

    def add2(self, count=1):
        """
        Метод для добавления кадров в очередь (в текущей реализации не используется).

        :param count: Количество кадров для добавления (по умолчанию 1).
        """
        if count==0:
            return

        if self.q2.full():
            return

        delay = 0.1  # Начальная задержка
        for _ in range(count):
            while True:  # Бесконечный цикл для ожидания нового кадра
                with self.lock:  # Используем контекстный менеджер для работы с блокировкой
                    if self.newframe:  # Проверяем, есть ли новый кадр
                        # Добавляем кадр и метку времени в очередь
                        try:
                            self.q2.put((self.frame, dt.now().strftime('%Y%m%d-%H%M%S-%f')), block=False)
                        except Queue.Full:
                            pass
                        self.newframe = False  # Сбрасываем флаг нового кадра
                        break  # Выходим из цикла ожидания
                sleep(delay)  # Задержка для уменьшения нагрузки на CPU
                delay = min(delay + 0.05, 0.5)  # Увеличение задержки, но не более 0.5 секунды

    def read2(self):
        """
        Метод для получения кадра из очереди.

        :return: Кадр из очереди.
        """
        return self.q2.get()

    def size2(self):
        """
        Метод для получения размера очереди.

        :return: Размер очереди.
        """
        return self.q2.qsize()

# Запуск программы
if __name__ == "__main__":
    #stream_to_parse='rtsp://user:KKK123kkk@192.168.111.210:554/ISAPI/Streaming/Channels/101'
    stream_to_parse='rtsp://user:KKK123kkk@192.168.111.209:554/ISAPI/Streaming/Channels/101'
    stream_to_parse='rtsp://user:KKK123kkk@192.168.111.212:554/ISAPI/Streaming/Channels/101'
    #cap = VideoCapture(stream_to_parse, namestream = 'FrameStream', resize = -1.5, crop_params = [100, 800, 400, 1200])
    #cap = VideoCapture(stream_to_parse, namestream = 'FrameStream', resize = (1280,720), crop_params = [100, 800, 400, 1200])
    #cap = VideoCapture(stream_to_parse, namestream = 'FrameStream')
    # Главный вход
    cap = VideoCapture(stream_to_parse, namestream = 'FrameStream')
    #cap = VideoCapture(stream_to_parse, namestream = 'FrameStream', resize = (1280,720), crop_params = [60, 1260, 100, 1900])

    #cap = VideoCapture('S:\\KKK.lestion\\Film\\25.09(1).avi', namestream = 'FrameStream')
