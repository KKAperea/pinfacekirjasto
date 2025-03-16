#Привести в порядок добавить подробный коментарий, отимизировать, улучшить скорость исполнения, восприятие и читаемость кода,  ремарки не удалять
# Импортируем необходимые библиотеки
from PIL import Image  # Библиотека для работы с изображениями
from time import time  # Библиотека для измерения времени выполнения
import numpy as np  # Библиотека для работы с массивами
import cv2  # Библиотека OpenCV для работы с изображениями и компьютерным зрением

# Определяем класс PinFace для работы с обнаружением и распознаванием лиц
class PinFace:
    def __init__(self, ffmode:str='opencv', frmode:str='sface', fmode: list = []):
        """
        Инициализация класса PinFace.

        Параметры:
        - ffmode: Режим обнаружения лиц. Может быть или/и 'opencv', 'adaface'
        - frmode: Режим распознавания лиц. Может быть 'sface' или/и 'mtcnn'.
        """
        if not ffmode in fmode: fmode.append(ffmode)
        if not frmode in fmode: fmode.append(frmode)
        try:
            # Проверяем, что выбранные режимы допустимы
            assert ffmode in ['opencv', 'adaface']
            valid_frmodes = ['sface', 'mtcnn']
            assert frmode in valid_frmodes, f"Недопустимое значение frmode: {frmode}. Ожидается одно из: {valid_frmodes}"
            for fmode_ in fmode:
                assert fmode_ in ['opencv', 'adaface','sface', 'mtcnn'], f"Недопустимое значение fmode: {fmode_}"
        except Exception as e:
            # Если возникла ошибка, выводим сообщение и завершаем программу
            s = str(e).strip()  # Преобразуем исключение в строку и убираем лишние пробелы
            print(
                f'Ошибка в блоке PinFace:__init__; '
                f'Имя ошибки: {e.__class__.__name__}'
                f'{"; Сообщение об ошибке: " + s if s else ""}'  # Добавляем сообщение об ошибке, если оно не пустое
            )
            exit()

        # Сохраняем выбранные режимы как атрибуты класса
        self.ffmode = ffmode
        self.frmode  = frmode
        self.fmode  = fmode


        self.timeff = 0  # Время, затраченное на обнаружение лиц
        self.timefr = 0  # Время, затраченное на распознавание лиц

        for fmode_ in fmode:
            print('[o] Инициализации модуля = ' + fmode_ + '\r', end = '', flush= True)
            # Импортируем необходимые модули в зависимости от выбранного режима
            if fmode_ == 'opencv':
                import pinfacekirjasto.face_detection.opencv.OpenCv as OpenCv
                self.OpenCv = OpenCv
            elif fmode_ == 'mtcnn':
                import pinfacekirjasto.face_detection.mtcnn.align as alignadaface
                self.alignadaface = alignadaface
                """
            elif fmode_ == 'mediapipe':
                import mediapipe as mp
                mp_face_detection = mp.solutions.face_detection
                self.faceDetection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.65)
                # https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/face_detection.md
                '''
                mp_face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=0.5) as face_detection:
                model_selection
                Целочисленный индекс 0или 1. Используйте 0 для выбора модели с коротким диапазоном,
                которая лучше всего подходит для лиц в пределах 2 метров от камеры, и 1для модели с полным диапазоном,
                которая лучше всего подходит для лиц в пределах 5 метров.
                Для опции с полным диапазоном используется разреженная модель для улучшенной скорости вывода.

                min_detection_confidence=0.5
                Минимальное значение достоверности ( [0.0, 1.0]) из модели обнаружения лиц, чтобы обнаружение считалось успешным. По умолчанию 0.5.
                '''
                """

            # Импортируем необходимые модули в зависимости от выбранного режима распознавания лиц
            elif fmode_ == 'sface':
                from pinfacekirjasto.face_recognition.sface.sface import FaceRecognizerSface
                self.modelSface = FaceRecognizerSface()
            elif fmode_ == 'adaface':
                import pinfacekirjasto.face_recognition.adaface.inference as inference
                self.inference = inference
                self.device = self.inference.load_model_cpu
                self.modeladaface = self.inference.load_pretrained_model()
        print('[i] Загружены модули     = ' + ', '.join(fmode) + \
               ('; device = (' + self.device + ')' if hasattr(self, 'device') else '') + \
               '        ', flush= True)
        # Вывод информации о выбранных режимах
        print(f'[i] Face_detection       = {ffmode}', flush=True)  # Режим детекции лиц
        print(f'[i] Facial_recognition   = {frmode}', flush=True)  # Режим распознавания лиц


    def face_detection(self, frame, ffmode=None):
        """
        Метод для обнаружения лиц на изображении.

        Параметры:
        - frame: Изображение, на котором нужно обнаружить лица.
        - ffmode: Режим обнаружения лиц. Если не указан, используется режим, заданный при инициализации.

        Возвращает:
        - bboxes: Координаты ограничивающих прямоугольников для обнаруженных лиц.
        - faces: Список изображений лиц в формате PIL.Image.
        - facescv: Список изображений лиц в формате OpenCV.
        """
        self.timeff = time()  # Засекаем время начала обнаружения лиц

        if ffmode is None:
            ffmode = self.ffmode  # Используем режим, заданный при инициализации, если не указан другой


        # Проверяем, является ли входной кадр объектом PIL.Image
        if isinstance(frame, Image.Image):
            # Если размер кадра равен (112, 112), возвращаем результат без обработки
            if (112, 112) == frame.size:
                # Вычисляем время выполнения и округляем до миллисекунд
                self.timeff = round((time() - self.timeff) * 1000)
                # Возвращаем:
                # - bboxes: координаты ограничивающего прямоугольника (0, 0, 112, 112)
                # - faces: список с одним изображением (исходный кадр)
                # - facescv: пустой список (так как кадр уже в формате PIL.Image)
                return [0, 0, 112, 112], [frame], []

        # Проверяем, является ли входной кадр объектом numpy.ndarray (формат OpenCV)
        if isinstance(frame, np.ndarray):
            # Если размер кадра равен (112, 112), возвращаем результат без обработки
            if (112, 112) == frame.shape[:2]:
                # Вычисляем время выполнения и округляем до миллисекунд
                self.timeff = round((time() - self.timeff) * 1000)
                # Конвертируем кадр из формата BGR (OpenCV) в RGB (PIL.Image)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Возвращаем:
                # - bboxes: координаты ограничивающего прямоугольника (0, 0, 112, 112)
                # - faces: список с одним изображением (кадр в формате PIL.Image)
                # - facescv: список с одним изображением (исходный кадр в формате OpenCV)
                return [0, 0, 112, 112], [Image.fromarray(frame_rgb)], [frame]



        if ffmode == 'adaface':
            # Обнаружение лиц с использованием AdaFace
            bboxes, faces = self.alignadaface.get_aligned_faces(frame)
            self.timeff = round((time() - self.timeff) * 1000)  # Вычисляем время выполнения
            return bboxes, faces, []  # Возвращаем результаты

        elif ffmode == 'opencv':
            # Обнаружение лиц с использованием OpenCV
            if  isinstance(frame, Image.Image):
                cvframe = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
            else:
                cvframe = frame



            bboxes, _, scores = self.OpenCv.detectMultiScale3(cvframe)
            facescv, faces = [], []
            for nbboxes_, bboxes_ in enumerate(bboxes):
                (x, y, w, h) = bboxes_
                img_bgr = cvframe[y:y + h, x:x + w]  # Вырезаем область лица
                img_bgr = cv2.resize(img_bgr, (112, 112), interpolation=cv2.INTER_AREA)  # Изменяем размер
                facescv.append(img_bgr)  # Добавляем изображение лица в формате OpenCV
                faces.append(Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)))  # Конвертируем в PIL Image
                bboxes[nbboxes_] = (x, y, x + w, y + h)  # Обновляем координаты ограничивающего прямоугольника

        self.timeff = round((time() - self.timeff) * 1000)  # Вычисляем время выполнения
        return bboxes, faces, facescv  # Возвращаем результаты

    def face_recognition(self, bboxes=[], faces=[], facescv=[], frmode=None):
        """
        Метод для распознавания лиц на основе обнаруженных изображений.

        Параметры:
        - bboxes: Координаты ограничивающих прямоугольников для обнаруженных лиц.
        - faces: Список изображений лиц в формате PIL.Image.
        - facescv: Список изображений лиц в формате OpenCV.
        - frmode: Режим распознавания лиц. Если не указан, используется режим, заданный при инициализации.

        Возвращает:
        - embeddings: Список эмбеддингов (векторных представлений) для каждого лица.
        """
        self.timefr = time()  # Засекаем время начала распознавания лиц

        if frmode is None:
            frmode = self.frmode  # Используем режим, заданный при инициализации, если не указан другой

        # Если входные данные пусты, возвращаем пустой список
        if (faces == [] and frmode == 'adaface') or \
           ((faces == [] and facescv == []) and frmode == 'sface'):
            self.timefr = 0
            return []

        embeddings = []
        if frmode == 'sface':
            # Распознавание лиц с использованием SFace
            for nfaces_, faces_ in enumerate(faces):
                if len(facescv) == 0:
                    facescv_ = cv2.cvtColor(np.array(faces_), cv2.COLOR_RGB2BGR)  # Конвертируем в BGR, если facescv пуст
                else:
                    facescv_ = facescv[nfaces_]

                embedding = self.modelSface.recognizer_(facescv_)  # Получаем эмбеддинг
                embeddings.append(embedding)

        elif frmode == 'mtcnn':
            # Распознавание лиц с использованием AdaFace
            for nfaces_, faces_ in enumerate(faces):
                bgr_tensor_input = self.inference.to_input(faces_)  # Подготавливаем входные данные
                embedding, norms = self.modeladaface(bgr_tensor_input)  # Получаем эмбеддинг
                embedding = embedding.cpu().detach().numpy()[0]  # Преобразуем в numpy массив
                embeddings.append(embedding)

        self.timefr = round((time() - self.timefr) * 1000)  # Вычисляем время выполнения
        return embeddings  # Возвращаем результаты
