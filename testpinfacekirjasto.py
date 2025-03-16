# Импортируем класс PinFace из модуля pinfacekirjasto.PinFace
from pinfacekirjasto.PinFace import PinFace, cv2

# Пример использования функций класса PinFace
if __name__ == "__main__":
    # Создаем экземпляр класса PinFace с указанием режимов обнаружения и распознавания лиц
    # Режим обнаружения лиц (ffmode) может быть 'opencv' или 'adaface'
    # Режим распознавания лиц (frmode) может быть 'sface' или/и 'mtcnn'


# Альтернативные варианты инициализации закомментированы ниже

    # pFace = PinFace(ffmode='opencv', frmode='sface')  # Альтернативный вариант инициализации
    pFace = PinFace(ffmode='adaface', frmode='sface', fmode =['opencv', 'mtcnn'] )
    # pFace = PinFace(ffmode='adaface', frmode='adaface')
    # pFace = PinFace(ffmode='opencv', frmode='adaface')

    # Загружаем изображение для обработки с использованием OpenCV
    # Функция cv2.imread загружает изображение из файла в формате BGR
    frame = cv2.imread("pinfacekirjasto/kk.jpg")

    if frame is None:
        print("Ошибка: изображение не загружено. Проверьте путь к файлу.")
        exit()

    # Выполняем обнаружение лиц на изображении
    # Метод face_detection возвращает:
    # - bboxes: координаты ограничивающих прямоугольников для обнаруженных лиц
    # - faces: список изображений лиц в формате PIL.Image
    # - facescv: список изображений лиц в формате OpenCV
    bboxes, faces, facescv = pFace.face_detection(frame)

    # Выводим время, затраченное на обнаружение лиц, и количество обнаруженных лиц
    # Форматированный вывод: время (timeff) и количество лиц (len(bboxes))
    print(f'timeff = {pFace.timeff:>4} count = {len(bboxes):>3}')

    # Если лица обнаружены, сохраняем каждое лицо как отдельное изображение
    if len(bboxes) != 0:
        for nfaces_, faces_ in enumerate(faces):
            # Сохраняем изображение лица в формате JPEG с максимальным качеством
            # Имя файла формируется как 'pinfacekirjasto/{номер_лица}.jpeg'
            faces_.save('pinfacekirjasto/' + str(nfaces_) + '.jpeg', "JPEG", quality=100)

        # Выполняем распознавание лиц на основе обнаруженных изображений
        # Метод face_recognition возвращает эмбеддинги (векторные представления лиц)
        embeddings = pFace.face_recognition(faces=faces, facescv=facescv)

        # Выводим время, затраченное на распознавание лиц, и количество эмбеддингов
        print(f'timefr = {pFace.timefr:>4} count = {len(embeddings):>3}')
