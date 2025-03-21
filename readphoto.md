# readphoto.py

Данный скрипт предназначен для обработки изображений с лицами, их анализа и добавления в базу данных. Программа использует библиотеки для работы с изображениями и компьютерным зрением, такие как `Pillow`, `OpenCV`, `numpy`, а также собственные модули для обнаружения и распознавания лиц.

---

## Основные функции

### 1. **Инициализация и настройка**
- Программа начинает работу с вывода версии и инициализации необходимых модулей.
- Используются режимы обнаружения (`ffmode`) и распознавания (`frmode`) лиц.
- Инициализируется объект `PinFace` для работы с лицами.

### 2. **Загрузка данных**
- Загружаются известные лица из базы данных (`known_file`, `known_vector`).
- Программа ищет все файлы с расширением `.jp*` (например, `.jpg`, `.jpeg`) в указанной директории.

### 3. **Обработка изображений**
- Для каждого изображения:
  - Проверяется, является ли файл дубликатом (на основе имени и размера).
  - Если файл уже обработан, он перемещается в папку `/double/`.
  - Если файл не может быть прочитан, он перемещается в папку `/error/`.
  - Изображение открывается и преобразуется в RGB.
  - На изображении обнаруживаются лица с помощью `PinFace.face_detection`.
  - Если лиц не найдено, файл перемещается в папку `/empty/`.
  - Если найдено несколько лиц, файл перемещается в папку `/many/`.
  - Если найдено одно лицо, оно обрабатывается и добавляется в базу данных.

### 4. **Распознавание лиц**
- Для каждого обнаруженного лица:
  - Извлекаются эмбеддинги (векторные представления) с использованием режимов `sface` и `adaface`.
  - Имя файла обрабатывается для извлечения имени человека.
  - Данные сохраняются в базу данных, а файл перемещается в папку `/ok/`.

### 5. **Завершение работы**
- Программа выводит статистику по добавленным записям.
- Если записи не добавлялись, выводится соответствующее сообщение.
- Если записи были добавлены, создается файл `newrecods.flag` с информацией о количестве добавленных записей.

### 6. **Очистка базы данных**
- Программа проверяет базу данных на наличие устаревших или дублирующихся записей и удаляет их.

---

## Зависимости

Для работы программы необходимы следующие библиотеки и модули:
- **Pillow (PIL)**: для работы с изображениями.
- **OpenCV (cv2)**: для обработки изображений и компьютерного зрения.
- **numpy**: для работы с массивами.
- **face_data**: пользовательский модуль для работы с данными лиц.
- **pinfacekirjasto**: пользовательский модуль для обнаружения и распознавания лиц.

---

## Пример использования

1. Убедитесь, что все зависимости установлены.
2. Укажите путь к каталогу с изображениями (`pathinput`).
3. Запустите скрипт. Программа автоматически обработает все изображения, добавит данные в базу и отсортирует файлы по папкам:
   - `/ok/` — успешно обработанные изображения.
   - `/double/` — дубликаты.
   - `/error/` — файлы с ошибками.
   - `/empty/` — изображения без лиц.
   - `/many/` — изображения с несколькими лицами.

---