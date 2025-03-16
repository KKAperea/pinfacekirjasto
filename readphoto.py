# Вывод сообщения о начале инициализации
print(f'[i] -= Версия 1.00 =-', flush=True)
from setmode import *

print(f'[o] Инициализация...\r', flush=True, end='')

# Импорт необходимых модулей
from os.path import basename as ospathbasename, getsize as ospathgetsize, splitext as ospathsplitext, isfile as ospathisfile
from glob import glob
#from PIL import Image
from shutil import move as shutilmove
from face_data import cv2, Image, np
import face_data

#sqlitelock = threading.Lock()  # Блокировка для работы с базой данных в многопоточной среде
from time import time

# 'opencv', 'adaface'
ffmode = 'adaface'
# 'sface', 'mtcnn'
frmode='sface'
fmode = ['mtcnn']
'''
if ffmode == 'AdaFace':
    from face_alignment import align
    from AdaFace import inference
    model = inference.load_pretrained_model()
elif ffmode == 'opencv':
    from SFace import FaceRecognizerSface
    import OpenCv
    FaceRecognizer = FaceRecognizerSface()
#    import cv2
#    import numpy as np'
'''

from pinfacekirjasto.PinFace import PinFace
pFace = PinFace(ffmode=ffmode, frmode=frmode, fmode=fmode)



known_file, known_vector = face_data.load_faces_from_db()



# Поиск всех файлов с расширением .jp* (например, .jpg, .jpeg) в каталоге
glob_ = glob(pathinput + '/*.jp*')
glob_len =len(glob_)
print(f'[i] В каталоге {pathinput} найдено {glob_len} файл(а/ов) по маске *.jp*')
countAddRec = 0
# Обработка каждого изображения
for nimage_path, image_path in enumerate(glob_, start= 1):
    # Получаем размер файла в байтах
    file_size = ospathgetsize(image_path)
    # Получение имени файла из пути
    image_file = ospathbasename(image_path)

    myRec =  known_file.get(image_file + '_' +str(file_size),'')

    if myRec !='':
        print(f'[-] {nimage_path:>3}/{glob_len:>3}.)            Дубликат       : {image_file}')
        shutilmove(image_path, pathinput + '/double/')
        known_file.pop(image_file + '_' +str(file_size), None)
        continue

    try:
        # Открытие изображения и преобразование в RGB
        pil_image = Image.open(image_path).convert('RGB')
    except Exception as e:
        # Обработка ошибок при чтении файла
        print(f'[e] {nimage_path:>3}/{glob_len:>3}.)            Ошибка чтения  : {image_file}')
        # Перемещение файла в каталог error
        shutilmove(image_path, pathinput + '/error/')
        continue

    # Выравнивание лиц на изображении
    bboxes, faces, facescv = pFace.face_detection(pil_image)
    tf= pFace.timeff

    # Проверка количества найденных лиц
    if len(faces) == 0:
        # Если лиц не найдено
        print(f'[-] {nimage_path:>3}/{glob_len:>3}.) {tf}       Нет лиц        : {image_file} ')
        shutilmove(image_path, pathinput + '/empty/')
    elif len(faces) > 1:
        # Если найдено несколько лиц
        print(f'[-] {nimage_path:>3}/{glob_len:>3}.) {tf}       Много лиц: {len(faces):<3} : {image_file}')
        shutilmove(image_path, pathinput + '/many/')
    else:
        # Если найдено одно лицо
        te = time()
        # Инициализация переменных
        embeddings128 = None
        embeddings512 = None
        # Обрабатываем каждый режим
        for mode in fmode:
            if mode == 'sface':
                embeddings128 = pFace.face_recognition(faces=faces, facescv=facescv, frmode='sface')[0][0]
            elif mode == 'mtcnn':
                embeddings512 = pFace.face_recognition(faces=faces, facescv=facescv, frmode='mtcnn')[0]


        namepercone = ospathsplitext(image_file)[0]
        if (namepercone.find(' ')>0) and (namepercone.find('_')>1):
            namepercone = namepercone[:namepercone.find('_')]
        elif (namepercone.find('_')>0):
            namepercone = namepercone.replace('_',' ')

        if (namepercone.find('(')>0):
            namepercone = namepercone[:namepercone.find('(')]



        namepercone = namepercone.title().strip()
        words = namepercone.rsplit(maxsplit=1)
        if words[-1] in ['Vu','Vk','Pb']:
            namepercone = words[0]




        te= str(round((time()-te)*1000)).rjust(4)

        face_data.add_face_to_db(namefile=image_file, filesize=file_size, namepercone=namepercone, vector128 = embeddings128, vector512 = embeddings512,  photo=faces[0])
        countAddRec +=1
        shutilmove(image_path, pathinput + '/ok/')
        print(f'[x] {nimage_path:>3}/{glob_len:>3}.) {tf:>4}-{te:>4}        Добавлен : {image_file} -> {namepercone}')

if countAddRec ==0:
    print(f'[i] Записи не добавлялись')
else:
    print(f'[i] Добавленно  {countAddRec} зап.')
    with open('newrecods.flag', "w+") as f:
        f.write(f'[i] Добавленно  {countAddRec} зап.\n')


# Функция для проверки условия удаления
def should_remove(key, value):
    #print(key, value)
    if ospathisfile(pathinput + '/ok/' + value['namefile']):
        return True
    if ospathisfile(pathinput + '/double/' + value['namefile']):
        return True
    return False  # Пример условия: удалить элементы, где значение больше 2
# Собираем ключи для удаления
keys_to_remove = [k for k, v in known_file.items() if should_remove(k, v)]
# Удаляем элементы из словаря
for key in keys_to_remove:
    del known_file[key]
for k, v in known_file.items():
    print('Удалить = ', v['namefile'])
