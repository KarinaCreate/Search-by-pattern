import numpy as np
import cv2 as cv
import cv2
import requests
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt

# Функция для загрузки изображения по URL
def read_image_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# Функция для добавления черных полос снизу изображению с меньшей высотой
def pad_images_to_same_height(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if h1 < h2:
        img1 = cv2.copyMakeBorder(img1, 0, h2-h1, 0, 0, cv.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        img2 = cv2.copyMakeBorder(img2, 0, h1-h2, 0, 0, cv.BORDER_CONSTANT, value=(0, 0, 0))
    return img1, img2

# Функция для обнаружения ключевых точек и вычисления дескрипторов с использованием SIFT
def detect_and_compute_sift(image, mask=None):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, mask)
    return keypoints, descriptors

# Функция для поиска совпадений между дескрипторами
def match_keypoints(des1, des2, ratio=0.75):
    bf = cv.BFMatcher(cv.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < ratio * n.distance]
    return good_matches

# Функция для нахождения гомографии между ключевыми точками
def find_homography(kp1, kp2, matches):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    return M, mask

# Функция для рисования обнаруженного объекта на изображении
def draw_detected_object(image, M, template_shape):
    h, w = template_shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M)
    image = cv.polylines(image, [np.int32(dst)], True, (0, 255, 0), 3, cv.LINE_AA)
    return image, dst

# Основная функция для поиска объекта на изображении
def predict_image(train_file, template_file):
    # Загрузка изображений
    original_img = read_image_from_url(train_file)
    template_img = read_image_from_url(template_file)

    # Преобразование в серое изображение
    img_gray = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)
    template_gray = cv.cvtColor(template_img, cv.COLOR_BGR2GRAY)

    # Маска для поиска областей
    mask = np.ones(img_gray.shape, dtype=np.uint8) * 255

    # Обнаружение ключевых точек и дескрипторов для шаблона
    kp1, des1 = detect_and_compute_sift(template_gray)

    while True:
        # Обнаружение ключевых точек и дескрипторов для изображения
        kp2, des2 = detect_and_compute_sift(img_gray, mask)
        if des2 is None:
            break

        # Поиск совпадений
        matches = match_keypoints(des1, des2)

        if len(matches) > 10:
            # Нахождение гомографии
            M, mask_homography = find_homography(kp1, kp2, matches)
            if M is not None and mask_homography is not None and mask_homography.sum() > 10:
                # Рисование границ найденного объекта
                img_with_object, detected_region = draw_detected_object(original_img, M, template_img.shape[:2])
                cv.fillConvexPoly(mask, np.int32(detected_region), 0)
            else:
                break
        else:
            break

    # Добавление черных полос снизу изображению с меньшей высотой
    template_img, final_img = pad_images_to_same_height(template_img, img_with_object)

    # Показ изображения
    #cv.imshow("Detected Object", final_img)
    #cv.waitKey(0)  # ждем нажатия клавиши
    #cv.destroyAllWindows()  # закрываем окно

    plt.figure(figsize=(12, 8))  # размер окна в дюймах
    plt.imshow(cv.cvtColor(final_img, cv.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

# Простой вариант для одного набора изображений
train_file = 'https://raw.githubusercontent.com/KarinaCreate/Search-by-pattern/8ae239181b7a92326f22365d481130d7f6a323bd/train/train_5.jpg'
template_file = 'https://raw.githubusercontent.com/KarinaCreate/Search-by-pattern/8ae239181b7a92326f22365d481130d7f6a323bd/template/template_5.jpg'

predict_image(train_file, template_file)