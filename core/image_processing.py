import cv2
import numpy as np
import os
from typing import Tuple, List
from sklearn.cluster import KMeans

def apply_median_blur(image: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """
    Применяет медианный фильтр к изображению для сглаживания.

    :param image: Исходное изображение в формате NumPy массива.
    :param kernel_size: Размер ядра для медианного фильтра.
    :return: Отфильтрованное изображение.
    """
    return cv2.medianBlur(image, kernel_size)

def perform_kmeans_clustering(image: np.ndarray, n_clusters: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Применяет KMeans кластеризацию к изображению.

    :param image: Изображение в формате NumPy массива.
    :param n_clusters: Число кластеров для KMeans.
    :return: Кластеризованное изображение и метки кластеров.
    """
    pixels = image.reshape((-1, 3))  # Преобразование в двумерный массив для кластеризации
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pixels)

    # Получение сегментированного изображения
    segmented_image = kmeans.cluster_centers_[kmeans.labels_]
    segmented_image = segmented_image.reshape(image.shape)
    segmented_image = segmented_image.astype(np.uint8)

    # Возвращаем сегментированное изображение и метки кластеров
    return segmented_image, kmeans.labels_.reshape(image.shape[:2])

def create_masks(labels: np.ndarray, n_clusters: int) -> List[np.ndarray]:
    """
    Создает бинарные маски для каждого из кластеров.

    :param labels: Метки кластеров.
    :param n_clusters: Число кластеров.
    :return: Список бинарных масок.
    """
    masks = []
    for i in range(n_clusters):
        mask = (labels == i).astype(np.uint8) * 255  # Создаем бинарную маску для каждого кластера
        masks.append(mask)
    return masks

def save_image(image: np.ndarray, path: str) -> None:
    """
    Сохраняет изображение по заданному пути, поддерживая Unicode в путях.

    :param image: Изображение в формате NumPy массива.
    :param path: Путь для сохранения изображения.
    """
    # Определяем расширение файла
    ext = os.path.splitext(path)[1]

    # Кодируем изображение в буфер
    result, encoded_img = cv2.imencode(ext, image)
    if result:
        # Сохраняем изображение с использованием Unicode-пути
        encoded_img.tofile(path)
    else:
        raise IOError("Ошибка при сохранении изображения.")
