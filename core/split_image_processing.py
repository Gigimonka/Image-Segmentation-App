from PIL import Image
import os

Image.MAX_IMAGE_PIXELS = None

def split_image(image_path: str, n: int, output_folder: str = None) -> None:
    """
    Разбивает изображение на n x n частей и сохраняет их в указанную папку.

    :param image_path: Путь к исходному изображению.
    :param n: Количество частей по вертикали и горизонтали.
    :param output_folder: Папка для сохранения частей изображения. По умолчанию — 'splitter_images' в текущей директории.
    """
    # Если output_folder не указан, используем 'splitter_images' в текущей директории
    if output_folder is None:
        output_folder = os.path.join(os.getcwd(), "splitter_images")
    
    # Открываем изображение
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Определяем размеры каждой части
    tile_width = img_width // n
    tile_height = img_height // n

    # Создаем папку для сохранения частей изображения
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Разделяем изображение на n x n частей
    for i in range(n):
        for j in range(n):
            # Определяем координаты текущего сегмента
            left = i * tile_width
            top = j * tile_height
            right = (i + 1) * tile_width if i != n - 1 else img_width
            bottom = (j + 1) * tile_height if j != n - 1 else img_height

            # Вырезаем и сохраняем сегмент
            tile = img.crop((left, top, right, bottom))
            tile_name = f"tile_{i}_{j}.png"
            tile.save(os.path.join(output_folder, tile_name))
    
    print(f"Изображение разделено на {n}x{n} частей и сохранено в папку: {output_folder}")
