import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from core.image_processing import apply_median_blur, perform_kmeans_clustering, create_masks, save_image
from core.split_image_processing import split_image  # Импортируем функцию разбиения изображения


class ImageSegmentationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Segmentation App")
        
        # Устанавливаем размеры окна
        self.window_width = 1200  # Увеличим ширину окна
        self.window_height = 800
        self.root.geometry(f"{self.window_width}x{self.window_height}")
        
        # Кнопка для загрузки изображения (отдельная колонка для кнопок)
        self.upload_button = tk.Button(root, text="Загрузить изображение", command=self.upload_image)
        self.upload_button.grid(row=0, column=3, padx=10, pady=10)

        # Площадка для отображения исходного изображения
        self.image_label = tk.Label(root)
        self.image_label.grid(row=1, column=0, columnspan=3, padx=10, pady=10)

        # Подпись для исходного изображения
        self.image_label_text = tk.Label(root, text="Исходное изображение")
        self.image_label_text.grid(row=2, column=0, columnspan=3)

        # Площадка для отображения кластеризованного изображения
        self.clustered_image_label = tk.Label(root)
        self.clustered_image_label.grid(row=3, column=0, columnspan=3, padx=10, pady=10)

        # Подпись для кластеризованного изображения
        self.clustered_label_text = tk.Label(root, text="Кластеризованное изображение")
        self.clustered_label_text.grid(row=4, column=0, columnspan=3)

        # Площадки для отображения масок (3 маски)
        self.mask_label_1 = tk.Label(root)
        self.mask_label_1.grid(row=5, column=0, padx=10, pady=10)

        self.mask_label_2 = tk.Label(root)
        self.mask_label_2.grid(row=5, column=1, padx=10, pady=10)

        self.mask_label_3 = tk.Label(root)
        self.mask_label_3.grid(row=5, column=2, padx=10, pady=10)

        # Подписи для масок
        self.mask_label_1_text = tk.Label(root, text="Бинарная маска 1")
        self.mask_label_1_text.grid(row=6, column=0)

        self.mask_label_2_text = tk.Label(root, text="Бинарная маска 2")
        self.mask_label_2_text.grid(row=6, column=1)

        self.mask_label_3_text = tk.Label(root, text="Бинарная маска 3")
        self.mask_label_3_text.grid(row=6, column=2)

        # Кнопка для выполнения кластеризации (в новой колонке)
        self.cluster_button = tk.Button(root, text="Применить кластеризацию", command=self.cluster_image)
        self.cluster_button.grid(row=1, column=3, padx=10, pady=10)
        self.cluster_button.config(state=tk.DISABLED)

        # Кнопка для разбиения изображения (в новой колонке)
        self.split_button = tk.Button(root, text="Разбить изображение на части", command=self.split_image_button)
        self.split_button.grid(row=2, column=3, padx=10, pady=10)
        self.split_button.config(state=tk.DISABLED)

        self.image = None  # Для хранения загруженного изображения
        self.file_path = ""  # Путь к загруженному файлу

    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return

        self.file_path = file_path  # Сохраняем путь к файлу

        try:
            # Загружаем изображение с поддержкой Unicode-пути
            image_data = np.fromfile(self.file_path, dtype=np.uint8)
            self.image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось загрузить изображение: {e}")
            return

        if self.image is None:
            messagebox.showerror("Ошибка", "Не удалось загрузить изображение")
            return

        # Отображаем масштабированное изображение в окне приложения
        self.display_resized_image(self.image, self.image_label, max_width=400, max_height=400)

        # Включаем кнопки кластеризации и разбиения
        self.cluster_button.config(state=tk.NORMAL)
        self.split_button.config(state=tk.NORMAL)

    def display_resized_image(self, image: np.ndarray, label_widget, max_width: int = 400, max_height: int = 400):
        """
        Масштабирует изображение для корректного отображения в окне приложения и показывает его в переданном Label виджете.

        :param image: Исходное изображение в формате NumPy массива.
        :param label_widget: Виджет Label для отображения изображения.
        :param max_width: Максимальная ширина изображения для отображения.
        :param max_height: Максимальная высота изображения для отображения.
        """
        # Получаем размеры изображения
        img_height, img_width = image.shape[:2]

        # Рассчитываем коэффициент масштабирования
        scale = min(max_width / img_width, max_height / img_height) / 2

        # Новые размеры изображения
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)

        # Изменяем размер изображения с сохранением пропорций
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Преобразуем в формат для отображения в tkinter
        image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image_pil)

        # Отображаем изображение в элементе Label
        label_widget.config(image=image_tk)
        label_widget.image = image_tk

    def cluster_image(self):
        if self.image is None:
            return

        # Применяем медианный фильтр и кластеризацию
        blurred_image = apply_median_blur(self.image)
        segmented_image, labels = perform_kmeans_clustering(blurred_image)

        # Директория для сохранения результатов
        output_dir = os.path.join(os.getcwd(), 'assets', 'images')

        # Проверяем, существует ли директория, если нет — создаем её
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Сохраняем сегментированное изображение
        segmented_image_path = os.path.join(output_dir, "segmented_image.png")
        save_image(segmented_image, segmented_image_path)

        # Отображаем кластеризованное изображение в интерфейсе
        self.display_resized_image(segmented_image, self.clustered_image_label, max_width=400, max_height=400)

        # Создаем и сохраняем маски
        masks = create_masks(labels, n_clusters=3)
        for i, mask in enumerate(masks):
            mask_path = os.path.join(output_dir, f"mask_cluster_{i}.png")
            save_image(mask, mask_path)

            # Отображаем маски
            if i == 0:
                self.display_resized_image(mask, self.mask_label_1, max_width=400, max_height=400)
            elif i == 1:
                self.display_resized_image(mask, self.mask_label_2, max_width=400, max_height=400)
            elif i == 2:
                self.display_resized_image(mask, self.mask_label_3, max_width=400, max_height=400)

        messagebox.showinfo("Завершено", f"Кластеризация и маски сохранены в {output_dir}")

    def split_image_button(self):
        if not self.file_path:
            messagebox.showerror("Ошибка", "Не загружено изображение для разбиения.")
            return

        # Запрашиваем у пользователя количество частей
        try:
            n = simpledialog.askinteger("Введите число частей", "На сколько частей разделить изображение (например, 4)?", minvalue=2)
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при выборе числа частей: {e}")
            return

        if n is None:
            messagebox.showwarning("Отмена", "Разбиение отменено.")
            return

        # Используем текущую директорию для сохранения изображений
        output_dir = os.path.join(os.getcwd(), "splitter_images")

        # Разделение изображения на части
        try:
            split_image(self.file_path, n, output_dir)
            messagebox.showinfo("Завершено", f"Изображение разделено на {n}x{n} частей и сохранено в {output_dir}")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при разбиении изображения: {e}")
