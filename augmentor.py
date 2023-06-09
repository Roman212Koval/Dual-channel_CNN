import Augmentor
import cv2
import glob

# Шлях до вашого датасету
input_dir = 'dataset/dop/bottle'


def my_augmentor():
    # Створити пайплайн Augmentor для вертикальних зображень
    p = Augmentor.Pipeline(input_dir)
    # Визначити аугментації
    p.rotate(probability=0.7, max_left_rotation=12, max_right_rotation=12)
    p.zoom(probability=0.7, min_factor=1.1, max_factor=1.5)
    p.flip_top_bottom(probability=0.5)
    p.random_distortion(probability=1, grid_width=4, grid_height=4, magnitude=8)
    # Додайте зміну розміру зображення
    p.resize(probability=1, width=640, height=480)
    # Генерувати 4N нових зображень (де N - кількість зображень в оригінальному датасеті)
    p.sample(1 * len(p.augmentor_images))
    p.process()


my_augmentor()
# Тепер використовуємо OpenCV для повороту зображень
image_files = glob.glob(input_dir + '/output/*')

for image_file in image_files:
    img = cv2.imread(image_file)

    # Повертання на 270 градусів проти годинникової стрілки
    img_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    # Зберегти зображення
    cv2.imwrite(image_file, img_rotated)

# my_augmentor()