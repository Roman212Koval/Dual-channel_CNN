import cv2
import numpy as np
import os
import pyrealsense2 as rs

# Ім'я каталогу, в якому будуть зберігатися зображення
save_dir = "path/to/save/directory"

# Ініціалізація камери
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
profile = pipeline.start(config)

# Ініціалізація змінних
saved_pairs = 0

try:
    while True:
        # Отримання кадру з камери
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        # Перетворення кадру у зображення
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Відображення зображення
        cv2.imshow("Color Image", color_image)
        cv2.imshow("Depth Image", depth_image)

        # Зчитування клавіші
        key = cv2.waitKey(1)

        # Збереження зображень при натисканні на клавішу 'Space'
        if key == ord(" "):
            filename_color = os.path.join(save_dir, f"{saved_pairs}_RGB.jpg")
            filename_depth = os.path.join(save_dir, f"{saved_pairs}_D.jpg")
            cv2.imwrite(filename_color, color_image)
            cv2.imwrite(filename_depth, depth_image)
            saved_pairs += 1

        # Закриття вікон при натисканні клавіші 'ESC'
        if key == key == 27:
            break

finally:
    # Зупинка камери та закриття вікна
    pipeline.stop()
    cv2.destroyAllWindows()
