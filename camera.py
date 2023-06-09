import cv2
import torch
import pyrealsense2 as rs
import numpy as np
from torchvision import transforms
from PIL import Image
from time import time

# Завантажте навчену модель
from TwoStreamCNN_model import TwoStreamCNN

num_classes = 2

model = TwoStreamCNN(num_classes=num_classes)
model.load_state_dict(torch.load("model.pth"))
model.eval()  # Переключення моделі в режим оцінки (виводу)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Перевірка на доступність камери
try:
    pipeline.start(config)
except RuntimeError as e:
    print(f'Помилка при підключенні до камери: {str(e)}')
    exit(-1)

prev_time = time()
while True:
    try:
        # Отримання кадрів
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            print('Не вдалося отримати кадри.')
            break

        # Преобразування кадрів до numpy масивів
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Перетворіть зображення на PIL Image для подальшого перетворення
        color_pil = Image.fromarray(color_image)
        depth_pil = Image.fromarray(depth_image)

        # Застосування трансформацій до кадрів
        color_input = transform(color_pil).unsqueeze(0)
        depth_input = transform(depth_pil).unsqueeze(0)

        # Застосування моделі та отримання передбаченого класу
        with torch.no_grad():
            outputs = model(color_input, depth_input)
            _, predicted = torch.max(outputs.data, 1)

        # Відображення передбаченого класу на зображенні
        cv2.putText(color_image, str(predicted.item()), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)

        # Розрахунок та відображення FPS
        curr_time = time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(color_image, f'FPS: {fps:.2f}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Real-time object detection', color_image)

        # Зупинка обробки при натисканні 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print(f'Помилка при обробці кадру: {str(e)}')
        break

pipeline.stop()
cv2.destroyAllWindows()
