import sys
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton
from PyQt5.QtGui import QPixmap, QImage, QPainter, QColor, QFont
from PyQt5.QtCore import Qt, QObject, QThread, pyqtSignal, QTimer
from torchvision.transforms import transforms

from OneStreamCNN import OneStreamCNN

classes = ['bottle', 'box', 'car', 'cup', 'glass', 'speaker']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
# Завантаження нейронної мережі та вагів
model = OneStreamCNN(num_classes=6).to(device)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Визначення преобразування для зображення
transform_img = transforms.Compose([
    transforms.Resize((224, 224)),  # добавляємо цей рядок
    transforms.ToTensor()
])


class CameraWorker(QObject):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self._run_flag = False
        self.timer = QTimer()
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(50)  # Для 30 кадрів в секунду
        self.qt_image = None  # Додали атрибут qt_image

    def update_frame(self):
        if not self._run_flag:
            return
        ret, frame = self.cap.read()
        if ret:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(np.uint8(rgb_image))
            rgb_tensor = transform_img(pil_image)
            rgb_tensor = rgb_tensor.to(device)
            rgb_tensor = rgb_tensor.unsqueeze(0).to(device)
            outputs = model(rgb_tensor)
            predicted_class = torch.argmax(outputs).item()

            # Додатковий код для виводу напису "РОМАН" поверх зображення
            painter = QPainter(qt_image)
            painter.setFont(QFont('Arial', 40))
            painter.setPen(QColor(217, 22, 54))
            painter.drawText(100, 65, classes[predicted_class])
            painter.end()

            self.qt_image = qt_image  # Оновлення атрибуту qt_image

            self.change_pixmap_signal.emit(qt_image)

    def start(self):
        self._run_flag = True

    def stop(self):
        self._run_flag = False
        self.take_screenshot()

    def take_screenshot(self):
        if self.qt_image is not None:
            pixmap = QPixmap.fromImage(self.qt_image)

            # Збереження зображення
            current_datetime = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"screenshots/screenshot_{current_datetime}.png"
            pixmap.save(filename)
            print("Screenshot saved as", filename)
        else:
            print("No pixmap available for screenshot.")


class Application(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video Stream App")
        self.setGeometry(100, 100, 800, 600)

        self.image_label = QLabel()
        self.start_button = QPushButton('Start')
        self.stop_button = QPushButton('Stop')
        self.quit_button = QPushButton('Quit')

        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.start_button)
        vbox.addWidget(self.stop_button)
        vbox.addWidget(self.quit_button)

        self.setLayout(vbox)

        self.camera_thread = QThread()
        self.camera_worker = CameraWorker()
        self.camera_worker.moveToThread(self.camera_thread)

        self.camera_worker.change_pixmap_signal.connect(self.update_image)
        self.start_button.clicked.connect(self.camera_worker.start)
        self.stop_button.clicked.connect(self.camera_worker.stop)
        self.quit_button.clicked.connect(self.close_application)
        self.camera_thread.start()

    def update_image(self, qt_image):
        pixmap = QPixmap.fromImage(qt_image)
        self.image_label.setPixmap(pixmap)

    def close_application(self):
        self.camera_worker.stop()
        self.camera_thread.quit()
        self.camera_thread.wait()
        self.close()


app = QApplication(sys.argv)
window = Application()
window.show()
sys.exit(app.exec_())
