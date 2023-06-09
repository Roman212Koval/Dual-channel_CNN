import os

import torch
from PIL import Image
import torch.utils.data as data
from torchvision.transforms import transforms

class_names = ['bottle', 'box', 'car', 'cup', 'glass', 'speaker']
# словник для перетворення словесних описів в числові індекси
class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}


class CustomDataset(data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.rgb_dir = os.path.join(root_dir, "rgb")
        self.depth_dir = os.path.join(root_dir, "depth")
        self.filenames = []

        # Додавання файлів зображень з кожної директорії RGB
        for subdir, _, files in os.walk(self.rgb_dir):
            for file in files:
                if file.endswith('.jpg'):
                    self.filenames.append(os.path.join(subdir, file))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Завантаження RGB та Depth зображення
        rgb_filename = self.filenames[idx]
        depth_filename = os.path.join(self.depth_dir, *rgb_filename.split(os.path.sep)[-2:]).replace('.jpg',
                                                                                                     '_depth.jpg')
        rgb_image = Image.open(rgb_filename)
        depth_image = Image.open(depth_filename).convert('L')

        # Визначення преобразування для зображення
        transform = transforms.Compose([
            transforms.Resize((224, 224)),  # добавляємо цей рядок
            transforms.ToTensor()
        ])

        # Перетворення зображення у тензори та нормалізація
        rgb_tensor = transform(rgb_image)
        depth_tensor = transform(depth_image)

        # Завантаження міток класів
        label = rgb_filename.split("\\")[-2]
        label = class_to_idx[label]
        # Перетворення міток класів у тензори
        label_tensor = torch.tensor(label, dtype=torch.long)

        # Повернення тензорів та міток
        return rgb_tensor, depth_tensor, label

