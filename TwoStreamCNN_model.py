import torch
import torch.nn as nn


# Визначте модель
class TwoStreamCNN(nn.Module):
    def __init__(self, num_classes):
        super(TwoStreamCNN, self).__init__()

        self.rgb_stream = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=(11, 11), stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=(5, 5), padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.depth_stream = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=(11, 11), stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=(5, 5), padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Обчислення розміру вхідних даних для повнозв'язного шару
        with torch.no_grad():
            sample_rgb = torch.zeros(1, 3, 224, 224)
            sample_depth = torch.zeros(1, 1, 224, 224)
            sample_rgb_output = self.rgb_stream(sample_rgb)
            sample_depth_output = self.depth_stream(sample_depth)
            sample_combined = torch.cat((sample_rgb_output, sample_depth_output), dim=1)
            input_size = sample_combined.view(-1).size(0)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(input_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, rgb, depth):
        rgb_features = self.rgb_stream(rgb)
        depth_features = self.depth_stream(depth)
        combined = torch.cat((rgb_features, depth_features), dim=1)  # Конкатенація відповідних шарів
        combined = combined.view(combined.size(0), -1)
        output = self.classifier(combined)
        return output
