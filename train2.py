import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter  # Додано
from CustomDataset import CustomDataset
from OneStreamCNN import OneStreamCNN  # Змінено на OneStreamCNN

print(torch.cuda.is_available())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)

# Задання гіперпараметрів
batch_size = 32
learning_rate = 0.001
num_epochs = 11
num_classes = 6

# Створення writer для TensorBoard
train_writer = SummaryWriter('runs/train')
val_writer = SummaryWriter('runs/validation')

train_dataset = CustomDataset("dataset/")
test_dataset = CustomDataset("dataset/validation/")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

model = OneStreamCNN(num_classes=num_classes).to(device)  # Змінено на OneStreamCNN

# Визначення функцію втрат та оптимізатор
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Навчання моделі
for epoch in range(num_epochs):
    print('Epoch #' + str(epoch))

    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for i, (rgb_images, _, labels) in enumerate(train_loader):  # Використання _ для ігнорування значень глибини
        rgb_images = rgb_images.to(device)
        labels = labels.to(device)

        outputs = model(rgb_images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    # Запис тренувальних даних до TensorBoard
    train_writer.add_scalar('Loss', train_loss / len(train_loader), epoch)
    train_writer.add_scalar('Accuracy', 100 * train_correct / train_total, epoch)


    # Оцінка моделі
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for rgb_images, _, labels in test_loader:  # Використання _ для ігнорування значень глибини
            rgb_images, labels = rgb_images.to(device), labels.to(device)

            outputs = model(rgb_images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_loss = val_loss / len(test_loader)
    val_accuracy = 100 * val_correct / val_total

    # Запис даних валідації до TensorBoard
    val_writer.add_scalar('Loss', val_loss, epoch)
    val_writer.add_scalar('Accuracy', val_accuracy, epoch)

# Закриття writer після завершення тренування
train_writer.close()
val_writer.close()

# Збереження моделі
torch.save(model.state_dict(), "model.pth")
print("Model saved.")
