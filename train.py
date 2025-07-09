import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
import json
from models.model import AntiSpoofNet
from PIL import Image

# === CONFIG ===
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULT_DIR, "logs"), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root="data", transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = AntiSpoofNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

writer = SummaryWriter(os.path.join(RESULT_DIR, "logs"))

num_epochs = 10
train_losses = []
train_accuracies = []

for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / total
    accuracy = (correct / total) * 100

    train_losses.append(avg_loss)
    train_accuracies.append(accuracy)

    print(f"Epoch {epoch+1}/{num_epochs} Loss: {avg_loss:.4f} Accuracy: {accuracy:.2f}%")

    writer.add_scalar("Loss/train", avg_loss, epoch+1)
    writer.add_scalar("Accuracy/train", accuracy, epoch+1)

# === Save model ===
torch.save(model.state_dict(), os.path.join(RESULT_DIR, "anti_spoof_model.pth"))

# === Save training metrics ===
metrics = {"loss": train_losses, "accuracy": train_accuracies}
with open(os.path.join(RESULT_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f)

# === Plot training curves ===
epochs = list(range(1, num_epochs + 1))
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.lineplot(x=epochs, y=train_losses, color="blue", label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")

plt.subplot(1, 2, 2)
sns.lineplot(x=epochs, y=train_accuracies, color="green", label="Train Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training Accuracy")

plt.tight_layout()
plt.savefig(os.path.join(RESULT_DIR, "training_curves.png"))
plt.show()

# === Save preview of predictions ===
for imgs, lbls in train_loader:
    imgs = imgs.to(device)
    outputs = model(imgs)
    preds = torch.argmax(outputs, dim=1)

    grid = utils.make_grid(imgs, nrow=8, normalize=True, pad_value=1)
    plt.figure(figsize=(15, 5))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title("Predictions:\n" + " ".join(str(int(x)) for x in preds.cpu()))
    plt.axis("off")
    plt.savefig(os.path.join(RESULT_DIR, "batch_predictions.png"))
    plt.show()
    break
