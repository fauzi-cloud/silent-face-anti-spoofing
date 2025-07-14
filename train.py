import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import matplotlib.pyplot as plt
import numpy as np

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# TRANSFORMS
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomRotation(10),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# DATASET DIRECTORY
data_dir = "data"

# LOAD DATASET
full_dataset = datasets.ImageFolder(root=data_dir, transform=transform_train)

# SPLIT DATASET ( BUAT BAGI DATA TRAIN SAMA DATA VALIDASI (0.8 MAKSUTNYA 80% MASUK DATA TRAIN, SISANYA DATA VALIDASI) )
num_train = int(0.8 * len(full_dataset))
num_val = len(full_dataset) - num_train
train_ds, val_ds = torch.utils.data.random_split(full_dataset, [num_train, num_val])

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)

print(f"Train samples: {len(train_ds)}")
print(f"Val samples: {len(val_ds)}")

# MODEL ( MODEL TRAININGNYA GANTI PAKE MobilenetV3 )
model = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
num_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(num_features, 2)
model = model.to(device)

# SETTING LOSS ( TINGKAT KESALAHAN ( MAKIN KECIL MAKIN BAGUS ) )
# CrossEntropyLoss = Function ( BUAT NGURANGIN OVERFITTING )
loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

# OPTIMIZER & SCHEDULER ( ALGORITMA YG DIPAKE (ARTINYA APA CARI DI GOOGLE, PANJANG NJING INIMAH) )
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=10)

# EARLY STOPPING CONFIG ( AUTO STOP KLO GADA PROGRES WAKTU TRAINING, BIAR GA OVERFITTING )
early_stop_patience = 5
early_stop_min_delta = 0.001
best_val_loss = float('inf')
early_stop_counter = 0

# TRAINING LOOP ( MASIH BINGUNG.? CARI DI GOOGLE, BINGUNG AKU JELASINNYA WKWKWK )
epochs = 20

# PLOTTING ( SETTING ITU YG GARIS-GARIS DI TRAINING RESULTS )
history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": []
}

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    scheduler.step()

    epoch_loss = running_loss / total
    epoch_acc = correct / total * 100

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_epoch_loss = val_loss / val_total
    val_epoch_acc = val_correct / val_total * 100

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"- Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.2f}% "
          f"- Val Loss: {val_epoch_loss:.4f} - Val Acc: {val_epoch_acc:.2f}%")

    history["train_loss"].append(epoch_loss)
    history["val_loss"].append(val_epoch_loss)
    history["train_acc"].append(epoch_acc)
    history["val_acc"].append(val_epoch_acc)

    # EARLY STOPPING CHECK
    if val_epoch_loss + early_stop_min_delta < best_val_loss:
        best_val_loss = val_epoch_loss
        early_stop_counter = 0
        # SAVE BEST MODEL ( INI YG DIPAKE DI DETECTION )
        torch.save(model.state_dict(), "results/anti_spoof_model.pth")
        print("✅ Saved new best model.")
    else:
        early_stop_counter += 1
        print(f"No improvement. Early stopping counter: {early_stop_counter}/{early_stop_patience}")
        if early_stop_counter >= early_stop_patience:
            print("⛔ Early stopping triggered!")
            break

# SAVE FINAL MODEL ( BELOM TENTU BAGUS, BISA NAIK BISA TURUN )
# MAKANYA ADA SETTING EARLY STOP
os.makedirs("results", exist_ok=True)
torch.save(model.state_dict(), "results/anti_spoof_model_final.pth")
print("✅ Model saved to results/anti_spoof_model_final.pth")

# VISUALIZE TRAINING ( INI BUAT NAMPILIN HASILNYA BENTUK GRAFIS )
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history["train_loss"], label='Train Loss')
plt.plot(history["val_loss"], label='Val Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Loss Curve")

plt.subplot(1,2,2)
plt.plot(history["train_acc"], label='Train Acc')
plt.plot(history["val_acc"], label='Val Acc')
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.title("Accuracy Curve")

os.makedirs("results", exist_ok=True)
plt.tight_layout()
plt.savefig("results/training_curves.png")
plt.close()
print("✅ Training curves saved to results/training_curves.png")

# BATCH PREDICTION PREVIEW
data_iter = iter(val_loader)
images, labels = next(data_iter)
images = images.to(device)
labels = labels.to(device)

model.eval()
with torch.no_grad():
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

idx_to_class = {v: k for k, v in full_dataset.class_to_idx.items()}

num_images = min(8, len(images))
images = images[:num_images].cpu()
labels = labels[:num_images].cpu()
predicted = predicted[:num_images].cpu()

plt.figure(figsize=(16, 6))
for i in range(num_images):
    img = images[i].permute(1, 2, 0).numpy()
    plt.subplot(2, 4, i+1)
    plt.imshow(img)
    gt = idx_to_class[labels[i].item()]
    pred = idx_to_class[predicted[i].item()]
    plt.title(f"GT: {gt} / Pred: {pred}",
              color="green" if gt == pred else "red")
    plt.axis("off")

plt.suptitle("Batch Prediction Preview", fontsize=16)
plt.tight_layout()
plt.savefig("results/batch_predictions.png")
plt.close()
print("✅ Batch prediction preview saved to results/batch_predictions.png")
