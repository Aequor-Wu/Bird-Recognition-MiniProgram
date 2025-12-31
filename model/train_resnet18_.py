# 唯一训练脚本（可冻结/可全训练）
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# ========= 路径 =========
TRAIN_DIR = r"D:\AAA-Bird_Species_Recognition\dataset\CUB-200-2011\train"
VAL_DIR   = r"D:\AAA-Bird_Species_Recognition\dataset\CUB-200-2011\test"

# ========= 参数 =========
BATCH_SIZE = 8
NUM_EPOCHS = 10
NUM_CLASSES = 200
LR = 0.001  #当为冻结模型“LR = 0.001” | 为全训练“LR = 1e-4”
FREEZE_BACKBONE = True  # True: 冻结骨干网络 | False: 全模型训练

# ========= 数据 =========
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
val_dataset   = datasets.ImageFolder(VAL_DIR, transform=val_tf)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ========= 模型 =========
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

if FREEZE_BACKBONE:
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if device.type == 'cuda':
    print("GPU name:", torch.cuda.get_device_name(0))

print("Freeze backbone:", FREEZE_BACKBONE)

model.to(device)

# ========= 训练 =========
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # ===== 验证阶段 =====
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
          f"Train Loss: {avg_loss:.4f} "
          f"Val Acc: {val_acc*100:.2f}%")
