from torchvision import datasets, transforms
from torch.utils.data import DataLoader

TRAIN_PATH = r"D:\AAA-Bird_Species_Recognition\dataset\CUB-200-2011\train"  # 改成你的真实路径，r是别乱反斜杠

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(
    root=TRAIN_PATH,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True
)

print("Number of training images:", len(train_dataset))
print("Number of classes:", len(train_dataset.classes))
print("First 5 classes:", train_dataset.classes[:5])

images, labels = next(iter(train_loader))
print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
