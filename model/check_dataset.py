from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 修改成你自己的数据集路径
DATASET_PATH = "D:\AAA-Bird_Species_Recognition\CUB-200-2011"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(
    root=DATASET_PATH,
    transform=transform
)

loader = DataLoader(dataset, batch_size=8, shuffle=True)

print("Number of images:", len(dataset))
print("Number of classes:", len(dataset.classes))
print("First 5 classes:", dataset.classes[:5])

images, labels = next(iter(loader))
print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
