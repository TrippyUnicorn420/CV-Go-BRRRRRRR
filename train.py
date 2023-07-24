import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
import torch.nn as nn


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train = datasets.ImageFolder("fishies/train/", transform=transform)
val = datasets.ImageFolder("fishies/valid/", transform=transform)
test = datasets.ImageFolder("fishies/test/", transform=transform)

train_loader = torch.utils.data.DataLoader(train, batch_size=16, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size=16)
test_loader = torch.utils.data.DataLoader(test, batch_size=16)


model = models.resnet50(pretrained=True)
model.fc = nn.Linear(2048, len(train.classes))
model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    val_accuracy = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.cuda(), labels.cuda()
            predictions = model(images)
            val_accuracy += (predictions.argmax(1) == labels).float().mean()
    val_accuracy /= len(val_loader)
    print("Epoch:", epoch, "Val Accuracy:", val_accuracy.item())

# Testing
model.eval()
accuracy = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.cuda(), labels.cuda()
        predictions = model(images)
        accuracy += (predictions.argmax(1) == labels).float().mean()

accuracy /= len(test_loader)
print("Accuracy:", accuracy.item())