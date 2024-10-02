import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms 
import torchvision.models as models
import torchvision.datasets 
from torch.utils.data import DataLoader
import torchvision
import os
from PIL import Image
#version 1.0

dev =torch.device('cuda' if torch.cuda.is_available else 'cpu')
transforme=torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)), torchvision.transforms.ToTensor()])

num_epoch=10
batch_size=4
learing_rat=0.001
train_dir = 'D:\\Data\\catsAndDogs\\train\\New folder\\New folder'
test_dir = 'D:\\Data\\catsAndDogs\\test1'

class data(torch.utils.data.Dataset):
    def __init__(self , image_file , transform):
        
        self.image_file=image_file
        self.transformer=transform
    def __getitem__(self , id):
        img , label= self.image_file[id]
        img=Image.open(img).convert('RGB')
        img=self.transformer(img)
        return img , label
    
    def __len__(self):
        return len(self.image_file)
    
image_file =[]

for fille in os.listdir(train_dir):
    if fille.startswith('cat'):
        image_file.append((os.path.join(train_dir,fille),0))
    elif fille.startswith('dog'):
        image_file.append((os.path.join(train_dir,fille),1))


trainData=data(image_file , transforme)   
train=DataLoader(trainData , batch_size=batch_size ,shuffle=True)

testData=torchvision.datasets.ImageFolder(test_dir, transforme)
test=DataLoader(testData , batch_size=batch_size ,shuffle=True)

model =torchvision.models.resnet18(pretrained=True)

model.to(dev)

opti=optim.Adam(model.parameters(),lr=learing_rat)

los=torch.nn.CrossEntropyLoss()
best_accuracy=0.0
for epoch in range(num_epoch):
    model.train()
    for images , labels in  train :
        images , labels= images.to(dev) , labels.to(dev)
        opti.zero_grad()
        outpuut=model(images)
        loss=los(outpuut, labels)
        loss.backward()
        opti.step()

    
    model.eval()
    val_correct = 0


    with torch.no_grad():
        for images, labels in test:
            images, labels = images.to(dev), labels.to(dev)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
    
    val_accuracy = val_correct / len(test.dataset) * 100
    
    print(f'Epoch {epoch+1}, '
          f' Accuracy: {val_accuracy:.2f}%')
    
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_model.pth')

print('Finished Training')


def nothing():
    return 0