import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import time
import os

from glob import glob

from PIL import Image
import random
import shutil

from pymongo import MongoClient
import pymongo
import uuid
import datetime
import pprint
from bson.objectid import ObjectId
from gridfs import GridFS


# Connect to the server
client = MongoClient('mongodb+srv://aiteacher:1234@aiteacher.2urehvj.mongodb.net/?retryWrites=true&w=majority')
# Connect to the database
mydb = client['aiteacher']
# Connect to the Collection
coll = mydb['data']
# Create a GridFS bucket
gfs = GridFS(mydb)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = './data'

# 데이터셋을 불러올 때 사용할 변형(transformation) 객체 정의
transforms_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transforms_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(), # 데이터 증진(augmentation)
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 정규화(normalization)
])


def train(model = models.resnet50(weights=None)):
    train_datasets = datasets.ImageFolder(os.path.join(data_dir,'train'), transforms_train)

    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=4, shuffle=True, num_workers=0)

    print('학습 데이터셋 크기:', len(train_datasets))

    class_names = train_datasets.classes
    print('학습 클래스:', class_names)

    # 학습 데이터를 배치 단위로 불러오기
    iterator = iter(train_dataloader)
    # 현재 배치를 이용해 격자 형태의 이미지를 만들어 시각화
    inputs, classes = next(iterator)
    out = torchvision.utils.make_grid(inputs)
    # imshow(out, title=[class_names[x] for x in classes])
    num_features = model.fc.in_features
    

    # transfer learning
    model.fc = nn.Sequential(     
        nn.Linear(num_features, 256),        # 마지막 완전히 연결된 계층에 대한 입력은 선형 계층, 256개의 출력값을 가짐
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, len(class_names)),
        nn.LogSoftmax(dim=1)              # For using NLLLoss()
    )

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    model = model.to(device)

    num_epochs = 5

    best_epoch = None
    best_loss = 5

    ''' Train '''
    # 전체 반복(epoch) 수 만큼 반복하며
    total_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        
        running_loss = 0.
        running_corrects = 0

        # 배치 단위로 학습 데이터 불러오기
        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 모델에 입력(forward)하고 결과 계산
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # 역전파를 통해 기울기(gradient) 계산 및 학습 진행
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_datasets)
        epoch_acc = running_corrects / len(train_datasets) * 100.

        # 학습 과정 중에 결과 출력
        print('#{} Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch, epoch_loss, epoch_acc, time.time() - start_time))

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_epoch = epoch
            print("best_loss: {:.4f} \t best_epoch: {}".format(best_loss, best_epoch))
    print(f"Total Running Time: {time.time() - total_time}")
    
    os.makedirs('./weight',exist_ok=True)
    # torch.save(model, './weight/model_best_epoch.pt')
    
    torch.save(model.state_dict(), './weight/model_best_epoch.pt')
    
    # gfs.put(model, filename='model_best_epoch.pt')
    
    return model
    

# Valid 
    # with torch.no_grad():
    #     model.eval()
    #     start_time = time.time()
        
    #     running_loss = 0.
    #     running_corrects = 0

    #     for inputs, labels in valid_dataloader:
    #         inputs = inputs.to(device)
    #         labels = labels.to(device)

    #         outputs = model(inputs)
    #         _, preds = torch.max(outputs, 1)
    #         loss = criterion(outputs, labels)

    #         running_loss += loss.item() * inputs.size(0)
    #         running_corrects += torch.sum(preds == labels.data)

    #         # 한 배치의 첫 번째 이미지에 대하여 결과 시각화
    #         print(f'[예측 결과: {class_names[preds[0]]}] (실제 정답: {class_names[labels.data[0]]})')
    #         imshow(inputs.cpu().data[0], title='예측 결과: ' + class_names[preds[0]])

    #     epoch_loss = running_loss / len(valid_datasets)
    #     epoch_acc = running_corrects / len(valid_datasets) * 100.
    #     print('[valid Phase] Loss: {:.4f} Acc: {:.4f}% Time: {:.4f}s'.format(epoch_loss, epoch_acc, time.time() - start_time))




''' Prediction '''
def prediction(trained_model):
    train_datasets = datasets.ImageFolder(os.path.join(data_dir,'train'), transforms_train)
    class_names = train_datasets.classes

    torch.save(trained_model, './weight/model_best_epoch.pt')
    valid_dir = data_dir + '/test'
    model = trained_model
    # model = models.resnet50(weights="IMAGENET1K_V1")
    model.eval()
    valid_images = []

    # test data에 있는 이미지 val_folders에 모아놓기
    valid_image = glob(valid_dir + '/*')

    image = Image.open(valid_image[0])
    image = transforms_test(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(image).squeeze(0).softmax(0)
        for i in range(len(class_names)):
            print(f"{class_names[i]}: {round(100 * prediction[i].item())}%")
    if os.path.exists(valid_dir):
        shutil.rmtree(valid_dir)
    

if __name__ == "__main__" :
    trained_model = train()
    prediction(trained_model)
    