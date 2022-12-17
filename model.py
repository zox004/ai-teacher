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

# print(fm.findSystemFonts(fontpaths=None, fontext='ttf'))

# 한글 폰트 설정하기
# fontpath = 'C:/Windows/Fonts/NanumGothicLight.ttf'
# font = fm.FontProperties(fname=fontpath, size=10).get_name()
# plt.rc('font', family=font)

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
    # test_datasets = datasets.ImageFolder(os.path.join(data_dir,'test'), transforms_test)


    train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=4, shuffle=True, num_workers=0)
    #valid_dataloader = torch.utils.data.DataLoader(valid_datasets, batch_size=4, shuffle=False, num_workers=0)

    print('학습 데이터셋 크기:', len(train_datasets))
    #print('테스트 데이터셋 크기:', len(valid_datasets))

    class_names = train_datasets.classes
    print('학습 클래스:', class_names)

    # 학습 데이터를 배치 단위로 불러오기
    iterator = iter(train_dataloader)
    # 현재 배치를 이용해 격자 형태의 이미지를 만들어 시각화
    inputs, classes = next(iterator)
    out = torchvision.utils.make_grid(inputs)
    # print(model)
    print(model.fc.in_features)
    # imshow(out, title=[class_names[x] for x in classes])
    num_features = model.fc.in_features
    

    # transfer learning
    model.fc = nn.Sequential(     
        nn.Linear(num_features, 256),        # 마지막 완전히 연결된 계층에 대한 입력은 선형 계층, 256개의 출력값을 가짐
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, len(class_names)),      # Since 10 possible outputs = 10 classes
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

    os.makedirs('./weight',exist_ok=True)
    # torch.save(model, './weight/model_best_epoch.pt')

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
def prediction():
    train_datasets = datasets.ImageFolder(os.path.join(data_dir,'train'), transforms_train)
    class_names = train_datasets.classes

    valid_dir = data_dir + '/test'
    model = torch.load("./weight/model_best_epoch.pt")
    # model = models.resnet50(weights="IMAGENET1K_V1")
    model.eval()
    valid_images = []

    # test data에 있는 이미지 val_folders에 모아놓기
    valid_image = glob(valid_dir + '/*')

    # for val_folder in val_folders:
    #     image_paths = glob(val_folder + '/*')
    #     for image_path in image_paths:
    #         valid_images.append(image_path)
    
    # print(len(valid_images))
    # num = random.randint(0,len(valid_images)-1)
    # valid_image = valid_images[num]

    image = Image.open(valid_image[0])
    image = transforms_test(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # origin code
        # outputs = model(image)
        # _, preds = torch.max(outputs, 1)
        # imshow(image.cpu().data[0], title=' Classification : ' + class_names[preds[0]])
    
        # modifying code - show answer percent
        prediction = model(image).squeeze(0).softmax(0)
        for i in range(len(class_names)):
            print(f"{class_names[i]}: {round(100 * prediction[i].item())}%")
    if os.path.exists(valid_dir):
        shutil.rmtree(valid_dir)
    
        
# def imshow(input, title):
#     # torch.Tensor를 numpy 객체로 변환
#     input = input.numpy().transpose((1, 2, 0))
#     # 이미지 정규화 해제하기
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     input = std * input + mean
#     input = np.clip(input, 0, 1)
#     # 이미지 출력
#     plt.imshow(input)
#     plt.title(title)
#     plt.show()
    

if __name__ == "__main__" :
    train()
    # prediction()