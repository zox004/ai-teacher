from PIL import Image
# %matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import numpy as np

trans = transforms.Compose([transforms.Resize((100,100)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))
                            ])
trainset = torchvision.datasets.ImageFolder(root = "/Users/sinq/Desktop/PythonWorkspace/AITeacher/images",
                                            transform = trans)
# print(trainset.__getitem__(5))

classes = trainset.classes
print(classes)
print("hello")