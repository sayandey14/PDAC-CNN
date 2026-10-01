import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import segmentation_models_pytorch as smp
from skimage.io import imread, imsave
from skimage.transform import resize
import random
import SimpleITK as sitk
from PIL import Image as im
import cv2
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.augmentations.transforms import ToFloat
import pydicom
import torchvision
import glob
from scipy import ndimage


import time



class ImageDataset(Dataset):
    def __init__(self, normal_scans_path, abnormal_scans_path, image_transform, general_transform):
        self.image_transform = image_transform
        self.general_transform = general_transform
        self.images_paths = []
        self.label = []
        self.imgAndLables = []
        self.patientidx = []
        # self.output = []

        normal_images_paths = f'{normal_scans_path}/**/*.nii.gz'
        abnormal_images_paths = f'{abnormal_scans_path}/**/*.nii.gz'


        abnormal_images_paths = sorted(glob.glob(abnormal_images_paths, recursive=True))
        normal_images_paths = sorted(glob.glob(normal_images_paths, recursive=True))

        # self.patientidx = sorted([w[0] for w in data])
        # self.imageidx = sorted([w[1] for w in data])
        # self.images_paths = sorted([w[2] for w in data])
        # self.annotations_paths = sorted([w[3] for w in data])

        self.patientidx = [i for i in range(len(normal_images_paths) + len(abnormal_images_paths))]
        self.images_paths = abnormal_images_paths + normal_images_paths
        self.label = [0 for item in abnormal_images_paths] + [1 for item in normal_images_paths]
        self.imgAndLables = list(zip(self.patientidx, self.images_paths, self.label ))
        
        # print(self.patientidx)
        random.shuffle(self.imgAndLables)
        
    def __len__(self):
        return len(self.imgAndLables)

    def __getitem__(self, idx):
        image = sitk.ReadImage(self.imgAndLables[idx][1])
        image_array = sitk.GetArrayFromImage(image)
        
        if self.general_transform is not None:
            transformed = self.general_transform(image=image_array)           
            image_array = transformed["image"]
            image_array = torch.from_numpy(image_array)

        if self.image_transform is not None:
            # image_array = self.image_transform(image_array)
            desired_depth = 256
            desired_width = 200
            desired_height = 256
            # Get current depth
            current_width = image_array.shape[0]
            current_height = image_array.shape[1]
            current_depth = image_array.shape[2]
            # Compute depth factor
            depth = current_depth / desired_depth
            width = current_width / desired_width
            height = current_height / desired_height
            depth_factor = 1 / depth
            width_factor = 1 / width
            height_factor = 1 / height
            # Rotate
            image_array = ndimage.rotate(image_array, 90, reshape=False)
            # Resize across z-axis
            image_array = ndimage.zoom(image_array, (width_factor, height_factor, depth_factor), order=1)
            

        return self.imgAndLables[idx][0], image_array, self.imgAndLables[idx][2]

def read_image(path):
    image = sitk.ReadImage(path) 
    image_array = sitk.GetArrayFromImage(image)

    return image_array


def visualize(image,depth):
    if depth < image.shape[0]:
        plt.figure()
        plt.imshow(image[depth,:,:].squeeze(), cmap='gray')
        plt.axis('off')
        plt.show()
    else:
        print('INVALID SLICE')

# general_transform = A.Compose([
#     # ToFloat(max_value=1.1),
#     # A.Resize([250,256,256]),
#     # ToTensorV2()
# ])

# image_transform = transforms.Compose([
#     transforms.Lambda(lambda x: np.clip(x, 10, 60)), 
#     transforms.Normalize(-666, 475),
#     transforms.Resize((200,256,256)),
# ])

data = ImageDataset('/scratch/student/aref/PancreasProj/resultV3','/scratch/student/aref/PancreasProj/resultV3new',
                         image_transform= 'None',general_transform= None)

traindata, valdata = torch.utils.data.random_split(data, [0.8, 0.2])

# valdata = ImageDataset('/scratch/student/aref/PancreasProj/result','/scratch/student/aref/PancreasProj/resultnew',
#                        image_transform= None,general_transform= None)

train_dataloader = DataLoader(traindata,32, num_workers=4,shuffle = True)

val_dataloader = DataLoader(valdata,32, num_workers=4,shuffle = True)

# for idx, image, label in train_dataloader:
#     # print(len(label))
#     # print(idx)
#     print(image[1].shape)
#     visualize(image[1],60)
#     break

# print(traindata[3][1].shape)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torchvision.models.googlenet(weights = torchvision.models.GoogLeNet_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)

num_gpus = torch.cuda.device_count()

if num_gpus > 1:
    model = torch.nn.DataParallel(model)

model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


num_epochs =100

loss_arrtrain = []
loss_arrval = []
acc_arrtrain = []
acc_arrval = []

best_acc = 0
patience = 0
max_patience = 10

for epoch in range(num_epochs):
    losstotal = 0
    acctotal = 0
    total = 0
    correct_pred = 0
    model.train()

    batch_idx = 0

    for idx, inputs, targets in train_dataloader:

        total += len(targets)

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs.float())
        _, preds = torch.max(outputs, 1)

        loss = criterion(outputs, targets)
        losstotal += loss * len(targets)
        loss.backward()
        optimizer.step()

        # if batch_idx <3:
        #     with open('logbatches2.txt', 'a') as the_file:
        #         the_file.write(f'BATCH:{batch_idx},\t P: {preds}\n{targets}\n{torch.sum(preds == targets)}\n')

        correct_pred += torch.sum(preds == targets)


        batch_idx += 1


    losstotal /= total
    acctotal = correct_pred / total

    loss_arrtrain.append(losstotal)
    acc_arrtrain.append(acctotal)

    with open('logClassifierV3.txt', 'a') as the_file:
        the_file.write(f'{epoch},\t Loss Train: {losstotal},\t Acc Train: {acctotal}')


    

    total = 0
    losstotal = 0
    acctotal = 0
    correct_pred = 0

    batch_idx = 0
    with torch.no_grad():
        for idx, inputs, targets in val_dataloader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            total += len(targets)

            outputs = model(inputs.float())
            _, preds = torch.max(outputs, 1)

            loss = criterion(outputs, targets)
            
            losstotal += loss * len(targets)
            correct_pred += torch.sum(preds == targets)


            batch_idx += 1

    
    losstotal /= total
    acctotal = correct_pred / total

    loss_arrval.append(losstotal)
    acc_arrval.append(acctotal)

    with open('logClassifierV3.txt', 'a') as the_file:
        the_file.write(f'\t Loss Val: {losstotal},\t Acc Val: {acctotal}\n')
    
    if acctotal > best_acc:
        torch.save(model.state_dict(), 'modelClassifierV3.pt')
        best_acc = acctotal
        patience = 0
    else:
        patience += 1
    if patience == max_patience:
        break


with open('logClassifierV3.txt', 'a') as the_file:
        the_file.write(f'\n\nTrainLossArray:{loss_arrtrain}\n\nValLossArray:{loss_arrval}\n\nTrainACCArray:{acc_arrtrain}\n\nValACCArray:{acc_arrval}')


# torch.save(model.state_dict(), 'modelClassifier.pt')
