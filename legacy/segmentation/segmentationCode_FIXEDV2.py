import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import segmentation_models_pytorch as smp
from skimage.io import imread, imsave
from skimage.transform import resize
import random
import torchvision.transforms as transforms
import albumentations as A
import SimpleITK as sitk
from skimage.util import random_noise
from scipy.ndimage import binary_dilation
from PIL import Image as im



class ImageDataset(Dataset):
    def __init__(self, datacsv, image_transform, general_transform):
        self.image_transform = image_transform
        self.general_transform = general_transform
        self.images_paths = []
        self.annotations_paths = []
        self.imgAndLables = []
        self.imageidx = []
        self.patientidx = []

        with open(datacsv) as file:
            content = file.readlines()
        data = [w.replace('\n','').split(',') for w in content[1:]]


        self.patientidx = sorted([w[0] for w in data])
        self.imageidx = sorted([w[1] for w in data])
        self.images_paths = sorted([w[2] for w in data])
        self.annotations_paths = sorted([w[3] for w in data])
        # self.annotations_paths = sorted(glob.glob(f'{root}/**/*.dcm', recursive=True))
        self.imgAndLables = list(zip(self.patientidx, self.imageidx, 
                                    self.images_paths, self.annotations_paths))
        # random.shuffle(self.imgAndLables)
        
    def __len__(self):
        return len(self.imgAndLables)

    def __getitem__(self, idx):
        image = sitk.ReadImage(self.imgAndLables[idx][2])
        image_array = sitk.GetArrayFromImage(image)
        mask = sitk.ReadImage(self.imgAndLables[idx][3]) 
        mask_array = sitk.GetArrayFromImage(mask)
        
        mask_array = np.expand_dims(np.rot90(mask_array[0],-1),0)
        mask_array = np.expand_dims(np.fliplr(mask_array[0]),0)
        mask_array = binary_dilation(mask_array,iterations=2).astype(float)


        if self.general_transform:
            # temp= image_array
            transformed= self.general_transform(image=image_array,mask=mask_array)
            image_array = transformed['image']
            mask_array = transformed['mask']

            # mask_array = self.general_transform(mask_array)

            # image_array, mask_array = self.general_transform(image=image_array, masks=mask_array)
            # transformed = transform(image=image, masks=masks)

            # params = self.general_transform(self.general_transform.degree, self.general_transform.translate, self.general_transform.scale, self.general_transform.shear, self.general_transform.size)
            # image_array = transforms.affine(image_array, *params)
            # mask_array = transforms.affine(mask_array, *params)

        if self.image_transform:
            # print(image_array)
            # image = im.fromarray(image_array.astype('uint8'))        
            image_array = self.image_transform(image_array)
            # for t in self.image_transform:
            #     image = t(image = image_array)['image']
            # image_array = sitk.GetArrayFromImage(image)


        return image_array, mask_array

def read_image(path):
    image = sitk.ReadImage(path) 
    image_array = sitk.GetArrayFromImage(image)

    return image_array


def visualize(image):
    plt.figure()
    plt.imshow(image.squeeze(), cmap='gray')
    plt.axis('off')
    plt.show()

def dice_metric(inputs, target):
    intersection = 2.0 * (target * inputs).sum()
    union = target.sum() + inputs.sum()
    if union == 0:
        return 1.0

    return intersection / union

general_transform = A.Compose([
    A.GaussNoise(p=0.5),
    A.Flip(p=0.3),
    A.HorizontalFlip(p=0.3),
    A.Rotate([-15,15],p=0.5),

])
image_transform = transforms.Compose([
    # transforms.ColorJitter(brightness=0.2, contrast=0.2),
    # A.ToFloat(max_value=1.1),
    # transforms.Lambda(lambda x: np.clip(x, -10, 200)),
    transforms.Lambda(lambda x: np.clip(x, -50, 300)),

    # transforms.ToTensor(),
    # A.ToFloat(max_value=1.1),
    # transforms.Normalize(mean=-666, std=475),
    # transforms.ToTensor(),
])



traindata = ImageDataset('traindataV2.csv',image_transform= image_transform,general_transform= general_transform)
# traindata1 = ImageDataset('traindataV2.csv', None, None)
# traindata = ImageDataset('traindata1.csv',image_transform= image_transform,general_transform= general_transform)

valdata = ImageDataset('valdataV2.csv',image_transform= image_transform,general_transform= general_transform)


torch.cuda.empty_cache()



train_dataloader = DataLoader(traindata, 32, shuffle = True, num_workers= 4)
val_dataloader = DataLoader(valdata, 32, num_workers=4)
# for img,msk in train_dataloader:
    # print(img)
# i = random.randint(0,2000)
# img, msk = traindata[20]
# visualize(img)

# visualize(msk)
# visualize(traindata1[20][0])

# visualize(traindata1[150][1])

# for i in range(10,500,10):
#     print(i)
#     visualize(traindata1[i][1])

# visualize(msk[1])
# print(traindata[0])












device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



model = smp.Unet("resnet34", encoder_weights="imagenet", in_channels=1, classes=1)


num_gpus = torch.cuda.device_count()

if num_gpus > 1:
    model = torch.nn.DataParallel(model)

model.to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


num_epochs = 50

loss_arrtrain = []
loss_arrval = []

best_acc = 0
patience = 0
max_patience = 5

for epoch in range(num_epochs):
    losstotal = 0
    total = 0
    correct_pred = 0
    model.train()

    batch_idx = 0

    for inputs, targets in train_dataloader:

        total += len(targets)

        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs.float())
        loss = criterion(outputs, targets.float())
        losstotal += loss * len(targets)
        correct_pred += dice_metric((outputs>0.5).float(),targets.float())

        loss.backward()
        optimizer.step()

        # if batch_idx < 10:
        #     with open('logbatches2v3.txt', 'a') as the_file:
        #         the_file.write(f'BATCH:{batch_idx},\t Loss: {loss}\n')

        batch_idx += 1


    losstotal /= total
    correct_pred /= total

    loss_arrtrain.append(losstotal)

    with open('logSegmentationFIXEDV2.txt', 'a') as the_file:
        the_file.write(f'{epoch},\t Loss Train: {losstotal}, \t DSC:{correct_pred}')


    

    total = 0
    losstotal = 0
    correct_pred = 0

    batch_idx = 0
    with torch.no_grad():
        for inputs, targets in val_dataloader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            total += len(targets)

            outputs = model(inputs.float())

            loss = criterion(outputs, targets.float())
            
            losstotal += loss * len(targets)
            correct_pred += dice_metric((outputs>0.5).float(),targets.float())


            batch_idx += 1

    
    correct_pred /= total
    losstotal /= total

    loss_arrval.append(losstotal)

    with open('logSegmentationFIXEDV2.txt', 'a') as the_file:
        the_file.write(f'\t Loss Val: {losstotal}, \t DSC:{correct_pred}\n')

    if epoch == 0 or best_acc >= losstotal:
        torch.save(model.state_dict(), 'modelsegmentationFIXEDV2.pt')
        patience = 0
        best_acc = losstotal
    else:
        patience +=1
    if patience == max_patience:
        break





with open('logSegmentationFIXEDV1.txt', 'a') as the_file:
        the_file.write(f'\n\n{loss_arrtrain} \n\n\n {loss_arrval} \n')


# torch.save(model.state_dict(), 'modelsegmentationV4.pt')
