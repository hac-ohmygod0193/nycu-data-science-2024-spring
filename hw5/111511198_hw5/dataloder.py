from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import PIL
from PIL import Image
import os, glob, cv2
import torch
import random
import linecache
import numpy as np



class MAKEUP(Dataset):
    def __init__(self, image_path, transform, mode, transform_mask, cls_list):
        self.image_path = image_path
        self.transform = transform
        self.mode = mode
        self.transform_mask = transform_mask
 
        self.A_seg = glob.glob(os.path.join(image_path, 'parsing', 'non-makeup', '*.png'))
        self.B_seg = glob.glob(os.path.join(image_path, 'parsing', 'makeup', '*.png'))
 
        split_symb = '\\' if  '\\' in self.A_seg[0] else '/'
 
        self.As = [os.path.join(image_path, 'images', 'non-makeup', x.split(split_symb)[-1]) for x in self.A_seg]
        self.Bs = [os.path.join(image_path, 'images', 'makeup', x.split(split_symb)[-1]) for x in self.B_seg]
 
        self.noiA = len(self.As)
        self.noiB = len(self.Bs)

    def __getitem__(self, index):
        if self.mode == 'train':
            idxA = random.choice(range(self.noiA))
            idxB = random.choice(range(self.noiB))

            mask_A = Image.open(self.A_seg[idxA]).convert("RGB")
            mask_B = Image.open(self.B_seg[idxB]).convert("RGB")
            
            image_A = Image.open(self.As[idxA]).convert("RGB")
            image_B = Image.open(self.Bs[idxB]).convert("RGB")
            
            image_A = Image.fromarray(cv2.resize(np.array(image_A), (256, 256)))
            image_B = Image.fromarray(cv2.resize(np.array(image_B), (256, 256)))
            return self.transform(image_A), self.transform(image_B), self.transform_mask(mask_A), self.transform_mask(mask_B)

    def __len__(self):
        if self.mode == 'train' or self.mode == 'train_finetune':
            num_A = len(self.As)
            num_B = len(self.Bs)
            return max(num_A, num_B)
        
        elif self.mode in ['test', "test_baseline", 'test_all']:
            num_A = len(self.As)
            num_B = len(self.Bs)
            return num_A * num_B

def ToTensor(pic):
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float()
    else:
        return img

def get_loader(config, mode="train"):
    # return the DataLoader
    dataset_name = config.dataset
    data_path = config.data_path
    transform = transforms.Compose([
    transforms.Resize(config.img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    transform_mask = transforms.Compose([
        transforms.Resize(config.img_size, interpolation=PIL.Image.NEAREST),
        ToTensor])
    print(config.data_path)
    #"""
    if mode=="train":
        dataset_train = eval(dataset_name)(data_path, transform=transform, mode= "train",
                                            transform_mask=transform_mask, cls_list = config.cls_list)
        dataset_test = eval(dataset_name)(data_path, transform=transform, mode= "test",
                                            transform_mask=transform_mask, cls_list = config.cls_list)

        #"""
        data_loader_train = DataLoader(dataset=dataset_train, batch_size=config.batch_size, shuffle=True)
    
    if mode=="test":
        data_loader_train = None
        dataset_test = eval(dataset_name)(data_path, transform=transform, mode= "test",\
                                                            transform_mask =transform_mask, cls_list = config.cls_list)

    data_loader_test = DataLoader(dataset=dataset_test,
                             batch_size=1,
                             shuffle=False)

    return [data_loader_train, data_loader_test]

# generate prediction images
class ImageDataset(Dataset):
    def __init__(self, image_root, txt_file, transform=None):
        self.image_root = image_root
        self.files = self._load_file_paths(txt_file)
        self.transform = transform

    def _load_file_paths(self, txt_file):
        with open(txt_file, 'r') as file:
            file_paths = [line.strip() for line in file.readlines()]
        return file_paths

    def __getitem__(self, index):
        img_path = os.path.join(self.image_root, self.files[index % len(self.files)])

        img = Image.open(img_path).convert("RGB")
        img = Image.fromarray(cv2.resize(np.array(img), (256, 256)))
        if self.transform:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.files)
def get_loader_output(args):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    path = args.data_path
    # Instantiate datasets
    dataset_A = ImageDataset(image_root=path+'images/', txt_file=path+'nomakeup_test.txt', transform=transform)
    dataset_B = ImageDataset(image_root=path+'images/', txt_file=path+'makeup_test.txt', transform=transform)
    # Create dataloaders
    dataloader_A = DataLoader(dataset_A, batch_size=1)
    dataloader_B = DataLoader(dataset_B, batch_size=1)
    return dataloader_A,dataloader_B