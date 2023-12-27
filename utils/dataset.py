from PIL import Image
import numpy as np
import json
import argparse
import torch
from tqdm import tqdm
from torch import nn
from torchvision import transforms

paths_dict = {
            'train' : 'Data_Paths/train.json',
            'test' : 'Data_Paths/test.json',
            }

label_dict = {
            6:0,
            20:1,
            82:2
            }

class DatitaSet(torch.utils.data.Dataset):
    def __init__(self, mode, n_classes, img_dim):
        super().__init__()
        self.mode = mode
        self.n_classes = n_classes
        self.img_dim = img_dim

        aug_transforms = {
                        "train": transforms.Compose([
                            transforms.Resize((img_dim,img_dim)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))
                        ]),
                        "test": transforms.Compose([
                            transforms.Resize((img_dim,img_dim)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5,0.5,0.5))
                        ]),
                        }

        img_paths, labels = self.get_paths()
        self.transform = aug_transforms[self.mode]
        patches, labels = self.transform_images(img_paths, labels)

        ## Elementos del dataset
        self.X = patches
        self.y = torch.from_numpy(labels)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return [self.X[i], self.y[i]]
    
    def get_paths(self):
        path = paths_dict[self.mode]
        with open(path, 'r') as file:
            paths_json = json.load(file)
        class_paths = paths_json[str(self.n_classes)]
        aug_factor = self.get_augmentation_factor()
        paths = []
        labels = []
        for idx in range(self.n_classes):
            _class = class_paths[str(idx)]
            aug_class = self.augmentate_class(_class, aug_factor)
            for item in aug_class:
                paths.append(item)
                labels.append(idx)
        return paths, labels
    
    def transform_images(self, imgs_paths, labels):
        patches = []
        labs = []
        print(f'Reading Dataset with {self.n_classes} classes.')
        good_imgs = 0
        for i in tqdm(range(len(imgs_paths))):
            try:
                img = Image.open(f'Images/{imgs_paths[i]}')
                patched_img = self.transform(img)
                patches.append(patched_img)
                img.close()
                labs.append(int(labels[i]))
                good_imgs += 1
            except:
                pass
        print('Stored Images: ', good_imgs)
        return patches, np.array(labs)
    
    def get_augmentation_factor(self):
        with open(f'Data_paths/{self.mode}_class_count.json', 'r+') as file:
            count_dict = json.load(file)
        
        class_count = count_dict[str(self.n_classes)]
        max_count, min_count = max(class_count), min(class_count)
        factor = min_count + int(0.5*(max_count-min_count))
        return factor

    def augmentate_class(self, aug_paths, factor):
        if len(aug_paths) > factor or self.mode=='test':
            return aug_paths
        else:
            indexes = np.random.choice(np.arange(0, len(aug_paths)), size=(factor-len(aug_paths)), replace=True)
            for idx in indexes:
                aug_paths.append(aug_paths[idx])
            return aug_paths


def main(mode, n_classes, img_dim, pt = False):
    if mode == "pretrained":
        #ds = DatitaSet(mode='train', n_classes=n_classes, img_dim=384, patch_size=24, pretrained=True)
        #torch.save(ds, f'Datasets/pt_train_{n_classes}.pkl')
        ds = DatitaSet(mode='test', n_classes=n_classes, img_dim=384, pretrained=True)
        torch.save(ds, f'Datasets/pt_test_{n_classes}.pkl')

    ds = DatitaSet(mode=mode, n_classes=n_classes, img_dim=img_dim)
    print(f'Dataset {mode} for {n_classes} created with:\n')
    print(f'{len(ds)} readable images\n')
    print(f'{ds[0][0].size()} tensor size\n')
    if pt:
        torch.save(ds, f'Datasets/pt_train_{n_classes}.pkl')
    else:
        torch.save(ds, f'Datasets/{mode}_{n_classes}_{img_dim}.pkl')




