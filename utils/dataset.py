from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from torch import nn
from torchvision import transforms

from .patch_embedding import PatchTransform

paths_dict = {
            'train' : 'Data_Paths/train.txt',
            'test' : 'Data_Paths/train.txt'
            }

label_dict = {
            6:0,
            20:1,
            82:2
            }

class DatitaSet(torch.utils.data.Dataset):
    def __init__(self,mode,n_classes, img_dim=128, patch_size = 16):
        super().__init__()
        self.mode = mode
        self.n_classes = n_classes
        self.img_dim = img_dim
        self.patch_size = patch_size

        if self.img_dim%patch_size != 0:
            raise Exception(f"Image size ({self.img_dim}) must be divisible by patch_size ({patch_size})")

        aug_transforms = {
                        "train": transforms.Compose([
                            transforms.Resize((img_dim,img_dim)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            PatchTransform(patch_size=self.patch_size)
                        ]),
                        "test": transforms.Compose([
                            transforms.Resize((img_dim,img_dim)),
                            transforms.ToTensor(),
                            PatchTransform(patch_size=self.patch_size)
                        ]),
                        }

        data = self.read_paths()
        self.transform = aug_transforms[self.mode]
        img_paths, label_6, label_20, label_82 = zip(*data)

        patches, labels = self.transform_images(img_paths, [label_6,label_20,label_82][label_dict[self.n_classes]])

        ## Elementos del dataset
        self.X = patches
        self.y = torch.from_numpy(labels)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return [self.X[i], self.y[i]]
    
    def read_paths(self):
        path = paths_dict[self.mode]
        data = []
        with open(path, 'r') as file:
            for line in file.readlines():
                if line:
                    data.append(line.split(','))
        return data
    
    def transform_images(self, imgs_paths, labels):
        patches = []
        labs = []
        print('Reading Dataset')
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
        print(good_imgs)
        return patches, np.array(labs)

if __name__ == '__main__':
    test = DatitaSet(mode='train', n_classes=6)
    print(len(test))
    print(test[0])
    print(test[0][0].size())
    print(test.y)



