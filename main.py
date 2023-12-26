import torch
import torchvision
from torchsummary import summary
import json
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from pytorch_pretrained_vit import ViT  as pretrained_ViT

from utils.train import train_model
import utils.dataset as ds
from utils.evaluate import test_model
from model.vit_model import ViT as my_ViT
import sys, os

sys.path.append('..')

N_ENCODERS = 1
N_HEADS = 3
IMG_SIZE =64
PATCH_SIZE = 8
HIDDEN_SIZE = 512
N_PATCHES = (IMG_SIZE//PATCH_SIZE)**2
N_CLASSES = 6
EPOCHS = 20 
BATCH_SIZE = 16



if __name__ == "__main__":
    ### Argument Parse
    parser = argparse.ArgumentParser(description='Main script for ViT proyect.')

    parser.add_argument('--mode',choices=["train", "test", "pretrained"], type=str, default="train", help='Mode to run.')
    parser.add_argument('--n_encoders', type=int, default=3, help='Number of encoders')
    parser.add_argument('--n_heads', type=int, default=5, help='Number of attention heads')
    parser.add_argument('--img_size', type=int, default=128, help='Image size')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size')
    parser.add_argument('--dim', type=int, default=768, help='Embedding Dimension')
    parser.add_argument('--hidden_size', type=int, default=512, help='Hidden size')
    parser.add_argument('--n_classes', type=int, default=6, help='Number of classes')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    args = parser.parse_args()

    file_name = f'{args.n_classes}_{args.batch_size}_{args.n_heads}_{args.n_encoders}_{args.img_size}_{args.patch_size}_{args.hidden_size}'

    ##### TRAIN MODE #####
    if args.mode == "train":
        ### Dataset Load / Generate
        if os.path.exists(f'Datasets/train_{args.n_classes}_{args.img_size}.pkl'):
            train_dataset = torch.load(f'Datasets/train_{args.n_classes}_{args.img_size}.pkl')
        else:
            if not os.path.exists(f'./Datasets'):
                os.mkdir('./Datasets')
            ds.main(args.mode, args.n_classes, args.img_size)
            train_dataset = torch.load(f'Datasets/train_{args.n_classes}_{args.img_size}.pkl')

        
        ### Path definitions
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dir_checkpoint = Path(f"./checkpoints/{file_name}/")

        print(f"Using: {device}")

        model = my_ViT(
            size=args.img_size,
            dim=args.dim,
            hidden_size=args.hidden_size,
            num_patches=(args.img_size//args.patch_size)**2,
            patch_size = args.patch_size,
            n_classes=args.n_classes,
            num_heads=args.n_heads,
            num_encoders=args.n_encoders,
        )

        model = model.to(memory_format=torch.channels_last)

        model.to(device=device)

        curves = train_model(
            model=model,
            device=device,
            dir_checkpoint=dir_checkpoint,
            epochs=args.epochs,
            batch_size=args.batch_size,
            dataset=train_dataset,)


        curves['val_loss'] = [val.cpu().detach().item() if type(val) != float else val  for val in curves['val_loss']]
        curves['train_loss'] = [val.cpu().detach().item() if type(val) != float else val for val in curves['train_loss']]
        curves['val_acc'] = [val.cpu().detach().item() if type(val) != float else val for  val in curves['val_acc']]
        curves['train_acc'] = [val.cpu().detach().item() if type(val) != float else val for val in curves['train_acc']]

        with open(f'Results/curves/curves_{file_name}.json', 'w') as file:
            json.dump(curves, file)


    ##### TEST MODE #####
    if args.mode == "test":
        model = my_ViT(
            size=args.img_size,
            dim=args.dim,
            hidden_size=args.hidden_size,
            num_patches=(args.img_size//args.patch_size)**2,
            patch_size = args.patch_size,
            n_classes=args.n_classes,
            num_heads=args.n_heads,
            num_encoders=args.n_encoders,
        )
        
        ### Dataset Load / Generate
        if os.path.exists(f'Datasets/test_{args.n_classes}_{args.img_size}.pkl'):
            test_dataset = torch.load(f'Datasets/test_{args.n_classes}_{args.img_size}.pkl')
        else:
            if not os.path.exists(f'./Datasets'):
                os.mkdir('./Datasets')
            ds.main(args.mode, args.n_classes, args.img_size)
            test_dataset = torch.load(f'Datasets/test_{args.n_classes}_{args.img_size}.pkl')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

        checkpoints = os.listdir(f'./checkpoints/{file_name}')

        accuracys = []
        cm_s = []

        for cp in checkpoints:
            state_dict = torch.load(f'./checkpoints/{file_name}/{cp}')
            del state_dict["labels"]
            model.load_state_dict(state_dict)
            acc, cm = test_model(model, test_dataloader, device, args.n_classes)

            accuracys.append(acc)
            cm_s.append(cm)

        max_idx = len(accuracys) - (np.argmax(accuracys[::-1])+1)
        best_acc, best_cm = accuracys[max_idx], cm_s[max_idx]

        if not os.path.exists(f'./Results/CMs'):
                os.mkdir('./Results/CMs')

        disp = ConfusionMatrixDisplay(confusion_matrix=best_cm, display_labels=list(range(args.n_classes)))
        disp.plot(cmap='Blues')
        plt.title(f'Matriz de confusion normalizada. Accuracy:{best_acc}')
        plt.ylabel('True Label')
        plt.xlabel('Predicated Label')
        plt.savefig(f'Results/CMs/cm_{file_name}.jpg')
    
    if args.mode == "pretrained":
        ds.main(args.mode, args.n_classes, args.img_size)
        """ pretrained_vit = pretrained_ViT('B_16_imagenet1k', pretrained=True)
        #summary(pretrained_vit.cuda(), (3, 384, 384))

        ### Dataset Load / Generate
        if os.path.exists(f'Datasets/pt_train_{args.n_classes}.pkl'):
            pt_train_dataset = torch.load(f'Datasets/pt_train_{args.n_classes}.pkl')
        else:
            if not os.path.exists(f'./Datasets'):
                os.mkdir('./Datasets')
            ds.main(args.mode, args.n_classes, args.img_size, args.patch_size)
            train_dataset = torch.load(f'Datasets/pt_train_{args.n_classes}.pkl') """