import torch
import torchvision
from torch import nn
from torchsummary import summary
import json
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from utils.train import train_model
import utils.dataset as ds
from utils.evaluate import test_model
from utils.confusion_matrix_display import plot_confusion_matrix
from model.vit_model import ViT as my_ViT
import sys, os

sys.path.append('..')

if __name__ == "__main__":
    
    if not os.path.exists('./Data_Paths'):
        raise Exception('Please remember to run the gen_readable_paths.py before working with this module.')
    
    if not os.path.exists('./Images'):
        raise Exception('Please remember to download the Yoga_82 dataset and create the "./Images" directory before working with this module.')
    
    ### Argument Parse
    parser = argparse.ArgumentParser(description='Main script for ViT proyect.')

    parser.add_argument('--mode',choices=["train", "test", "transfer"], type=str, default="train", help='Mode to run.')
    parser.add_argument('--n_encoders', type=int, default=3, help='Number of encoders')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--img_size', type=int, default=128, help='Image size')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size')
    parser.add_argument('--dim', type=int, default=768, help='Embedding Dimension')
    parser.add_argument('--hidden_size', type=int, default=3072, help='Hidden size')
    parser.add_argument('--n_classes', type=int, default=6, help='Number of classes')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--transfer',choices=["20to82", "6to20to82", " "], type=str, default=' ', help='Batch size')
    parser.add_argument('--checkpoint', type=str, default=' ', help='Path to checkpoint to use.')

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

        checkpoint = args.checkpoint
        state_dict = torch.load(checkpoint)
        del state_dict["labels"]
        model.load_state_dict(state_dict)
        acc, cm = test_model(model, test_dataloader, device, args.n_classes)

        if not os.path.exists(f'./Results/CMs'):
                os.mkdir('./Results/CMs')

        plot_confusion_matrix(cm, acc, list(range(args.n_classes)), file_name)

    if args.mode ==  "transfer":
        if args.transfer == " ":
            print('Please use --transfer to indicate transfer mode.')
        if args.transfer == '20to82':
            #### Transfer 20 to 82  
            model = my_ViT(
                size=args.img_size,
                dim=args.dim,
                hidden_size=args.hidden_size,
                num_patches=(args.img_size//args.patch_size)**2,
                patch_size = args.patch_size,
                n_classes=20,
                num_heads=args.n_heads,
                num_encoders=args.n_encoders,
            )
            
            ### Dataset Load / Generate
            if os.path.exists(f'Datasets/train_{82}_{args.img_size}.pkl'):
                train_dataset = torch.load(f'Datasets/train_{82}_{args.img_size}.pkl')
            else:
                if not os.path.exists(f'./Datasets'):
                    os.mkdir('./Datasets')
                ds.main(args.mode, 82, args.img_size)
                train_dataset = torch.load(f'Datasets/train_{82}_{args.img_size}.pkl')
            
            file_20 = f'20_{args.batch_size}_{args.n_heads}_{args.n_encoders}_{args.img_size}_{args.patch_size}_{args.hidden_size}'

            state_dict = torch.load(f'./checkpoints/{file_20}/checkpoint_epoch30.pth')
            del state_dict["labels"]
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dir_checkpoint = Path(f'./checkpoints/transfer_20_to_82/')
            
            model.load_state_dict(state_dict)
            
            model.change_classification_head(new_class_num = 82)
            
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

            with open(f'Results/curves/curves_transfer_20_to_82.json', 'w') as file:
                json.dump(curves, file)

        if args.transfer == '6to20to82':
            ########### 6 a 20 ###########
            model = my_ViT(
                size=args.img_size,
                dim=args.dim,
                hidden_size=args.hidden_size,
                num_patches=(args.img_size//args.patch_size)**2,
                patch_size = args.patch_size,
                n_classes=6,
                num_heads=args.n_heads,
                num_encoders=args.n_encoders,
            )
            
            ### Dataset Load / Generate
            if os.path.exists(f'Datasets/train_{20}_{args.img_size}.pkl'):
                train_dataset = torch.load(f'Datasets/train_{20}_{args.img_size}.pkl')
            else:
                if not os.path.exists(f'./Datasets'):
                    os.mkdir('./Datasets')
                ds.main(args.mode, 20, args.img_size)
                train_dataset = torch.load(f'Datasets/train_{20}_{args.img_size}.pkl')

            file_6 = f'6_{args.batch_size}_{args.n_heads}_{args.n_encoders}_{args.img_size}_{args.patch_size}_{args.hidden_size}'

            state_dict = torch.load(f'./checkpoints/{file_6}/checkpoint_epoch30.pth')
            del state_dict["labels"]
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dir_checkpoint = Path(f'./checkpoints/6_to_20/')
            
            model.load_state_dict(state_dict)
            
            model.change_classification_head(new_class_num = 20)
            
            model = model.to(memory_format=torch.channels_last)

            model.to(device=device)

            curves = train_model(
                model=model,
                device=device,
                dir_checkpoint=dir_checkpoint,
                epochs=args.epochs,
                batch_size=args.batch_size,
                dataset=train_dataset,)

            ########### 20 a 82 ###########

            ### Dataset Load / Generate
            if os.path.exists(f'Datasets/train_{82}_{args.img_size}.pkl'):
                train_dataset = torch.load(f'Datasets/train_{82}_{args.img_size}.pkl')
            else:
                if not os.path.exists(f'./Datasets'):
                    os.mkdir('./Datasets')
                ds.main(args.mode, 82, args.img_size)
                train_dataset = torch.load(f'Datasets/train_{82}_{args.img_size}.pkl')
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dir_checkpoint = Path(f'./checkpoints/6_to_20_to_82/')

            model.change_classification_head(new_class_num = 82)

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

            with open(f'Results/curves/curves_6_to_20_to_82.json', 'w') as file:
                json.dump(curves, file)
