import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
import os

def gen_curves(path, name):
    
    with open(path, 'r') as file:
        curves = json.load(file)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    fig.set_facecolor('white')

    epochs = np.arange(len(curves["val_loss"])) + 1

    ax[0].plot(epochs, curves["val_loss"], label='validation')
    ax[0].plot(len(curves["val_loss"])*np.array(range(len(curves["train_loss"])))/len(curves["train_loss"]), curves["train_loss"], label='training')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Loss evolution during training')
    ax[0].legend()

    ax[1].plot(epochs, curves["val_acc"], label='validation')
    ax[1].plot(epochs*np.array(range(len(curves["train_acc"])))/len(curves["train_acc"]), curves["train_acc"], label='training')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Accuracy evolution during training')
    ax[1].legend()

    if not os.path.exists(f'./Results/plots'):
            os.mkdir('./Results/plots')
    plt.savefig(f'./Results/plots/{name}.png')

    #[tensor.cpu().detach().item() for tensor in

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Curve generator.')
    parser.add_argument('--path', type=str, default="", help='Path to curve json.')
    parser.add_argument('--name', type=str, default="", help='Nombre del grafico.')
    args = parser.parse_args()
    
    gen_curves(args.path, args.name)
