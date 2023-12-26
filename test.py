import torch
import torchvision
import torchaudio
import numpy as np

cifar_trainset = torchvision.datasets.CIFAR10(root='./CIFAR', train=True, download=True, transform=None)
