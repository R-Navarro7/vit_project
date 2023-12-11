import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

from vit_parts import ImageEmbedding, Encoder

class ViT(nn.Module):

    def __init__(self, size: int, hidden_size: int, num_patches: int, num_classes: int, num_heads: int,
                 num_encoders: int, emb_dropout: float = 0.1, dropout: float = 0.1,
                 lr: float = 1e-4, min_lr: float = 4e-5,
                 weight_decay: float = 0.1, epochs: int = 200):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.min_lr = min_lr
        self.weight_decay = weight_decay
        self.epochs = epochs

        self.embedding = ImageEmbedding(size, hidden_size, num_patches, dropout=emb_dropout)

        self.encoders = nn.Sequential(
            *[Encoder(hidden_size, num_heads, dropout=dropout) for _ in range(num_encoders)],
        )
        self.mlp_head = nn.Linear(hidden_size, num_classes)
    
    def forward(self, input_tensor: torch.Tensor):  
        emb = self.embedding(input_tensor)   
        attn = self.encoders(emb)   
        return self.mlp_head(attn[:, 0, :]) 
    