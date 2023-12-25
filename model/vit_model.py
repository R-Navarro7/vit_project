import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import math

from .vit_parts import ImageEmbedding, Encoder

class ViT(nn.Module):

    def __init__(self, size: int, hidden_size: int, num_patches: int, n_classes: int, num_heads: int,
                num_encoders: int, emb_dropout: float = 0.1, dropout: float = 0.1, 
                freeze = False):
        super().__init__()
        
        self.n_classes = n_classes
        self.freeze = freeze

        ### Embeddings
        self.embedding = ImageEmbedding(size, hidden_size, num_patches, dropout=emb_dropout)

        ### Encoder Array 
        encoder_list = [Encoder(hidden_size, num_heads, dropout=dropout) for encoder in range(num_encoders)]
        self.encoders = nn.Sequential(
            *encoder_list,
        )

        ### Classifier (Feed Forward)
        self.mlp_head = nn.Linear(hidden_size, n_classes)
    
    def forward(self, input_tensor: torch.Tensor):  

        if self.freeze:
            with torch.no_grad():
                emb = self.embedding(input_tensor)   
                attn = self.encoders(emb)   
        else:
            emb = self.embedding(input_tensor)   
            attn = self.encoders(emb)
        
        return self.mlp_head(attn[:, 0, :])    

