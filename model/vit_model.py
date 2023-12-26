import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import math

from .vit_parts import Patch_Embedding, ImageEmbedding, Encoder


class ViT(nn.Module):

    def __init__(self, size: int, dim : int, hidden_size: int, num_patches: int, n_classes: int, 
                num_heads: int, patch_size : int, num_encoders: int, emb_dropout: float = 0.1,
                dropout: float = 0.1, freeze = False):
        super().__init__()
        
        self.n_classes = n_classes
        self.freeze = freeze

        ### Patchify Transform
        self.patchify = Patch_Embedding(3, dim, patch_size)

        ### Embeddings
        self.embedding = ImageEmbedding(dim, num_patches, dropout=emb_dropout)

        ### Encoder Array 
        encoder_list = [Encoder(dim, hidden_size, num_heads, dropout=dropout) for encoder in range(num_encoders)]
        self.encoders = nn.Sequential(
            *encoder_list,
        )

        ### Classifier (Feed Forward)
        self.mlp_head = nn.Linear(dim, n_classes)
    
    def forward(self, input_tensor: torch.Tensor):  
        patch = self.patchify(input_tensor)
        if self.freeze:
            with torch.no_grad():
                emb = self.embedding(patch)   
                attn = self.encoders(emb)   
        else:
            emb = self.embedding(patch)   
            attn = self.encoders(emb)
        
        return self.mlp_head(attn[:, 0, :])    


