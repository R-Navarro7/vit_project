import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

class Patch_Embedding(nn.Module):
    def __init__(self, in_channels, dim, patch_size):
        super().__init__()
        self.patch_embedding = nn.Conv2d(in_channels, dim, kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size))

    def forward(self, img: torch.Tensor):
        patch = self.patch_embedding(img)   
        return patch.flatten(2).transpose(1, 2)

class ImageEmbedding(nn.Module):
    def __init__(self, size: int, num_patches: int, dropout: float = 0.1):
        super().__init__()

        self.label_token = nn.Parameter(torch.zeros(1, 1, size))
        self.position = nn.Parameter(torch.rand(1, num_patches + 1, size))

        self.dropout = nn.Dropout(dropout)

    def forward(self, input: torch.Tensor):
        label_token = self.label_token.repeat(
            input.size(0), 1, 1
        )  # batch_size x 1 x output_size
        output = torch.concat([label_token, input], dim=1)

        position_encoding = self.position.repeat(output.size(0), 1, 1)
        return self.dropout(output + position_encoding)


class AttentionHead(nn.Module):
    def __init__(self, size: int):  # size is hidden size
        super().__init__()

        self.query = nn.Linear(size, size)
        self.key = nn.Linear(size, size)
        self.value = nn.Linear(size, size)

    def forward(self, input_tensor: torch.Tensor):
        q, k, v = (
            self.query(input_tensor),
            self.key(input_tensor),
            self.value(input_tensor),
        )

        scale = q.size(1) ** 0.5
        scores = torch.bmm(q, k.transpose(1, 2)) / scale

        scores = F.softmax(scores, dim=-1)

        output = torch.bmm(scores, v)
        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, size: int, num_heads: int):
        super().__init__()

        self.heads = nn.ModuleList([AttentionHead(size) for _ in range(num_heads)])
        self.linear = nn.Linear(size * num_heads, size)

    def forward(self, input_tensor: torch.Tensor):
        s = [head(input_tensor) for head in self.heads]
        s = torch.cat(s, dim=-1)

        output = self.linear(s)
        return output


class Encoder(nn.Module):
    def __init__(self, size: int, hidden_size : int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        self.attention = MultiHeadAttention(size, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(size, hidden_size),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_size, size),
            nn.Dropout(dropout),
        )
        self.norm_attention = nn.LayerNorm(size)
        self.norm_feed_forward = nn.LayerNorm(size)

    def forward(self, input_tensor):
        attn = input_tensor + self.attention(self.norm_attention(input_tensor))
        output = attn + self.feed_forward(self.norm_feed_forward(attn))
        return output
