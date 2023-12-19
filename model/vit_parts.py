import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F


class ImageEmbedding(nn.Module):
    def __init__(self, size: int, hidden_size:int, num_patches: int, dropout: float = 0.2):
        super().__init__()

        self.projection = nn.Linear(size, hidden_size)
        self.label_token = nn.Parameter(torch.rand(1, hidden_size))
        self.position = nn.Parameter(torch.rand(1, num_patches + 1, hidden_size))

        self.dropout = nn.Dropout(dropout)

    def forward(self, input: torch.Tensor):
        output = self.projection(input)

        label_token = self.label_token.repeat(
            output.size(0), 1, 1
        )  # batch_size x 1 x output_size
        output = torch.concat([label_token, output], dim=1)

        position_encoding = self.position.repeat(output.size(0), 1, 1)
        return self.dropout(output + position_encoding)


class AttentionHead(nn.Module):
    def __init__(self, size: int):  # size is hidden size
        super(AttentionHead, self).__init__()

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

        # 8 x 64 x 64 @ 8 x 64 x 48 = 8 x 64 x 48
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
    def __init__(self, size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        self.attention = MultiHeadAttention(size, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(size, 4 * size),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(4 * size, size),
            nn.Dropout(dropout),
        )
        self.norm_attention = nn.LayerNorm(size)
        self.norm_feed_forward = nn.LayerNorm(size)

    def forward(self, input_tensor):
        attn = input_tensor + self.attention(self.norm_attention(input_tensor))
        output = attn + self.feed_forward(self.norm_feed_forward(attn))
        return output
