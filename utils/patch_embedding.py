import torch

class PatchTransform():
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, img: torch.Tensor):
        output = img.unfold(1, self.patch_size, self.patch_size)
        output = img.unfold(2, self.patch_size, self.patch_size)

        return output.reshape(-1, self.patch_size * self.patch_size * 3)    