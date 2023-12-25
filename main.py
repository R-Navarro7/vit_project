import torch
from pathlib import Path

from utils.train import train_model

from model.vit_model import ViT

N_ENCODERS = 1
N_HEADS = 3
IMG_SIZE =64
PATCH_SIZE = 8
HIDDEN_SIZE = 512
N_PATCHES = (IMG_SIZE//PATCH_SIZE)**2
N_CLASSES = 6
EPOCHS = 10
BATCH_SIZE = 16


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = 'test'
    dir_checkpoint = Path(f"./checkpoints/{checkpoint}/")

    print(f"Using: {device}")
    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = ViT(
        size=3*(PATCH_SIZE**2),
        hidden_size=HIDDEN_SIZE,
        num_patches=N_PATCHES,
        n_classes=N_CLASSES,
        num_heads=N_HEADS,
        num_encoders=N_ENCODERS,
    )

    model = model.to(memory_format=torch.channels_last)

    model.to(device=device)
    try:
        curves = train_model(
            model=model,
            device=device,
            dir_checkpoint=dir_checkpoint,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            patch_size=PATCH_SIZE,
            img_dim=IMG_SIZE,
            n_classes=N_CLASSES,
        )
    except torch.cuda.OutOfMemoryError:

        torch.cuda.empty_cache()
        model.use_checkpointing()
        curves = train_model(
            model=model,
            device=device,
            dir_checkpoint=dir_checkpoint,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            patch_size=PATCH_SIZE,
            img_dim=IMG_SIZE,
            n_classes=N_CLASSES,
        )
