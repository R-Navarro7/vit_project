import torch

from utils.train import train_model

from model.vit_model import ViT


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = ViT(
        size=192,
        hidden_size=512,
        num_patches=64,
        n_classes=6,
        num_heads=3,
        num_encoders=1,
        epochs=10
    )

    model = model.to(memory_format=torch.channels_last)

    model.to(device=device)
    try:
        curves = train_model(
            model=model,
            device=device,
            epochs=10,
            batch_size=64,
        )
    except torch.cuda.OutOfMemoryError:

        torch.cuda.empty_cache()
        model.use_checkpointing()
        curves = train_model(
            model=model,
            device=device,
            epochs=10,
            batch_size=64,
        )
