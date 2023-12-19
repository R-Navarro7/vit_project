import logging
import os
import torch
import torch.nn as nn
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from .dataset import DatitaSet
from .evaluate import evaluate

dir_checkpoint = Path("./checkpoints/")


def train_model(
    model,
    device,
    mode = 'train',
    epochs: int = 200,  
    batch_size: int = 64,
    learning_rate: float = 1e-5,
    val_percent: float = 0.2,  
    save_checkpoint: bool = True,
    amp: bool = False,
    weight_decay: float = 1e-8,
    gradient_clipping: float = 1.0,
    n_classes=6,
    img_dim=64,
    patch_size=8,
):
    # Create dataset
    dataset = DatitaSet(n_classes=n_classes, img_dim=img_dim, patch_size=patch_size, mode=mode) 

    # Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0)
    )

    # Create data loaders
    loader_args = dict(
        batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    # Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        foreach=True,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=10
    )  
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    ### Training Curves
    curves = {
        "train_loss": [],
        "val_loss": [],
    }

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        #with tqdm(total=n_train, desc=f"Epoch {epoch}/{epochs}", unit="img") as pbar:
        batch_count = 0
        for batch in train_loader:
            batch_count += 1
            print(f'Epoca {epoch} de {epochs}: batch {batch_count}')
            embeddings, labels = batch

            embeddings = embeddings.to(
                device=device,
                dtype=torch.float32
            )

            labels = labels.to(device=device, dtype=torch.long)

            with torch.autocast(
                device.type if device.type != "mps" else "cpu", enabled=amp
            ):
                masks_pred = model(embeddings)

                loss = criterion(masks_pred, labels)  ### train loss

                # Store train loss
                curves["train_loss"].append(loss)

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            global_step += 1
            epoch_loss += loss.item()


            # Evaluation round
            division_step = n_train // (5 * batch_size)
            if division_step > 0:
                if global_step % division_step == 0:
                    val_loss = evaluate(
                        model, val_loader, device, amp, criterion
                    )  ### validation loss
                    print('Validation End :D')
                    scheduler.step(val_loss)
                    ### Store val_loss value everytime the model it's evaluated
                    curves["val_loss"].append(val_loss)


        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict["labels"] = dataset.y
            torch.save(
                state_dict, str(dir_checkpoint / "checkpoint_epoch{}.pth".format(epoch))
            )
            logging.info(f"Checkpoint {epoch} saved!")
    return curves
