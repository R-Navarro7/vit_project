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


def train_model(
    model,
    device,
    dir_checkpoint,
    img_dim,
    epochs, 
    batch_size,
    patch_size,
    n_classes,
    mode = 'train',
    learning_rate: float = 1e-3,
    val_percent: float = 0.2,  
    save_checkpoint: bool = True,
    weight_decay: float = 0.1,
    gradient_clipping: float = 1.0,
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
    grad_scaler = torch.cuda.amp.GradScaler()
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    ### Training Curves
    curves = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0

        cumulative_train_loss = 0
        cumulative_train_corrects = 0
        train_loss_count = 0
        train_acc_count = 0

        print(f'Epoca {epoch} de {epochs}')
        for batch in tqdm(train_loader):

            #print(f'Epoca {epoch} de {epochs}: batch {batch_count}')
            embeddings, labels = batch

            embeddings = embeddings.to(
                device=device,
                dtype=torch.float32
            )

            labels = labels.to(device=device, dtype=torch.long)

            with torch.autocast(
                device.type if device.type != "mps" else "cpu"
            ):

                predictions = model(embeddings)
                loss = criterion(predictions, labels)  ### train loss

                # Store train loss
                curves["train_loss"].append(loss)

            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            grad_scaler.step(optimizer)
            grad_scaler.update()

            global_step += 1
            epoch_loss += loss.item()

            cumulative_train_loss += loss.item()
            train_loss_count += 1
            train_acc_count += labels.shape[0]

            # Accuracy
            class_prediction = torch.argmax(predictions, axis=1).long()
            cumulative_train_corrects += (labels == class_prediction).sum().item()

        ### Train Metrics
        train_loss = cumulative_train_loss / train_loss_count
        train_acc = cumulative_train_corrects / train_acc_count

        ### Evaluation Round
        val_loss, val_acc = evaluate(
            model, val_loader, device, criterion
        )  ### validation loss
        print('Validation Ended :D')
        print(f"Val loss: {val_loss}, Val acc: {val_acc}")
        scheduler.step(val_loss)

        ### Store val_loss value everytime the model it's evaluated
        curves["train_acc"].append(train_acc)
        curves["train_loss"].append(train_loss)
        curves["val_acc"].append(val_acc)
        curves["val_loss"].append(val_loss)


        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict["labels"] = dataset.y
            torch.save(
                state_dict, str(dir_checkpoint / "checkpoint_epoch{}.pth".format(epoch))
            )
            logging.info(f"Checkpoint {epoch} saved!")
        
        print('\n-----------------------------------------------------------------------------------------\n')

    return curves
