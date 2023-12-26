import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix 

@torch.inference_mode()
def evaluate(net, dataloader, device, criterion):

    cumulative_loss = 0
    cumulative_predictions = 0
    data_count = 0

    net.eval()
    print('Validation Time!')

    for batch in tqdm(dataloader):

        embeddings, labels = batch

        # move embeddingss and labels to correct device and type
        embeddings = embeddings.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.long)

        # predict the mask
        predictions = net(embeddings)
        val_loss = criterion(predictions, labels)

        class_prediction = torch.argmax(predictions, axis=1).long()

        cumulative_predictions += (labels == class_prediction).sum()
        cumulative_loss += val_loss
        data_count += labels.shape[0]

    val_acc = cumulative_predictions / data_count
    val_loss = cumulative_loss / len(dataloader)
    return val_loss, val_acc

@torch.inference_mode()
def test_model(net, dataloader, device, n_classes):
    cumulative_predictions = 0
    data_count = 0

    net.eval()
    net.to(device)

    all_labels = []
    all_predictions = []

    for batch in tqdm(dataloader):

        embeddings, labels = batch

        # move embeddingss and labels to correct device and type
        embeddings = embeddings.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.long)

        # predict the mask
        predictions = net(embeddings)

        class_prediction = torch.argmax(predictions, axis=1).long()

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(class_prediction.cpu().numpy())

        cumulative_predictions += (labels == class_prediction).sum()
        data_count += labels.shape[0]

    cm_val = confusion_matrix(all_labels, all_predictions, normalize='true', labels=list(range(n_classes)))

    val_acc = cumulative_predictions / data_count

    return val_acc.cpu().detach().item(), cm_val