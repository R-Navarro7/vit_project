import torch
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

        cumulative_predictions += (labels == class_prediction).sum().item()
        cumulative_loss += val_loss.item()
        data_count += labels.shape[0]

    val_acc = cumulative_predictions / data_count
    val_loss = cumulative_loss / len(dataloader)
    return val_loss, val_acc