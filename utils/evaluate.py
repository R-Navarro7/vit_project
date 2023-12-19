import torch
from tqdm import tqdm


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, criterion):
    net.eval()
    print('Validation Time!')
    batch_count=0
    for batch in dataloader:
        batch_count += 1
        print(f'Batch {batch_count}')
        embeddings, labels = batch

        # move embeddingss and labels to correct device and type
        embeddings = embeddings.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.long)

        # predict the mask
        prediction = net(embeddings)

        ### compute the "Normal" loss
        val_loss = criterion(prediction, labels)

    net.train()
    return val_loss