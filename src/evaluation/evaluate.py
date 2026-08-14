import torch
import torch.nn as nn
from torch.utils.data import DataLoader



def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:

    model.eval()

    total_samples = 0
    total_correct = 0

    with torch.no_grad():

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            correct = (predictions == labels).sum().item()
            total_correct += correct

            batch_size = images.size(0)
            total_samples += batch_size

    return total_correct / total_samples
