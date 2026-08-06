import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Optimizer

def train_one_batch(
        model: nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
        criterion: nn.Module,
        optimizer: Optimizer,
)-> float:

    model.train()

    optimizer.zero_grad()

    outputs = model(images)
    loss = criterion(outputs,labels)

    loss.backward()
    optimizer.step()

    return loss.item()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
)-> float:

    total_loss = 0.0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        batch_loss = train_one_batch(
            model=model,
            images=images,
            labels=labels,
            criterion=criterion,
            optimizer=optimizer
        )

        batch_size = images.size(0)

        total_loss += batch_loss * batch_size
        total_samples += batch_size

    return total_loss / total_samples


if __name__ == "__main__":

    device = torch.device("cpu")

    images = torch.randn(100,3,32,32)

    labels = torch.randint(0,43, (100,))

    dataset = TensorDataset(images, labels)

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True
    )

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3*32*32,43),
        
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001,
    )

    epoch_loss = train_one_epoch(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    print(f" epoch loss: {epoch_loss}")
