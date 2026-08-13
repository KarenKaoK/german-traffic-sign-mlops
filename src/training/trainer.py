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
) -> float:

    model.train()

    optimizer.zero_grad()

    outputs = model(images)
    loss = criterion(outputs, labels)

    loss.backward()
    optimizer.step()

    return loss.item()


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> float:

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
            optimizer=optimizer,
        )

        batch_size = images.size(0)

        total_loss += batch_loss * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def evaluate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:

    model.eval()

    total_loss = 0.0
    total_samples = 0
    total_correct = 0

    with torch.no_grad():

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            predictions = torch.argmax(outputs, dim=1)
            correct = (predictions == labels).sum().item()
            total_correct += correct

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size
    return (total_loss / total_samples, total_correct / total_samples)


def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optimizer,
    num_epoch: int,
    checkpoint_path: str,
) -> tuple[list[float], list[float], list[float]]:

    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float("inf")

    for epoch in range(num_epoch):

        train_epoch_loss = train_one_epoch(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_epoch_loss, val_epoch_acc = evaluate_one_epoch(
            model=model,
            dataloader=val_dataloader,
            criterion=criterion,
            device=device,
        )

        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            torch.save(model.state_dict(), checkpoint_path)

        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)

    return train_losses, val_losses, val_accuracies


if __name__ == "__main__":

    device = torch.device("cpu")

    images = torch.randn(100, 3, 32, 32)

    labels = torch.randint(0, 43, (100,))

    dataset = TensorDataset(images, labels)

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 43),
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

    print("---")

    checkpoint_path = "artifacts/best_model.pt"

    # fake train data
    train_images = torch.randn(100, 3, 32, 32)
    train_labels = torch.randint(0, 43, (100,))

    train_dataset = TensorDataset(train_images, train_labels)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
    )

    # fake validation data
    val_images = torch.randn(40, 3, 32, 32)
    val_labels = torch.randint(0, 43, (40,))

    val_dataset = TensorDataset(val_images, val_labels)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
    )

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 43),
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001,
    )

    train_losses, val_losses, val_accuracies = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        device=device,
        optimizer=optimizer,
        num_epoch=5,
        checkpoint_path=checkpoint_path,
    )

    print("train losses:", train_losses)
    print("val losses:", val_losses)
    print("val accuracies:", val_accuracies)
