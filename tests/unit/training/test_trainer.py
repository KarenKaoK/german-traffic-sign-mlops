import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.trainer import train_one_batch, train_one_epoch, evaluate_one_epoch


def test_train_one_batch_returns_loss():

    # arrange
    device = torch.device("cpu")

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 43),
    ).to(device)

    images = torch.randn(100, 3, 32, 32)

    labels = torch.randint(0, 43, (100,))

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001,
    )

    # act
    loss = train_one_batch(
        model=model,
        images=images,
        labels=labels,
        criterion=criterion,
        optimizer=optimizer,
    )

    # assert
    assert isinstance(loss, float)
    assert loss >= 0


def test_train_one_batch_update_model_weight():

    # arrange
    device = torch.device("cpu")

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 43),
    ).to(device)

    images = torch.randn(100, 3, 32, 32)

    labels = torch.randint(0, 43, (100,))

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001,
    )

    before_weights = []

    for param in model.parameters():
        copied_param = param.clone().detach()
        before_weights.append(copied_param)

    # act
    train_one_batch(
        model=model,
        images=images,
        labels=labels,
        criterion=criterion,
        optimizer=optimizer,
    )

    # assert
    after_weights = list(model.parameters())

    assert any(
        not torch.equal(before, after)
        for before, after in zip(before_weights, after_weights)
    )


def test_train_one_epoch_return_loss():

    # arrange
    device = torch.device("cpu")
    images = torch.randn(100, 3, 32, 32)
    labels = torch.randint(0, 43, (100,))

    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 43),
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001,
    )

    # act
    epoch_loss = train_one_epoch(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    # assert
    assert isinstance(epoch_loss, float)
    assert epoch_loss >= 0


def test_evaluate_one_epoch_returns_loss():

    # arrange
    device = torch.device("cpu")
    images = torch.randn(100, 3, 32, 32)
    labels = torch.randint(0, 43, (100,))

    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 43),
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    # act
    val_loss = evaluate_one_epoch(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
    )

    # assert
    assert isinstance(val_loss, float)
    assert val_loss >= 0


def test_evaluate_one_epoch_does_not_update_model_weights():

    # arrange
    device = torch.device("cpu")
    images = torch.randn(100, 3, 32, 32)
    labels = torch.randint(0, 43, (100,))

    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 43),
    ).to(device)

    before_weights = []

    for param in model.parameters():
        before_weights.append(param.clone().detach())

    criterion = nn.CrossEntropyLoss()

    # act
    evaluate_one_epoch(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
    )
    after_weights = list(model.parameters())

    # assert
    assert all(
        torch.equal(before, after)
        for before, after in zip(before_weights, after_weights)
    )
