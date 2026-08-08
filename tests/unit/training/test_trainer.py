import torch
import torch.nn as nn

from src.training.trainer import train_one_batch


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
