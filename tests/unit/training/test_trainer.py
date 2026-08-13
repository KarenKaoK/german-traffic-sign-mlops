import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from unittest.mock import patch

from src.training.trainer import (
    train_one_batch,
    train_one_epoch,
    evaluate_one_epoch,
    train_model,
)


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


def test_evaluate_one_epoch_returns_loss_and_accuracy():

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
    val_loss, val_acc = evaluate_one_epoch(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
    )

    # assert
    assert isinstance(val_loss, float)
    assert isinstance(val_acc, float)
    assert val_loss >= 0
    assert 1 >= val_acc >= 0


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


def test_train_model_return_losses_for_each_epoch():

    # arrange

    device = torch.device("cpu")

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 43),
    ).to(device)

    train_images = torch.randn(100, 3, 32, 32)
    train_labels = torch.randint(0, 43, (100,))

    train_dataset = TensorDataset(train_images, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False)

    val_images = torch.randn(40, 3, 32, 32)
    val_labels = torch.randint(0, 43, (40,))

    val_dataset = TensorDataset(val_images, val_labels)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001,
    )

    checkpoint_path = "artifacts/test_best_model.pt"

    # act
    train_losses, val_losses, val_accuracies = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        criterion=criterion,
        device=device,
        optimizer=optimizer,
        num_epoch=3,
        checkpoint_path=checkpoint_path,
    )

    # assert
    assert len(train_losses) == 3
    assert len(val_losses) == 3
    assert len(val_accuracies) == 3

    assert all(isinstance(loss, float) for loss in train_losses)
    assert all(isinstance(loss, float) for loss in val_losses)
    assert all(isinstance(acc, float) for acc in val_accuracies)
    assert all(0 <= acc <= 1 for acc in val_accuracies)


def test_train_model_saves_checkpoint_when_val_loss_improves():

    # arrange
    device = torch.device("cpu")

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 43),
    ).to(device)

    train_images = torch.randn(100, 3, 32, 32)
    train_labels = torch.randint(0, 43, (100,))

    train_dataset = TensorDataset(train_images, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False)

    val_images = torch.randn(40, 3, 32, 32)
    val_labels = torch.randint(0, 43, (40,))

    val_dataset = TensorDataset(val_images, val_labels)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001,
    )

    checkpoint_path = "fake_best_model.pt"

    with patch("src.training.trainer.train_one_epoch") as mock_train, patch(
        "src.training.trainer.evaluate_one_epoch"
    ) as mock_evaluate, patch("src.training.trainer.torch.save") as mock_save:

        mock_train.return_value = 0.7
        mock_evaluate.side_effect = [
            (1.0, 0.5),
            (0.8, 0.6),
            (0.9, 0.7),
        ]

        # act
        train_losses, val_losses, val_accuracies = train_model(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            num_epoch=3,
            checkpoint_path=checkpoint_path,
        )

        # assert
        assert mock_save.call_count == 2
