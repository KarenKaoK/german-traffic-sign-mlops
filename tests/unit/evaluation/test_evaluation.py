import torch
import torch.nn as nn
from src.evaluation.evaluate import evaluate_model
from torch.utils.data import DataLoader, TensorDataset


def test_evaluate_model_return_accuracy():

    # arrange
    device = torch.device("cpu")

    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, 43),
    ).to(device)

    images = torch.randn(40, 3, 32, 32)
    labels = torch.randint(0, 43, (40,))

    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

    # act
    accuracy = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
    )

    # assert
    assert 0 <= accuracy <= 1
