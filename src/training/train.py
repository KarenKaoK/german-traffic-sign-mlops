import torch
import torch.nn as nn
from src.data.dataset import GTSRBDataset
from src.preprocess.transforms import build_transforms
from src.data.split_dataset import split_train_val
from src.data.dataloader import build_dataloader
from src.models.model import CNN
from src.training.trainer import train_model

batch_size = 32
learning_rate = 0.0001
num_epochs = 5

device = torch.device("cpu")

transform = build_transforms()

dataset = GTSRBDataset(
    manifest_path="data/processed/train_manifest.csv",
    transform=transform
)

train_dataset, val_dataset = split_train_val(
    dataset=dataset,
    val_ratio=0.2,
    seed=42,
)

train_dataloader = build_dataloader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)

val_dataloader = build_dataloader(
    dataset=val_dataset,
    batch_size=batch_size,
    shuffle=False
)

model = CNN(in_channels=3, num_classes=43,).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
)

train_losses, val_losses = train_model(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    criterion=criterion,
    device=device,
    optimizer=optimizer,
    num_epoch=num_epochs,
)

print(f"train_losses: {train_losses}")
print(f"val_losses: {val_losses}")