import torch 
from torchvision.transforms import v2

def build_transforms():
    return v2.Compose(
        [
        v2.Resize((32,32)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            ),
        ]
    )