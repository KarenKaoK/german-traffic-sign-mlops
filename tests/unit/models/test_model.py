import torch

from src.models.model import CNN

def test_CNN_return_excepted_output_shape():

    # arrange
    inputs = torch.randn(4,3,32,32)
    model = CNN(in_channels=3,num_classes=43,)


    # act
    
    outputs = model(inputs)

    # assert 
    assert outputs.shape == (4,43)
