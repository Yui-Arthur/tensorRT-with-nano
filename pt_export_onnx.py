import os
import numpy as np
import torch
from PIL import Image
import torchvision
import torch.onnx


if __name__ == '__main__':
    model = torchvision.models.resnet50(weights = "ResNet50_Weights.IMAGENET1K_V2")
    model.eval()
    # Input to the model
    batch_size = 1
    x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
    torch_out = model(x)

    # Export the model
    torch.onnx.export(model,               # model being run
                    x,                         # model input (or a tuple for multiple inputs)
                    "resnet50.onnx",   # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=10,          # the ONNX version to export the model to
                    do_constant_folding=True,  # whether to execute constant folding for optimization
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})