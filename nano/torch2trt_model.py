"""## torch2trt"""



import torch
from torch2trt import torch2trt
import torchvision


# create some regular pytorch model...
model = torchvision.models.resnet50(pretrained=True)
model.eval().cuda()
# Input to the model
batch_size = 1
x = torch.randn(batch_size, 3, 224, 224, requires_grad=True).cuda()

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x])
print("OK")
torch.save(model_trt.state_dict(), 'resnet50_trt.pth')

