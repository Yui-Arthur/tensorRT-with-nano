from torch2trt import TRTModule
import torch
import torchvision
from torch import nn
from PIL import Image
from torchvision import transforms

model_trt = TRTModule()

model_trt.load_state_dict(torch.load('resnet50_trt.pth'))



input_file_path = "hen.jpg"
img = Image.open(input_file_path)

tfm = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

img = tfm(img).cuda()

pred = model_trt(img).to("cpu")
softmax = nn.Softmax(dim=1)
pred = softmax(pred)
# print(pred)
label = torch.argmax(pred)
conf = torch.max(pred)
print(label , conf)