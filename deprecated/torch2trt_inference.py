from torch2trt import TRTModule
import torch
import torchvision
from torch import nn
from PIL import Image
from torchvision import transforms
import argparse

def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--pt', type=str, help='pt path')
    parser.add_argument('--source', type=str, help='image path')
    parser.add_argument('--imgsz', nargs='+', type=int, default=[320,263], help='inference size h,w')

    opt = parser.parse_args()
    return opt

def main(opt):

    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(opt['pt']))

    img = Image.open(opt['source'])

    HEIGHT , WIDTH = opt['imgsz']
    tfm = transforms.Compose([
        transforms.Resize((HEIGHT,WIDTH)),
        transforms.ToTensor(),
    ])

    img = tfm(img).cuda()

    pred = model_trt(img).to("cpu")
    softmax = nn.Softmax(dim=1)
    pred = softmax(pred)

    label = torch.argmax(pred)
    conf = torch.max(pred)

    class_dic = {
        0 : "daisy",
        1 : "dandelion",
        2 : "roses",
        3 : "sunflowers",
        4 : "tulips"
    }


    print(class_dic[label] , conf)

if __name__ == '__main__':
   opt = parse_opt()
   main(vars(opt))